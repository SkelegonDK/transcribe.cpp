#!/usr/bin/env python3
"""
convert-whisper_qwen.py - convert a WhisperQwen checkpoint (e.g.
syvai/hviske-v6) to a GGUF that transcribe.cpp's loader can ingest.
Preserves the source BF16 dtype; block quantization goes through
tools/transcribe-quantize.

Source format:
    HuggingFace directory with config.json (model_type "whisper_qwen",
    nested `whisper` + `qwen` sub-configs), vocab.json + merges.txt +
    tokenizer_config.json (Qwen2Tokenizer), and model.safetensors (BF16).

Architecture: audio-LLM with prefix injection.

    16 kHz -> Whisper log-mel(128)
           -> Whisper encoder (2x Conv1d stem + 32 pre-LN blocks). The
              absolute sinusoid table is zeroed in the checkpoint; q/k are
              rotated by frame index instead (partial NeoX RoPE over the
              first `rotary_dim` dims of each head).
           -> concatenate `stack` consecutive frames (one token / 80 ms)
           -> RMSNorm(d_model * stack) -> Linear(-> decoder hidden)
           -> Qwen3 causal LM, prompt = [audio][style controls][BOS]

Safetensors prefixes:

    encoder.*            -> enc.*      (Whisper encoder)
    norm.weight          -> adapter.norm_stack.weight
    proj.weight          -> adapter.proj.weight
    llm.model.*          -> dec.*      (Qwen3 LM)
    llm.lm_head.weight   -> skipped (verified bitwise equal to embed_tokens)
    encoder.embed_positions.weight -> skipped (verified all zeros)

Layout conversions: NONE. Linear weights are PyTorch (out, in) and Conv1d
kernels are (out, in, k), both of which ggml consumes as-is.

CLI:

    uv run --project scripts/envs/whisper_qwen \
      scripts/convert-whisper_qwen.py syvai/hviske-v6 \
      --revision 323b5ffdf54ef22196340a0b37cb11e668fefc5a

    uv run --project scripts/envs/whisper_qwen \
      scripts/convert-whisper_qwen.py models/hviske-v6 out.gguf \
      --repo-id syvai/hviske-v6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import download_snapshot, looks_like_repo_id  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

ARCH = "whisper_qwen"
REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16
REFERENCE_GGML_TYPE = GGMLQuantizationType.BF16

# Known checkpoints. Identity metadata is per-publisher, so unknown repos
# fall back to the generic block below and should be added here.
KNOWN_VARIANTS = {
    "hviske-v6": {
        "name": "Hviske v6",
        "basename": "hviske",
        "author": "syv.ai",
        "organization": "syvai",
        "license": "cc-by-nc-4.0",
        "license_name": "Creative Commons Attribution-NonCommercial 4.0",
        "license_link": "https://creativecommons.org/licenses/by-nc/4.0/",
    },
}

# Style-control tokens (input-only conditioning; never prediction targets).
STYLE_TOKENS = ("<|nocase|>", "<|cased|>", "<|nopunc|>", "<|punc|>")


# ---------------------------------------------------------------------------
# Tokenizer (Qwen2Tokenizer: byte-level BPE, tagged "gpt2" per llama.cpp)
# ---------------------------------------------------------------------------


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    with (model_dir / "vocab.json").open(encoding="utf-8") as f:
        base_vocab: dict[str, int] = json.load(f)
    with (model_dir / "merges.txt").open(encoding="utf-8") as f:
        merges = [line.rstrip("\n") for line in f.readlines()]
        if merges and merges[0].startswith("#"):
            merges = merges[1:]
    with (model_dir / "tokenizer_config.json").open(encoding="utf-8") as f:
        tokcfg = json.load(f)

    tok_by_id: dict[int, tuple[str, bool]] = {
        int(tid): (tok, False) for tok, tid in base_vocab.items()
    }
    for tid_str, info in (tokcfg.get("added_tokens_decoder") or {}).items():
        tok_by_id[int(tid_str)] = (info["content"], bool(info.get("special", False)))

    if max(tok_by_id) + 1 > vocab_size:
        raise ValueError(f"tokenizer id {max(tok_by_id)} >= vocab_size {vocab_size}")

    tokens: list[str] = []
    types: list[int] = []
    for i in range(vocab_size):
        if i not in tok_by_id:
            # Embedding rows Qwen allocates past the tokenizer. Placeholder
            # names keep detokenization uniform; they are never generated.
            tokens.append(f"<|unused_{i}|>")
            types.append(TOKEN_TYPE_CONTROL)
            continue
        tok, is_special = tok_by_id[i]
        tokens.append(tok)
        types.append(TOKEN_TYPE_CONTROL if is_special else TOKEN_TYPE_NORMAL)

    ids = {tok: i for i, (tok, _) in tok_by_id.items()}
    return {"tokens": tokens, "types": types, "merges": merges, "ids": ids}


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict) -> dict:
    if config.get("model_type") != "whisper_qwen":
        raise ValueError(f"expected model_type whisper_qwen, got {config.get('model_type')!r}")
    w = config["whisper"]
    q = config["qwen"]

    d_model = int(w["d_model"])
    n_heads = int(w["encoder_attention_heads"])
    head_dim = d_model // n_heads
    partial = float(config.get("partial_rotary_factor", 0.875))
    # Mirrors WhisperQwenConfig.rotary_dim: even number of rotated dims.
    rotary_dim = (int(head_dim * partial) // 2) * 2

    if not bool(config.get("tie_word_embeddings", True)) or not bool(q.get("tie_word_embeddings", True)):
        raise ValueError("untied lm_head is not supported by this converter")
    if q.get("rope_scaling"):
        raise ValueError(f"unsupported decoder rope_scaling {q['rope_scaling']!r}")
    if str(w.get("activation_function", "gelu")).lower() != "gelu":
        raise ValueError("encoder activation must be gelu")
    if str(q.get("hidden_act", "silu")).lower() not in ("silu", "swish"):
        raise ValueError("decoder hidden_act must be silu")

    return {
        "enc_n_layers": int(w["encoder_layers"]),
        "enc_d_model": d_model,
        "enc_n_heads": n_heads,
        "enc_ffn_dim": int(w["encoder_ffn_dim"]),
        "enc_n_mels": int(w["num_mel_bins"]),
        "enc_max_src_pos": int(w["max_source_positions"]),
        "enc_rotary_dim": rotary_dim,
        "enc_rope_theta": float(config.get("rope_theta", 10000.0)),
        "stack": int(config["stack"]),
        "adapter_rms_eps": 1e-6,  # Qwen3RMSNorm(stacked, eps=1e-6) in modeling
        "dec_n_layers": int(q["num_hidden_layers"]),
        "dec_hidden": int(q["hidden_size"]),
        "dec_intermediate": int(q["intermediate_size"]),
        "dec_n_heads": int(q["num_attention_heads"]),
        "dec_n_kv_heads": int(q["num_key_value_heads"]),
        "dec_head_dim": int(q.get("head_dim", int(q["hidden_size"]) // int(q["num_attention_heads"]))),
        "dec_rms_eps": float(q["rms_norm_eps"]),
        "dec_rope_theta": float(q["rope_theta"]),
        "dec_max_pos": int(q["max_position_embeddings"]),
        "dec_vocab": int(q["vocab_size"]),
        "bos_id": int(config["bos_token_id"]),
        "eos_id": int(config["eos_token_id"]),
        "pad_id": int(config["pad_token_id"]),
        "style_tokens": dict(config.get("style_tokens") or {}),
    }


# ---------------------------------------------------------------------------
# Tensor tables
# ---------------------------------------------------------------------------

ENC_TOP = [
    ("encoder.conv1.weight", "enc.conv.0.weight"),
    ("encoder.conv1.bias", "enc.conv.0.bias"),
    ("encoder.conv2.weight", "enc.conv.1.weight"),
    ("encoder.conv2.bias", "enc.conv.1.bias"),
    ("encoder.layer_norm.weight", "enc.final_norm.weight"),
    ("encoder.layer_norm.bias", "enc.final_norm.bias"),
]

# Whisper attention: q/v/out carry bias, k does not.
ENC_BLOCK = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn_layer_norm.bias", "norm_attn.bias"),
    ("self_attn.q_proj.weight", "attn.q.weight"),
    ("self_attn.q_proj.bias", "attn.q.bias"),
    ("self_attn.k_proj.weight", "attn.k.weight"),
    ("self_attn.v_proj.weight", "attn.v.weight"),
    ("self_attn.v_proj.bias", "attn.v.bias"),
    ("self_attn.out_proj.weight", "attn.out.weight"),
    ("self_attn.out_proj.bias", "attn.out.bias"),
    ("final_layer_norm.weight", "norm_ffn.weight"),
    ("final_layer_norm.bias", "norm_ffn.bias"),
    ("fc1.weight", "ffn.fc1.weight"),
    ("fc1.bias", "ffn.fc1.bias"),
    ("fc2.weight", "ffn.fc2.weight"),
    ("fc2.bias", "ffn.fc2.bias"),
]

ADAPTER = [
    ("norm.weight", "adapter.norm_stack.weight"),
    ("proj.weight", "adapter.proj.weight"),
]

DEC_TOP = [
    ("llm.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("llm.model.norm.weight", "dec.output_norm.weight"),
]

DEC_BLOCK = [
    ("input_layernorm.weight", "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight", "attn.q.weight"),
    ("self_attn.k_proj.weight", "attn.k.weight"),
    ("self_attn.v_proj.weight", "attn.v.weight"),
    ("self_attn.o_proj.weight", "attn.o.weight"),
    ("self_attn.q_norm.weight", "attn.q_norm.weight"),
    ("self_attn.k_norm.weight", "attn.k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn.gate.weight"),
    ("mlp.up_proj.weight", "ffn.up.weight"),
    ("mlp.down_proj.weight", "ffn.down.weight"),
]

SKIP = {"llm.lm_head.weight", "encoder.embed_positions.weight"}


def compute_size_label(total: int) -> str:
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.1f}B"
    return f"{total / 1_000_000:.0f}M"


# ---------------------------------------------------------------------------
# Frontend buffers
# ---------------------------------------------------------------------------


def whisper_frontend(n_mels: int) -> tuple[np.ndarray, np.ndarray]:
    """Filterbank + window exactly as the publisher's LogMel builds them:
    filters from WhisperFeatureExtractor, periodic torch Hann window."""
    try:
        from transformers import WhisperFeatureExtractor

        fb = np.asarray(WhisperFeatureExtractor(feature_size=n_mels).mel_filters, dtype=np.float32).T
    except ImportError:
        import librosa

        fb = librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels, fmin=0.0, fmax=8000.0,
                                 norm="slaney", htk=False).astype(np.float32)
    win = torch.hann_window(400).numpy().astype(np.float32)
    return np.ascontiguousarray(fb), np.ascontiguousarray(win)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_dir: Path, out_path: Path, variant: str, repo_id: str | None, revision: str | None,
            out_type: GGMLQuantizationType = REFERENCE_GGML_TYPE) -> None:
    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    hp = read_hparams(config)
    print(f"Encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}, rotary_dim={hp['enc_rotary_dim']}")
    print(f"Adapter: stack={hp['stack']} -> {hp['enc_d_model'] * hp['stack']} -> {hp['dec_hidden']}")
    print(f"Decoder: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, vocab={hp['dec_vocab']}")

    tok = extract_tokenizer(model_dir, hp["dec_vocab"])
    for name, want in (("<|im_start|>", hp["bos_id"]), ("<|im_end|>", hp["eos_id"]), ("<|endoftext|>", hp["pad_id"])):
        if tok["ids"].get(name) != want:
            raise ValueError(f"{name}: tokenizer id {tok['ids'].get(name)} != config id {want}")
    style_ids = {}
    for name in STYLE_TOKENS:
        tid = tok["ids"].get(name)
        if tid is None or hp["style_tokens"].get(name, tid) != tid:
            raise ValueError(f"style token {name} missing or inconsistent with config.style_tokens")
        style_ids[name] = tid

    ident = KNOWN_VARIANTS.get(variant, {
        "name": variant, "basename": variant, "author": None, "organization": None,
        "license": None, "license_name": None, "license_link": None,
    })

    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as st:
        st_keys = set(st.keys())

        if not torch.equal(st.get_tensor("llm.lm_head.weight"), st.get_tensor("llm.model.embed_tokens.weight")):
            raise ValueError("llm.lm_head.weight differs from embed_tokens; untied head unsupported")
        if st.get_tensor("encoder.embed_positions.weight").abs().max().item() != 0.0:
            raise ValueError("encoder.embed_positions.weight is not zero; this port assumes the rotary encoder")

        total = sum(st.get_tensor(k).numel() for k in st_keys if k not in SKIP)
        size_label = compute_size_label(total)
        print(f"Params (deduplicated): {total:,} -> {size_label}")

        writer = gguf_writer(str(out_path), ARCH)
        src_url = None
        if repo_id:
            src_url = f"https://huggingface.co/{repo_id}" + (f"/tree/{revision}" if revision else "")
        add_general_identity(
            writer,
            name=ident["name"],
            basename=ident["basename"],
            size_label=size_label,
            file_type=(REFERENCE_FILE_TYPE if out_type == REFERENCE_GGML_TYPE else LlamaFileType.ALL_F32),
            languages=["da"],
            author=ident["author"],
            organization=ident["organization"],
            license=ident["license"],
            license_name=ident["license_name"],
            license_link=ident["license_link"],
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
            source_url=src_url,
        )
        writer.add_string("stt.variant", variant)

        # ---- tokenizer ----
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre", "qwen2")
        writer.add_array("tokenizer.ggml.tokens", tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges", tok["merges"])
        writer.add_uint32("tokenizer.ggml.bos_token_id", hp["bos_id"])
        writer.add_uint32("tokenizer.ggml.eos_token_id", hp["eos_id"])
        writer.add_uint32("tokenizer.ggml.padding_token_id", hp["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)

        # ---- stt.whisper_qwen.* ----
        p = "stt.whisper_qwen"
        writer.add_uint32(f"{p}.encoder.n_layers", hp["enc_n_layers"])
        writer.add_uint32(f"{p}.encoder.d_model", hp["enc_d_model"])
        writer.add_uint32(f"{p}.encoder.n_heads", hp["enc_n_heads"])
        writer.add_uint32(f"{p}.encoder.ffn_dim", hp["enc_ffn_dim"])
        writer.add_uint32(f"{p}.encoder.num_mel_bins", hp["enc_n_mels"])
        writer.add_uint32(f"{p}.encoder.max_source_positions", hp["enc_max_src_pos"])
        writer.add_uint32(f"{p}.encoder.rotary_dim", hp["enc_rotary_dim"])
        writer.add_float32(f"{p}.encoder.rope_theta", hp["enc_rope_theta"])
        writer.add_uint32(f"{p}.adapter.stack", hp["stack"])
        writer.add_float32(f"{p}.adapter.rms_norm_eps", hp["adapter_rms_eps"])
        writer.add_uint32(f"{p}.decoder.n_layers", hp["dec_n_layers"])
        writer.add_uint32(f"{p}.decoder.hidden_size", hp["dec_hidden"])
        writer.add_uint32(f"{p}.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32(f"{p}.decoder.n_heads", hp["dec_n_heads"])
        writer.add_uint32(f"{p}.decoder.n_kv_heads", hp["dec_n_kv_heads"])
        writer.add_uint32(f"{p}.decoder.head_dim", hp["dec_head_dim"])
        writer.add_float32(f"{p}.decoder.rms_norm_eps", hp["dec_rms_eps"])
        writer.add_float32(f"{p}.decoder.rope_theta", hp["dec_rope_theta"])
        writer.add_uint32(f"{p}.decoder.max_position_embeddings", hp["dec_max_pos"])
        writer.add_uint32(f"{p}.decoder.vocab_size", hp["dec_vocab"])
        writer.add_uint32(f"{p}.style.nocase_token_id", style_ids["<|nocase|>"])
        writer.add_uint32(f"{p}.style.cased_token_id", style_ids["<|cased|>"])
        writer.add_uint32(f"{p}.style.nopunc_token_id", style_ids["<|nopunc|>"])
        writer.add_uint32(f"{p}.style.punc_token_id", style_ids["<|punc|>"])

        # ---- stt.frontend.* (Whisper log-mel, per-utterance normalisation) ----
        writer.add_string("stt.frontend.type", "mel")
        writer.add_uint32("stt.frontend.num_mels", hp["enc_n_mels"])
        writer.add_uint32("stt.frontend.sample_rate", 16000)
        writer.add_uint32("stt.frontend.n_fft", 400)
        writer.add_uint32("stt.frontend.win_length", 400)
        writer.add_uint32("stt.frontend.hop_length", 160)
        writer.add_string("stt.frontend.window", "hann_periodic")
        writer.add_string("stt.frontend.normalize", "per_utterance")
        writer.add_float32("stt.frontend.dither", 0.0)
        writer.add_float32("stt.frontend.pre_emphasis", 0.0)
        writer.add_float32("stt.frontend.f_min", 0.0)
        writer.add_float32("stt.frontend.f_max", 8000.0)
        writer.add_string("stt.frontend.pad_mode", "reflect")
        writer.add_bool("stt.frontend.center", True)
        writer.add_string("stt.frontend.mel_norm", "slaney")

        # ---- tensors ----
        fb, win = whisper_frontend(hp["enc_n_mels"])
        writer.add_tensor("frontend.mel_filterbank", fb, raw_dtype=GGMLQuantizationType.F32)
        writer.add_tensor("frontend.window", win, raw_dtype=GGMLQuantizationType.F32)
        n_added = 2

        def add(src: str, dst: str) -> None:
            nonlocal n_added
            t = st.get_tensor(src)
            if t.dtype != torch.bfloat16:
                raise ValueError(f"{src}: expected bfloat16, got {t.dtype}")
            arr = np.ascontiguousarray(t.float().numpy())
            encoded, raw_dtype = encode_for_gguf(arr, reference_dtype_for(dst, out_type))
            writer.add_tensor(dst, encoded, raw_dtype=raw_dtype)
            n_added += 1

        consumed = set(SKIP)
        plan: list[tuple[str, str]] = list(ENC_TOP)
        for i in range(hp["enc_n_layers"]):
            plan += [(f"encoder.layers.{i}.{s}", f"enc.blocks.{i}.{d}") for s, d in ENC_BLOCK]
        plan += ADAPTER + DEC_TOP
        for i in range(hp["dec_n_layers"]):
            plan += [(f"llm.model.layers.{i}.{s}", f"dec.blocks.{i}.{d}") for s, d in DEC_BLOCK]
        for src, dst in plan:
            if src not in st_keys:
                raise KeyError(f"safetensors missing tensor {src!r}")
            add(src, dst)
            consumed.add(src)

        unused = sorted(st_keys - consumed)
        if unused:
            raise RuntimeError(f"{len(unused)} unconsumed safetensors keys, e.g. {unused[:5]}")
        print(f"Added {n_added} tensors")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Convert a WhisperQwen checkpoint to a BF16 GGUF.")
    ap.add_argument("model", help="HF repo id (e.g. syvai/hviske-v6) or local directory")
    ap.add_argument("out_path", type=Path, nargs="?", help="output .gguf (derived from repo id when omitted)")
    ap.add_argument("--repo-id", default=None, help="HF repo id when converting from a local directory")
    ap.add_argument("--revision", default=None, help="HF revision to pin (recorded in general.source.url)")
    ap.add_argument("--variant", default=None, help="stt.variant (default: lowercased repo slug)")
    ap.add_argument("--outtype", choices=["bf16", "f32"], default="bf16",
                    help="bf16 (reference dtype, default) or f32 (numerical validation only)")
    args = ap.parse_args(argv[1:])

    if looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = download_snapshot(args.model, args.revision)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory or HF repo id", file=sys.stderr)
            return 2
        repo_id = args.repo_id

    if args.out_path is None and not repo_id:
        print("error: give out_path or --repo-id", file=sys.stderr)
        return 2
    slug = slug_from_repo_id(repo_id) if repo_id else args.out_path.stem.rsplit("-", 1)[0]
    variant = args.variant or slug.lower()
    out_type = GGMLQuantizationType.F32 if args.outtype == "f32" else REFERENCE_GGML_TYPE
    out_path = args.out_path or REPO_ROOT / "models" / slug / gguf_name(slug, args.outtype.upper())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(model_dir, out_path, variant, repo_id, args.revision, out_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
