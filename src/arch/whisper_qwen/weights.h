// arch/whisper_qwen/weights.h - WhisperQwen tensor catalog and hparams.
//
// INTERNAL to src/arch/whisper_qwen/. Architecture pattern: audio-LLM with
// prefix injection (reference: syvai/hviske-v6 modeling_whisper_qwen.py).
//
//   enc.*      Whisper encoder: 2x Conv1d stem + pre-LN blocks + final LN.
//              No absolute positions (the checkpoint's sinusoid table is all
//              zeros); q/k are rotated by frame index with partial NeoX RoPE.
//   adapter.*  stack `stack` frames -> RMSNorm(d_model*stack) -> Linear.
//   dec.*      Qwen3 causal LM (GQA, per-head Q/K RMSNorm, SwiGLU, tied head).
//
// Tensor naming mirrors scripts/convert-whisper_qwen.py. Linear weights use
// PyTorch [out, in] order; Conv1d kernels use [out, in, k].

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::whisper_qwen {

struct HParams {
    // Encoder.
    int32_t enc_n_layers             = 0;
    int32_t enc_d_model              = 0;
    int32_t enc_n_heads              = 0;
    int32_t enc_ffn_dim              = 0;
    int32_t enc_num_mel_bins         = 0;
    int32_t enc_max_source_positions = 0;  // trained window (frames after conv2)
    int32_t enc_rotary_dim           = 0;  // rotated dims per head (even, <= head_dim)
    float   enc_rope_theta           = 0.0f;

    // Adapter.
    int32_t stack           = 0;
    float   adapter_rms_eps = 0.0f;

    // Decoder (Qwen3).
    int32_t dec_n_layers                = 0;
    int32_t dec_hidden                  = 0;
    int32_t dec_intermediate            = 0;
    int32_t dec_n_heads                 = 0;
    int32_t dec_n_kv_heads              = 0;
    int32_t dec_head_dim                = 0;
    float   dec_rms_norm_eps            = 0.0f;
    float   dec_rope_theta              = 0.0f;
    int32_t dec_max_position_embeddings = 0;
    int32_t dec_vocab_size              = 0;

    // Style-control tokens (input-only conditioning, suppressed in output).
    int32_t tok_nocase = -1;
    int32_t tok_cased  = -1;
    int32_t tok_nopunc = -1;
    int32_t tok_punc   = -1;

    // Resolved from the tokenizer at load time.
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;

    // Frontend (Whisper log-mel).
    std::string fe_type;
    std::string fe_window;
    std::string fe_normalize;
    std::string fe_pad_mode;
    int32_t     fe_num_mels     = 0;
    int32_t     fe_sample_rate  = 0;
    int32_t     fe_n_fft        = 0;
    int32_t     fe_win_length   = 0;
    int32_t     fe_hop_length   = 0;
    float       fe_pre_emphasis = 0.0f;
    float       fe_f_min        = 0.0f;
    float       fe_f_max        = 0.0f;
};

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp);

struct EncBlock {
    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;  // no bias (Whisper)
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * norm_ffn_w  = nullptr;
    ggml_tensor * norm_ffn_b  = nullptr;
    ggml_tensor * fc1_w       = nullptr;
    ggml_tensor * fc1_b       = nullptr;
    ggml_tensor * fc2_w       = nullptr;
    ggml_tensor * fc2_b       = nullptr;
};

struct DecBlock {
    ggml_tensor * norm_attn_w   = nullptr;
    ggml_tensor * norm_ffn_w    = nullptr;
    ggml_tensor * attn_q_w      = nullptr;
    ggml_tensor * attn_k_w      = nullptr;
    ggml_tensor * attn_v_w      = nullptr;
    ggml_tensor * attn_o_w      = nullptr;
    ggml_tensor * attn_q_norm   = nullptr;
    ggml_tensor * attn_k_norm   = nullptr;
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
    ggml_tensor * ffn_gate_up_w = nullptr;  // packed at load (causal_lm::pack_gate_up)
};

struct Weights {
    ggml_tensor *         conv0_w      = nullptr;  // [k=3, n_mels, d_model]
    ggml_tensor *         conv0_b      = nullptr;
    ggml_tensor *         conv1_w      = nullptr;  // [k=3, d_model, d_model], stride 2
    ggml_tensor *         conv1_b      = nullptr;
    std::vector<EncBlock> enc_blocks;
    ggml_tensor *         final_norm_w = nullptr;
    ggml_tensor *         final_norm_b = nullptr;

    ggml_tensor * adapter_norm_w = nullptr;  // [d_model * stack]
    ggml_tensor * adapter_proj_w = nullptr;  // [d_model * stack, dec_hidden]

    ggml_tensor *         dec_token_w = nullptr;  // [dec_hidden, vocab]; tied lm_head
    std::vector<DecBlock> dec_blocks;
    ggml_tensor *         dec_norm_w  = nullptr;
};

transcribe_status build_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w);

}  // namespace transcribe::whisper_qwen
