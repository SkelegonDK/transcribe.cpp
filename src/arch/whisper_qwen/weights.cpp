// arch/whisper_qwen/weights.cpp - read_hparams + build_weights.
//
// Same pattern as arch/qwen3_asr/weights.cpp: every required KV is read
// explicitly (BadType is fatal) and every tensor is looked up with its
// expected shape; a missing tensor or shape mismatch is TRANSCRIBE_ERR_GGUF.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

namespace transcribe::whisper_qwen {

namespace {
constexpr const char * kTag = "whisper_qwen";
}

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    struct U32 {
        const char * key;
        int32_t *    dst;
    };

    const U32 u32s[] = {
        { "stt.whisper_qwen.encoder.n_layers",                &hp.enc_n_layers                },
        { "stt.whisper_qwen.encoder.d_model",                 &hp.enc_d_model                 },
        { "stt.whisper_qwen.encoder.n_heads",                 &hp.enc_n_heads                 },
        { "stt.whisper_qwen.encoder.ffn_dim",                 &hp.enc_ffn_dim                 },
        { "stt.whisper_qwen.encoder.num_mel_bins",            &hp.enc_num_mel_bins            },
        { "stt.whisper_qwen.encoder.max_source_positions",    &hp.enc_max_source_positions    },
        { "stt.whisper_qwen.encoder.rotary_dim",              &hp.enc_rotary_dim              },
        { "stt.whisper_qwen.adapter.stack",                   &hp.stack                       },
        { "stt.whisper_qwen.decoder.n_layers",                &hp.dec_n_layers                },
        { "stt.whisper_qwen.decoder.hidden_size",             &hp.dec_hidden                  },
        { "stt.whisper_qwen.decoder.intermediate_size",       &hp.dec_intermediate            },
        { "stt.whisper_qwen.decoder.n_heads",                 &hp.dec_n_heads                 },
        { "stt.whisper_qwen.decoder.n_kv_heads",              &hp.dec_n_kv_heads              },
        { "stt.whisper_qwen.decoder.head_dim",                &hp.dec_head_dim                },
        { "stt.whisper_qwen.decoder.max_position_embeddings", &hp.dec_max_position_embeddings },
        { "stt.whisper_qwen.decoder.vocab_size",              &hp.dec_vocab_size              },
        { "stt.whisper_qwen.style.nocase_token_id",           &hp.tok_nocase                  },
        { "stt.whisper_qwen.style.cased_token_id",            &hp.tok_cased                   },
        { "stt.whisper_qwen.style.nopunc_token_id",           &hp.tok_nopunc                  },
        { "stt.whisper_qwen.style.punc_token_id",             &hp.tok_punc                    },
        { "stt.frontend.num_mels",                            &hp.fe_num_mels                 },
        { "stt.frontend.sample_rate",                         &hp.fe_sample_rate              },
        { "stt.frontend.n_fft",                               &hp.fe_n_fft                    },
        { "stt.frontend.win_length",                          &hp.fe_win_length               },
        { "stt.frontend.hop_length",                          &hp.fe_hop_length               },
    };
    for (const auto & e : u32s) {
        if (auto st = read_required_u32_kv(gguf, e.key, kTag, *e.dst); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    struct F32 {
        const char * key;
        float *      dst;
    };

    const F32 f32s[] = {
        { "stt.whisper_qwen.encoder.rope_theta",     &hp.enc_rope_theta   },
        { "stt.whisper_qwen.adapter.rms_norm_eps",   &hp.adapter_rms_eps  },
        { "stt.whisper_qwen.decoder.rms_norm_eps",   &hp.dec_rms_norm_eps },
        { "stt.whisper_qwen.decoder.rope_theta",     &hp.dec_rope_theta   },
        { "stt.frontend.pre_emphasis",               &hp.fe_pre_emphasis  },
        { "stt.frontend.f_min",                      &hp.fe_f_min         },
        { "stt.frontend.f_max",                      &hp.fe_f_max         },
    };
    for (const auto & e : f32s) {
        if (auto st = read_required_f32_kv(gguf, e.key, kTag, *e.dst); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    if (auto st = read_required_string_kv(gguf, "stt.frontend.type", kTag, hp.fe_type); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.window", kTag, hp.fe_window); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.normalize", kTag, hp.fe_normalize); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.pad_mode", kTag, "reflect", hp.fe_pad_mode);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Cross-field invariants.
    if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 || hp.enc_ffn_dim <= 0 ||
        hp.enc_num_mel_bins <= 0 || hp.stack <= 0 || hp.dec_n_layers <= 0 || hp.dec_hidden <= 0 ||
        hp.dec_n_heads <= 0 || hp.dec_n_kv_heads <= 0 || hp.dec_head_dim <= 0 || hp.dec_intermediate <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: hparams must be positive");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model % hp.enc_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: encoder d_model %d not divisible by n_heads %d",
                hp.enc_d_model, hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    const int32_t enc_head_dim = hp.enc_d_model / hp.enc_n_heads;
    if (hp.enc_rotary_dim < 0 || hp.enc_rotary_dim % 2 != 0 || hp.enc_rotary_dim > enc_head_dim) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: encoder rotary_dim %d invalid for head_dim %d",
                hp.enc_rotary_dim, enc_head_dim);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: n_heads %d not divisible by n_kv_heads %d",
                hp.dec_n_heads, hp.dec_n_kv_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_type != "mel" || hp.fe_num_mels != hp.enc_num_mel_bins) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: frontend must be a %d-bin mel", hp.enc_num_mel_bins);
        return TRANSCRIBE_ERR_GGUF;
    }
    for (int32_t id : { hp.tok_nocase, hp.tok_cased, hp.tok_nopunc, hp.tok_punc }) {
        if (id < 0 || id >= hp.dec_vocab_size) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: style token id %d out of vocab range", id);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    return TRANSCRIBE_OK;
}

namespace {

using transcribe::weights::lname;

#define GET_F32(slot, name, ...)                                                                          \
    do {                                                                                                  \
        ggml_tensor * _t =                                                                                \
            transcribe::weights::find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                                \
            return TRANSCRIBE_ERR_GGUF;                                                                   \
        (slot) = _t;                                                                                      \
    } while (0)

#define GET_CONV(slot, name, ...)                                                                              \
    do {                                                                                                       \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_CONV_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                            \
        if (_t == nullptr)                                                                                     \
            return TRANSCRIBE_ERR_GGUF;                                                                        \
        (slot) = _t;                                                                                           \
    } while (0)

#define GET_LIN(slot, name, ...)                                                                                 \
    do {                                                                                                         \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                              \
        if (_t == nullptr)                                                                                       \
            return TRANSCRIBE_ERR_GGUF;                                                                          \
        (slot) = _t;                                                                                             \
    } while (0)

}  // namespace

transcribe_status build_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w) {
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t d     = hp.enc_d_model;
    const int64_t ffn   = hp.enc_ffn_dim;
    const int64_t mels  = hp.enc_num_mel_bins;
    const int64_t stk   = static_cast<int64_t>(hp.enc_d_model) * hp.stack;
    const int64_t dec_h = hp.dec_hidden;

    // Encoder stem. Conv1d kernels are [out, in, k] in PyTorch -> ggml ne [k, in, out].
    GET_CONV(w.conv0_w, "enc.conv.0.weight", 3, mels, d);
    GET_F32(w.conv0_b, "enc.conv.0.bias", d);
    GET_CONV(w.conv1_w, "enc.conv.1.weight", 3, d, d);
    GET_F32(w.conv1_b, "enc.conv.1.bias", d);

    w.enc_blocks.assign(hp.enc_n_layers, EncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = w.enc_blocks[i];
        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), d);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias", i), d);
        GET_LIN(b.attn_q_w, lname("enc.blocks.%d.attn.q.weight", i), d, d);
        GET_F32(b.attn_q_b, lname("enc.blocks.%d.attn.q.bias", i), d);
        GET_LIN(b.attn_k_w, lname("enc.blocks.%d.attn.k.weight", i), d, d);
        GET_LIN(b.attn_v_w, lname("enc.blocks.%d.attn.v.weight", i), d, d);
        GET_F32(b.attn_v_b, lname("enc.blocks.%d.attn.v.bias", i), d);
        GET_LIN(b.attn_out_w, lname("enc.blocks.%d.attn.out.weight", i), d, d);
        GET_F32(b.attn_out_b, lname("enc.blocks.%d.attn.out.bias", i), d);
        GET_F32(b.norm_ffn_w, lname("enc.blocks.%d.norm_ffn.weight", i), d);
        GET_F32(b.norm_ffn_b, lname("enc.blocks.%d.norm_ffn.bias", i), d);
        GET_LIN(b.fc1_w, lname("enc.blocks.%d.ffn.fc1.weight", i), d, ffn);
        GET_F32(b.fc1_b, lname("enc.blocks.%d.ffn.fc1.bias", i), ffn);
        GET_LIN(b.fc2_w, lname("enc.blocks.%d.ffn.fc2.weight", i), ffn, d);
        GET_F32(b.fc2_b, lname("enc.blocks.%d.ffn.fc2.bias", i), d);
    }
    GET_F32(w.final_norm_w, "enc.final_norm.weight", d);
    GET_F32(w.final_norm_b, "enc.final_norm.bias", d);

    // Adapter.
    GET_F32(w.adapter_norm_w, "adapter.norm_stack.weight", stk);
    GET_LIN(w.adapter_proj_w, "adapter.proj.weight", stk, dec_h);

    // Decoder.
    GET_LIN(w.dec_token_w, "dec.token_embd.weight", dec_h, hp.dec_vocab_size);
    const int64_t q_out  = static_cast<int64_t>(hp.dec_n_heads) * hp.dec_head_dim;
    const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
    const int64_t im     = hp.dec_intermediate;
    const int64_t hd     = hp.dec_head_dim;
    w.dec_blocks.assign(hp.dec_n_layers, DecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & b = w.dec_blocks[i];
        GET_F32(b.norm_attn_w, lname("dec.blocks.%d.norm_attn.weight", i), dec_h);
        GET_F32(b.norm_ffn_w, lname("dec.blocks.%d.norm_ffn.weight", i), dec_h);
        GET_LIN(b.attn_q_w, lname("dec.blocks.%d.attn.q.weight", i), dec_h, q_out);
        GET_LIN(b.attn_k_w, lname("dec.blocks.%d.attn.k.weight", i), dec_h, kv_out);
        GET_LIN(b.attn_v_w, lname("dec.blocks.%d.attn.v.weight", i), dec_h, kv_out);
        GET_LIN(b.attn_o_w, lname("dec.blocks.%d.attn.o.weight", i), q_out, dec_h);
        GET_F32(b.attn_q_norm, lname("dec.blocks.%d.attn.q_norm.weight", i), hd);
        GET_F32(b.attn_k_norm, lname("dec.blocks.%d.attn.k_norm.weight", i), hd);
        GET_LIN(b.ffn_gate_w, lname("dec.blocks.%d.ffn.gate.weight", i), dec_h, im);
        GET_LIN(b.ffn_up_w, lname("dec.blocks.%d.ffn.up.weight", i), dec_h, im);
        GET_LIN(b.ffn_down_w, lname("dec.blocks.%d.ffn.down.weight", i), im, dec_h);
        if (b.ffn_gate_w->type != b.ffn_up_w->type) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: ffn gate/up dtype mismatch at layer %d", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    GET_F32(w.dec_norm_w, "dec.output_norm.weight", dec_h);
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::whisper_qwen
