// arch/whisper_qwen/graph.cpp - encoder/adapter + Qwen3 prefill/step graphs.
//
// Encoder (reference: `_encode_rotary` in modeling_whisper_qwen.py):
//
//   mel [T_mel, n_mels]
//     -> conv1 (k3 s1 p1) + GELU(erf) -> conv2 (k3 s2 p1) + GELU(erf)
//     -> [d_model, T_enc]               (no absolute position term)
//     -> N x pre-LN block: x += MHSA(LN(x)) with partial NeoX RoPE on q/k
//                          x += FC2(GELU(FC1(LN(x))))
//     -> final LN
//
// Adapter: view the first n_tokens*stack frames as [d_model*stack, n_tokens]
// (stacking consecutive frames, d innermost), RMSNorm(eps), Linear.
//
// RoPE: the reference rotates dims [0, rotary_dim) of each head with
// rotate_half over that slice and cos/sin = cat(ang, ang), inv freq
// theta^(-2i/rotary_dim). That is ggml's NEOX mode with n_dims=rotary_dim
// (the tail [rotary_dim, head_dim) passes through unrotated). The reference
// folds the 1/sqrt(head_dim) scale into q before rotation; rotation is linear
// so applying the scale inside attention is equivalent.
//
// Decoder: Qwen3 blocks from src/causal_lm/. Prompt = [audio][suffix ids],
// injected by concatenation (audio block always starts at position 0).

#include "causal_lm/causal_lm.h"
#include "conformer/conformer.h"
#include "ggml.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "whisper_qwen.h"

#include <cmath>
#include <cstdio>

namespace transcribe::whisper_qwen {

namespace {

namespace conf = transcribe::conformer;

ggml_tensor * add_conv1d_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b) {
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, b, 1, b->ne[0], 1, 1));
}

ggml_tensor * rope_partial(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, const HParams & hp) {
    // x: [head_dim, n_heads, T]
    return ggml_rope_ext(ctx, x, pos, nullptr, hp.enc_rotary_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                         hp.enc_rope_theta, /*freq_scale=*/1.0f, /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

ggml_tensor * enc_attention(ggml_context *  ctx,
                            ggml_tensor *   x,
                            ggml_tensor *   pos,
                            const EncBlock & b,
                            const HParams &  hp,
                            bool             use_flash) {
    const int     d        = hp.enc_d_model;
    const int     n_heads  = hp.enc_n_heads;
    const int     head_dim = d / n_heads;
    const int64_t T        = x->ne[1];
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, b.attn_q_w, x), b.attn_q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, b.attn_v_w, x), b.attn_v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, T);
    if (hp.enc_rotary_dim > 0) {
        q = rope_partial(ctx, q, pos, hp);
        k = rope_partial(ctx, k, pos, hp);
    }

    // -> [head_dim, T, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * o;
    if (use_flash) {
        ggml_tensor * k16 = ggml_cast(ctx, k, GGML_TYPE_F16);
        ggml_tensor * v16 = ggml_cast(ctx, v, GGML_TYPE_F16);
        o = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q), k16, v16, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        o = ggml_reshape_2d(ctx, o, d, T);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));  // [T_k, T_q, H]
        kq               = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));        // [T, head_dim, H]
        o                = ggml_mul_mat(ctx, vt, kq);                                // [head_dim, T_q, H]
        o                = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));          // [head_dim, H, T_q]
        o                = ggml_reshape_2d(ctx, o, d, T);
    }
    return ggml_add(ctx, ggml_mul_mat(ctx, b.attn_out_w, o), b.attn_out_b);
}

causal_lm::BlockView to_block_view(const DecBlock & b) {
    causal_lm::BlockView v{};
    v.norm_attn_w   = b.norm_attn_w;
    v.norm_ffn_w    = b.norm_ffn_w;
    v.attn_q_w      = b.attn_q_w;
    v.attn_k_w      = b.attn_k_w;
    v.attn_v_w      = b.attn_v_w;
    v.attn_o_w      = b.attn_o_w;
    v.attn_q_norm   = b.attn_q_norm;
    v.attn_k_norm   = b.attn_k_norm;
    v.ffn_gate_up_w = b.ffn_gate_up_w;
    v.ffn_down_w    = b.ffn_down_w;
    return v;
}

causal_lm::BlockParams to_block_params(const HParams & hp) {
    causal_lm::BlockParams p{};
    p.n_heads      = hp.dec_n_heads;
    p.n_kv_heads   = hp.dec_n_kv_heads;
    p.head_dim     = hp.dec_head_dim;
    p.max_position = hp.dec_max_position_embeddings;
    p.rms_eps      = hp.dec_rms_norm_eps;
    p.rope_theta   = hp.dec_rope_theta;
    return p;
}

// Final RMSNorm + tied head + suppression bias + argmax. x: [hidden, 1].
ggml_tensor * head_argmax(ggml_context * ctx, const Weights & w, const HParams & hp, ggml_tensor * x,
                          ggml_tensor * bias) {
    x                    = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp.dec_rms_norm_eps), w.dec_norm_w);
    ggml_tensor * logits = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, w.dec_token_w, x), hp.dec_vocab_size);
    logits               = ggml_add(ctx, logits, bias);
    ggml_set_name(logits, "dec.logits");
    transcribe::debug::mark_tensor_for_dump(logits);
    return ggml_argmax(ctx, logits);
}

}  // namespace

EncoderBuild build_encoder_graph(ggml_context * ctx, const Weights & w, const HParams & hp, int T_mel, bool use_flash) {
    EncoderBuild eb{};
    if (ctx == nullptr || T_mel <= 0 || T_mel % 2 != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen encoder: T_mel=%d must be positive and even", T_mel);
        return eb;
    }
    const int T_enc    = T_mel / 2;
    const int n_tokens = T_enc / hp.stack;
    if (n_tokens <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen encoder: clip too short (%d frames, stack %d)", T_enc,
                hp.stack);
        return eb;
    }
    eb.T_enc    = T_enc;
    eb.n_tokens = n_tokens;

    const int d = hp.enc_d_model;

    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, hp.enc_num_mel_bins);
    ggml_set_name(eb.mel_in, "enc.mel.in");
    ggml_set_input(eb.mel_in);

    eb.pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_enc);
    ggml_set_name(eb.pos_in, "enc.pos");
    ggml_set_input(eb.pos_in);

    // Stem: conv_1d_f32 takes data [W, C_in] and returns [W_out, C_out, 1].
    ggml_tensor * x = conf::conv_1d_f32(ctx, w.conv0_w, eb.mel_in, 1, 1, 1);
    x               = ggml_gelu_erf(ctx, add_conv1d_bias(ctx, x, w.conv0_b));
    x               = conf::conv_1d_f32(ctx, w.conv1_w, ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]), 2, 1, 1);
    x               = ggml_gelu_erf(ctx, add_conv1d_bias(ctx, x, w.conv1_b));
    x               = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1])));  // [d, T_enc]
    ggml_set_name(x, "enc.conv.out");
    transcribe::debug::mark_tensor_for_dump(x);

    for (int i = 0; i < hp.enc_n_layers; ++i) {
        const EncBlock & b = w.enc_blocks[i];
        ggml_tensor *    h = conf::layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
        x                  = ggml_add(ctx, x, enc_attention(ctx, h, eb.pos_in, b, hp, use_flash));
        h                  = conf::layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
        h                  = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, b.fc1_w, h), b.fc1_b));
        h                  = ggml_add(ctx, ggml_mul_mat(ctx, b.fc2_w, h), b.fc2_b);
        x                  = ggml_add(ctx, x, h);
        if (i == 0 || i == hp.enc_n_layers - 1 || transcribe::debug::enabled()) {
            char nm[48];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            ggml_set_name(x, nm);
            transcribe::debug::mark_tensor_for_dump(x);
        }
    }
    x = conf::layer_norm(ctx, x, w.final_norm_w, w.final_norm_b);
    ggml_set_name(x, "enc.final");
    transcribe::debug::mark_tensor_for_dump(x);
    eb.enc_out = x;

    // Adapter: [d, T_enc] contiguous -> first n_tokens*stack frames as
    // [d*stack, n_tokens] (frame f of token t at columns d*(t*stack+f)).
    const int64_t stk = static_cast<int64_t>(d) * hp.stack;
    ggml_tensor * g   = ggml_view_2d(ctx, x, stk, n_tokens, ggml_element_size(x) * stk, 0);
    g                 = ggml_cont(ctx, g);
    g                 = ggml_mul(ctx, ggml_rms_norm(ctx, g, hp.adapter_rms_eps), w.adapter_norm_w);
    g                 = ggml_mul_mat(ctx, w.adapter_proj_w, g);  // [dec_hidden, n_tokens]
    ggml_set_name(g, "adapter.out");
    transcribe::debug::mark_tensor_for_dump(g);

    eb.out = g;
    ggml_set_output(eb.out);
    eb.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (eb.graph == nullptr) {
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);
    if (transcribe::debug::enabled()) {
        ggml_build_forward_expand(eb.graph, eb.enc_out);
    }
    return eb;
}

PrefillBuild build_prefill_graph(ggml_context *                   ctx,
                                 const Weights &                  w,
                                 const HParams &                  hp,
                                 transcribe::causal_lm::KvCache & kv,
                                 ggml_tensor *                    suppress_bias,
                                 int                              n_audio,
                                 int                              n_suffix,
                                 bool                             use_flash) {
    PrefillBuild pb{};
    const int    T = n_audio + n_suffix;
    if (ctx == nullptr || n_audio <= 0 || n_suffix <= 0 || kv.self_k == nullptr || T > kv.n_ctx) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen prefill: invalid shape (audio=%d suffix=%d n_ctx=%d)",
                n_audio, n_suffix, kv.n_ctx);
        return pb;
    }

    pb.audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.dec_hidden, n_audio);
    ggml_set_name(pb.audio_in, "dec.audio");
    ggml_set_input(pb.audio_in);
    pb.suffix_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_suffix);
    ggml_set_name(pb.suffix_ids, "dec.suffix_ids");
    ggml_set_input(pb.suffix_ids);
    pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pb.positions_in, "dec.positions");
    ggml_set_input(pb.positions_in);
    pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    ggml_set_name(pb.mask_in, "dec.mask");
    ggml_set_input(pb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    if (gf == nullptr) {
        return pb;
    }
    pb.graph = gf;

    ggml_tensor * x = ggml_concat(ctx, pb.audio_in, ggml_get_rows(ctx, w.dec_token_w, pb.suffix_ids), 1);
    ggml_set_name(x, "dec.input");
    transcribe::debug::mark_tensor_for_dump(x);

    const auto params  = to_block_params(hp);
    const int  n_layer = hp.dec_n_layers;
    for (int il = 0; il < n_layer; ++il) {
        causal_lm::BlockOpts opts{};
        opts.use_flash             = use_flash;
        opts.slice_last_before_ffn = (il == n_layer - 1);
        x = causal_lm::block_prefill(ctx, gf, x, to_block_view(w.dec_blocks[il]), params, kv, il, T, pb.mask_in,
                                     pb.positions_in, opts);
    }
    pb.out = head_argmax(ctx, w, hp, x, suppress_bias);
    ggml_set_output(pb.out);
    ggml_build_forward_expand(gf, pb.out);
    return pb;
}

StepBuild build_step_graph(ggml_context *                   ctx,
                           const Weights &                  w,
                           const HParams &                  hp,
                           transcribe::causal_lm::KvCache & kv,
                           ggml_tensor *                    suppress_bias,
                           int                              max_n_kv,
                           bool                             use_flash) {
    StepBuild sb{};
    if (ctx == nullptr || max_n_kv <= 0 || kv.self_k == nullptr || max_n_kv > kv.n_ctx) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen step: invalid max_n_kv=%d", max_n_kv);
        return sb;
    }
    sb.input_id_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(sb.input_id_in);
    sb.position_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(sb.position_in);
    sb.kv_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_input(sb.kv_idx_in);
    sb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_n_kv, 1);
    ggml_set_input(sb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    if (gf == nullptr) {
        return sb;
    }
    sb.graph = gf;

    ggml_tensor * x      = ggml_get_rows(ctx, w.dec_token_w, sb.input_id_in);
    const auto    params = to_block_params(hp);
    for (int il = 0; il < hp.dec_n_layers; ++il) {
        x = causal_lm::block_step(ctx, gf, x, to_block_view(w.dec_blocks[il]), params, kv, il, max_n_kv, sb.mask_in,
                                  sb.position_in, sb.kv_idx_in, use_flash);
    }
    sb.out = head_argmax(ctx, w, hp, x, suppress_bias);
    ggml_set_output(sb.out);
    ggml_build_forward_expand(gf, sb.out);
    return sb;
}

}  // namespace transcribe::whisper_qwen
