// arch/whisper_qwen/whisper_qwen.h - WhisperQwen model, session, graph builders.
//
// INTERNAL to src/arch/whisper_qwen/. WhisperQwen (syvai/hviske-v6) is a
// RoPE-adapted Whisper encoder whose frames are stacked 4:1, RMS-normed and
// projected into a Qwen3 causal LM. The LM prompt is
//
//   [audio embeddings][style controls...][<|im_start|>]
//
// and the transcript is generated greedily up to <|im_end|>. There is no
// chat template and no cross-attention.

#pragma once

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <optional>
#include <vector>

namespace transcribe::whisper_qwen {

void apply_family_invariants(transcribe_model & model);

struct WqModel final : public transcribe_model {
    Tokenizer      tok;
    HParams        hparams;
    Weights        weights;
    ggml_context * ctx_meta = nullptr;

    transcribe::BackendPlan                    plan;
    ggml_backend_buffer_t                      backend_buffer = nullptr;
    transcribe::causal_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    // Logit bias [vocab]: -inf on the style-control and other control ids
    // that must never be generated (EOS stays open), 0 elsewhere. Lives in
    // its own weights-usage backend buffer: a graph input set once would be
    // recycled by the allocator between step-graph computes.
    ggml_context *        ctx_aux       = nullptr;
    ggml_backend_buffer_t aux_buffer    = nullptr;
    ggml_tensor *         suppress_bias = nullptr;  // [vocab] f32

    WqModel() = default;
    ~WqModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct WqSession final : public transcribe_session {
    transcribe::causal_lm::KvCache kv_cache;

    std::vector<float> mel_buf;
    std::vector<float> enc_host;

    bool encoder_use_flash = false;
    bool decoder_use_flash = true;

    WqSession() = default;
    ~WqSession() override;
};

// ---------------------------------------------------------------------------
// Encoder + adapter graph
// ---------------------------------------------------------------------------

struct EncoderBuild {
    ggml_tensor * mel_in    = nullptr;  // [T_mel, n_mels] f32 (row-major [n_mels][T_mel])
    ggml_tensor * pos_in    = nullptr;  // [T_enc] i32 frame index for RoPE
    ggml_tensor * enc_out   = nullptr;  // [d_model, T_enc] after final LN (dump)
    ggml_tensor * out       = nullptr;  // [dec_hidden, n_tokens] adapter output
    ggml_cgraph * graph     = nullptr;
    int           T_enc     = 0;
    int           n_tokens  = 0;
};

// T_mel must be even. n_tokens = (T_mel / 2) / stack; the trailing partial
// group is dropped exactly as the reference's `enc[:, : n * s]` does.
EncoderBuild build_encoder_graph(ggml_context * ctx, const Weights & w, const HParams & hp, int T_mel, bool use_flash);

// ---------------------------------------------------------------------------
// Decoder graphs
// ---------------------------------------------------------------------------

struct PrefillBuild {
    ggml_tensor * audio_in     = nullptr;  // [dec_hidden, n_audio] f32
    ggml_tensor * suffix_ids   = nullptr;  // [n_suffix] i32 (controls + BOS)
    ggml_tensor * positions_in = nullptr;  // [T_prompt] i32
    ggml_tensor * mask_in      = nullptr;  // [T_prompt, T_prompt] f16 causal
    ggml_tensor * out          = nullptr;  // [1] i32 argmax of the last position
    ggml_cgraph * graph        = nullptr;
};

PrefillBuild build_prefill_graph(ggml_context *                   ctx,
                                 const Weights &                  w,
                                 const HParams &                  hp,
                                 transcribe::causal_lm::KvCache & kv,
                                 ggml_tensor *                    suppress_bias,
                                 int                              n_audio,
                                 int                              n_suffix,
                                 bool                             use_flash);

struct StepBuild {
    ggml_tensor * input_id_in = nullptr;  // [1] i32
    ggml_tensor * position_in = nullptr;  // [1] i32
    ggml_tensor * kv_idx_in   = nullptr;  // [1] i64
    ggml_tensor * mask_in     = nullptr;  // [max_n_kv, 1] f16
    ggml_tensor * out         = nullptr;  // [1] i32 argmax
    ggml_cgraph * graph       = nullptr;
};

StepBuild build_step_graph(ggml_context *                   ctx,
                           const Weights &                  w,
                           const HParams &                  hp,
                           transcribe::causal_lm::KvCache & kv,
                           ggml_tensor *                    suppress_bias,
                           int                              max_n_kv,
                           bool                             use_flash);

}  // namespace transcribe::whisper_qwen
