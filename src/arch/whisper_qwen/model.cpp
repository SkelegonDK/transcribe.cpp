// arch/whisper_qwen/model.cpp - WhisperQwen family handler (syvai/hviske-v6).
//
// Inference contract mirrored from the publisher's processing_whisper_qwen.py
// (HviskeASR.transcribe / transcribe_batch, batch of one):
//
//   - Audio over 30 s is split into <= 28 s pieces at quiet points; each
//     piece is transcribed independently and the texts are joined by " ".
//   - Each piece is zero-padded to a mel length bucket: ceil(frames / 100)
//     * 100, clamped to [200, 3000] frames. The encoder runs over the padded
//     window (no attention mask, as in the reference) and only the audio
//     tokens covering real frames enter the prompt.
//   - Prompt: [audio][style controls][<|im_start|>]. Style controls come
//     from transcribe_run_params.pnc: DEFAULT = none (the model picks the
//     format from the audio, which is how the published WER was measured),
//     ON = <|cased|><|punc|>, OFF = <|nocase|><|nopunc|>.
//   - Greedy decode, style tokens suppressed, stop at <|im_end|> or the
//     per-piece budget ceil(seconds * 8) + 12 (capped at 440).
//   - Output: special tokens dropped, stripped, runaway n-gram repeats
//     collapsed to at most three copies (collapse_repeats).
//
// The reference decodes with num_beams=2; this port is greedy like every
// other causal-LM family here.

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-repetition-guard.h"
#include "whisper_qwen.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace transcribe::whisper_qwen {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, WqModel>);
static_assert(std::is_base_of_v<transcribe_session, WqSession>);

WqSession::~WqSession() {
    kv_cache.free();
}

WqModel::~WqModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    packed_gate_up.free();
    if (aux_buffer != nullptr) {
        safe_buffer_free(aux_buffer);
        aux_buffer = nullptr;
    }
    if (ctx_aux != nullptr) {
        ggml_free(ctx_aux);
        ctx_aux = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

namespace {

constexpr char k_default_variant[] = "whisper-qwen";

// Reference inference constants (processing_whisper_qwen.py).
constexpr int    k_mel_bucket        = 100;   // pad mel length to a multiple (1 s)
constexpr int    k_min_mel           = 200;   // 2 s floor
constexpr int    k_max_mel           = 3000;  // 30 s, the pretrained window
constexpr int    k_max_new_tokens    = 440;
constexpr double k_tokens_per_second = 8.0;
constexpr int    k_token_slack       = 12;
constexpr double k_split_trigger_s   = 30.0;
constexpr double k_max_chunk_s       = 28.0;

// KV window: <= 375 audio tokens + 3 prompt tokens + 440 generated.
constexpr int k_kv_ctx = 1024;

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t0 = ggml_time_us();

    auto m     = std::make_unique<WqModel>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;
    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    HParams & hp    = m->hparams;
    hp.bos_token_id = m->tok.bos_id();
    hp.eos_token_id = m->tok.eos_id();
    if (m->tok.n_tokens() != hp.dec_vocab_size || hp.bos_token_id < 0 || hp.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: tokenizer vocab %d / bos %d / eos %d inconsistent",
                m->tok.n_tokens(), hp.bos_token_id, hp.eos_token_id);
        return TRANSCRIBE_ERR_GGUF;
    }

    m->caps.max_audio_ms = 0;  // long input is chunked internally

    // Mel frontend (Whisper: periodic Hann, log10, max-8 clamp, (x+4)/4).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = hp.fe_sample_rate;
        cfg.num_mels     = hp.fe_num_mels;
        cfg.n_fft        = hp.fe_n_fft;
        cfg.win_length   = hp.fe_win_length;
        cfg.hop_length   = hp.fe_hop_length;
        cfg.pre_emphasis = hp.fe_pre_emphasis;
        cfg.f_min        = hp.fe_f_min;
        cfg.f_max        = hp.fe_f_max;
        cfg.pad_mode     = hp.fe_pad_mode;
        cfg.window_type  = hp.fe_window;
        cfg.normalize    = hp.fe_normalize;
        using R          = transcribe::load_common::ReadF32Result;
        const size_t fb  = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
        const auto   rc1 = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(), "frontend.mel_filterbank", fb, "whisper_qwen", cfg.filterbank);
        const auto rc2 = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(), "frontend.window", static_cast<size_t>(cfg.win_length), "whisper_qwen",
            cfg.window);
        if ((rc1 != R::Ok && rc1 != R::Absent) || (rc2 != R::Ok && rc2 != R::Absent)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        m->mel.emplace(cfg);
    }

    gguf_init_params ip{};
    ip.no_alloc            = true;
    ip.ctx                 = &m->ctx_meta;
    gguf_context * gguf_dt = gguf_init_from_file(loader.path().c_str(), ip);
    if (gguf_dt == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (auto st = build_weights(m->ctx_meta, hp, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_dt);
        return st;
    }

    const transcribe_backend_request req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(req, (params != nullptr) ? params->device : nullptr,
                                                          "whisper_qwen", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_dt);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    m->backend_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (m->backend_buffer == nullptr) {
        gguf_free(gguf_dt);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (auto st = transcribe::load_common::stream_tensor_data(loader.path(), gguf_dt, m->ctx_meta, "whisper_qwen");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_dt);
        return st;
    }
    gguf_free(gguf_dt);

    // Generation suppression bias: every control token except EOS (the style
    // controls are the reference's suppress_tokens; the other specials are
    // never targets and would otherwise leak into the text).
    {
        std::vector<float> bias(static_cast<size_t>(hp.dec_vocab_size), 0.0f);
        for (int id = 0; id < hp.dec_vocab_size; ++id) {
            if (id != hp.eos_token_id && m->tok.is_control(id)) {
                bias[id] = -std::numeric_limits<float>::infinity();
            }
        }
        for (int32_t id : { hp.tok_nocase, hp.tok_cased, hp.tok_nopunc, hp.tok_punc }) {
            bias[id] = -std::numeric_limits<float>::infinity();
        }
        ggml_init_params aip{};
        aip.mem_size = ggml_tensor_overhead() * 2;
        aip.no_alloc = true;
        m->ctx_aux   = ggml_init(aip);
        if (m->ctx_aux == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
        m->suppress_bias = ggml_new_tensor_1d(m->ctx_aux, GGML_TYPE_F32, hp.dec_vocab_size);
        ggml_set_name(m->suppress_bias, "dec.suppress_bias");
        m->aux_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_aux, m->plan.primary);
        if (m->aux_buffer == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(m->aux_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(m->suppress_bias, bias.data(), 0, bias.size() * sizeof(float));
    }

    {
        std::vector<transcribe::causal_lm::GateUpEntry> entries;
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({ b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w });
        }
        if (auto st = transcribe::causal_lm::pack_gate_up(m->plan.primary, hp.dec_hidden, hp.dec_intermediate,
                                                          entries, m->packed_gate_up, "whisper_qwen");
            st != TRANSCRIBE_OK) {
            m->packed_gate_up.free();
            return st;
        }
    }

    m->t_load_us = ggml_time_us() - t0;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto cc       = std::make_unique<WqSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    auto *          cm      = static_cast<WqModel *>(model);
    const ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, k_kv_ctx, cm->hparams.dec_n_kv_heads,
                                        cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen init_context: KV cache allocation failed");
        return TRANSCRIBE_ERR_OOM;
    }
    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Long-form planning: cut audio over 30 s into <= 28 s spans at quiet points.
// The reference uses librosa.effects.split(top_db=30) and cuts at the middle
// of the gaps between voiced intervals; this picks, for each span, the
// quietest 200 ms window in its last half and cuts at that window's centre.
// ---------------------------------------------------------------------------
std::vector<std::pair<int, int>> plan_chunks(const float * pcm, int n, int sr) {
    std::vector<std::pair<int, int>> spans;
    if (n <= static_cast<int>(k_split_trigger_s * sr)) {
        spans.emplace_back(0, n);
        return spans;
    }
    const int hop = sr / 100;  // 10 ms energy frames
    const int nf  = n / hop;
    std::vector<double> energy(static_cast<size_t>(nf) + 1, 0.0);  // prefix sums
    for (int f = 0; f < nf; ++f) {
        double e = 0.0;
        for (int i = 0; i < hop; ++i) {
            const double s = pcm[f * hop + i];
            e += s * s;
        }
        energy[f + 1] = energy[f] + e;
    }
    const int limit = static_cast<int>(k_max_chunk_s * sr);
    const int win   = 20;  // 200 ms
    int       start = 0;
    while (n - start > limit) {
        const int lo_f = (start + limit / 2) / hop;
        const int hi_f = std::min((start + limit) / hop - win, nf - win);
        int       best = hi_f;
        double    best_e = std::numeric_limits<double>::infinity();
        for (int f = lo_f; f <= hi_f; ++f) {
            const double e = energy[f + win] - energy[f];
            if (e < best_e) {
                best_e = e;
                best   = f;
            }
        }
        int cut = (best + win / 2) * hop;
        cut     = std::clamp(cut, start + sr, start + limit);
        spans.emplace_back(start, cut);
        start = cut;
    }
    if (start < n) {
        spans.emplace_back(start, n);
    }
    return spans;
}

// collapse_repeats(text, keep=3) from the reference: trim word n-grams
// (n = 1..4) repeated more than `keep` times in a row.
std::string collapse_repeats(const std::string & text, int keep = 3) {
    std::vector<std::string> words;
    {
        std::istringstream is(text);
        std::string        w;
        while (is >> w) {
            words.push_back(w);
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t size = 1; size <= 4; ++size) {
            std::vector<std::string> out;
            size_t                   i = 0;
            while (i < words.size()) {
                if (i + size > words.size()) {
                    out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(i), words.end());
                    break;
                }
                size_t reps = 1;
                while (i + (reps + 1) * size <= words.size() &&
                       std::equal(words.begin() + static_cast<std::ptrdiff_t>(i),
                                  words.begin() + static_cast<std::ptrdiff_t>(i + size),
                                  words.begin() + static_cast<std::ptrdiff_t>(i + reps * size))) {
                    ++reps;
                }
                for (size_t r = 0; r < std::min<size_t>(reps, static_cast<size_t>(keep)); ++r) {
                    out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(i),
                               words.begin() + static_cast<std::ptrdiff_t>(i + size));
                }
                if (reps > static_cast<size_t>(keep)) {
                    changed = true;
                }
                i += reps * size;
            }
            words.swap(out);
        }
    }
    std::string res;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            res += ' ';
        }
        res += words[i];
    }
    return res;
}

transcribe_status reset_compute_ctx(WqSession * cc, size_t mb) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = mb * 1024 * 1024;
    ip.no_alloc     = true;
    cc->compute_ctx = ggml_init(ip);
    return cc->compute_ctx != nullptr ? TRANSCRIBE_OK : TRANSCRIBE_ERR_OOM;
}

// Transcribe one <= 30 s piece. Appends the cleaned text to `out_text` and
// the raw generated ids' text to `out_raw`.
transcribe_status run_piece(WqSession *                  cc,
                            WqModel *                    cm,
                            const float *                pcm,
                            int                          n,
                            const std::vector<int32_t> & controls,
                            std::string &                out_text,
                            std::string &                out_raw,
                            bool &                       repeated,
                            bool &                       truncated) {
    const HParams & hp  = cm->hparams;
    const int       hop = hp.fe_hop_length;

    // Mel bucket + zero pad (reference bucket_frames / transcribe_batch).
    const int frames_real = static_cast<int>((static_cast<int64_t>(n) + hop - 1) / hop);
    int       target      = ((frames_real + k_mel_bucket - 1) / k_mel_bucket) * k_mel_bucket;
    target                = std::clamp(target, k_min_mel, k_max_mel);
    const int samples     = target * hop;
    const int n_used      = std::min(n, samples);
    const int mel_len     = std::min(static_cast<int>((static_cast<int64_t>(n_used) + hop - 1) / hop), target);

    std::vector<float> padded(static_cast<size_t>(samples), 0.0f);
    std::memcpy(padded.data(), pcm, static_cast<size_t>(n_used) * sizeof(float));

    const int64_t t_mel0 = ggml_time_us();
    int           n_mels = 0;
    int           T_mel  = 0;
    if (auto st = cm->mel->compute(padded.data(), padded.size(), cc->mel_buf, n_mels, T_mel, cc->n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    cc->t_mel_us += ggml_time_us() - t_mel0;
    if (T_mel != target) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_qwen: mel produced %d frames, expected %d", T_mel, target);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { n_mels, T_mel };
        transcribe::debug::dump_host_f32("enc.mel.in", cc->mel_buf.data(), static_cast<long long>(cc->mel_buf.size()),
                                         shape, 2, "frontend.mel");
    }

    // ---- encoder + adapter ----
    const int64_t t_enc0 = ggml_time_us();
    if (auto st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, hp, T_mel, cc->encoder_use_flash);
    if (eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0, cc->mel_buf.size() * sizeof(float));
    {
        std::vector<int32_t> pos(static_cast<size_t>(eb.T_enc));
        for (int i = 0; i < eb.T_enc; ++i) {
            pos[i] = i;
        }
        ggml_backend_tensor_set(eb.pos_in, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (transcribe::debug::enabled()) {
        transcribe::debug::dump_tensor("enc.final", eb.enc_out, "enc.final");
        transcribe::debug::dump_tensor("adapter.out", eb.out, "adapter.out");
        if (ggml_tensor * t = ggml_graph_get_tensor(eb.graph, "enc.conv.out")) {
            transcribe::debug::dump_tensor("enc.conv.out", t, "enc.conv.out");
        }
        for (int i = 0; i < hp.enc_n_layers; ++i) {
            char nm[48];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            if (ggml_tensor * t = ggml_graph_get_tensor(eb.graph, nm)) {
                transcribe::debug::dump_tensor(nm, t, nm);
            }
        }
    }

    // Audio tokens covering real frames: conv2 maps n -> (n-1)//2 + 1, then
    // groups of `stack` rounded up; clamp to [1, n_tokens].
    const int enc_real = (mel_len - 1) / 2 + 1;
    const int n_audio  = std::clamp((enc_real + hp.stack - 1) / hp.stack, 1, eb.n_tokens);
    cc->enc_host.resize(static_cast<size_t>(hp.dec_hidden) * n_audio);
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
    cc->t_encode_us += ggml_time_us() - t_enc0;

    // ---- prefill ----
    const int64_t        t_dec0 = ggml_time_us();
    std::vector<int32_t> suffix(controls);
    suffix.push_back(hp.bos_token_id);
    const int n_suffix = static_cast<int>(suffix.size());
    const int T_prompt = n_audio + n_suffix;
    const int seconds_budget =
        static_cast<int>(std::ceil(static_cast<double>(n) / hp.fe_sample_rate * k_tokens_per_second)) + k_token_slack;
    int max_new = std::max(1, std::min(k_max_new_tokens, seconds_budget));
    max_new     = std::min(max_new, cc->kv_cache.n_ctx - T_prompt);
    if (max_new <= 0) {
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    if (cc->kv_cache.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
    }
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    if (auto st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    PrefillBuild pb = build_prefill_graph(cc->compute_ctx, cm->weights, hp, cc->kv_cache, cm->suppress_bias, n_audio,
                                          n_suffix, cc->decoder_use_flash);
    if (pb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(pb.audio_in, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
    ggml_backend_tensor_set(pb.suffix_ids, suffix.data(), 0, suffix.size() * sizeof(int32_t));
    {
        std::vector<int32_t> pos(static_cast<size_t>(T_prompt));
        for (int i = 0; i < T_prompt; ++i) {
            pos[i] = i;
        }
        ggml_backend_tensor_set(pb.positions_in, pos.data(), 0, pos.size() * sizeof(int32_t));
        const ggml_fp16_t        z = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, ninf);
        for (int r = 0; r < T_prompt; ++r) {
            for (int c = 0; c <= r; ++c) {
                mask[static_cast<size_t>(r) * T_prompt + c] = z;
            }
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(cc->sched, pb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    int32_t next = 0;
    ggml_backend_tensor_get(pb.out, &next, 0, sizeof(int32_t));
    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    std::vector<int32_t> gen;
    gen.push_back(next);

    // ---- step loop ----
    int max_n_kv = 256;
    while (max_n_kv < T_prompt + max_new) {
        max_n_kv *= 2;
    }
    max_n_kv = std::min(max_n_kv, cc->kv_cache.n_ctx);

    if (auto st = reset_compute_ctx(cc, 8); st != TRANSCRIBE_OK) {
        return st;
    }
    StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, hp, cc->kv_cache, cm->suppress_bias, max_n_kv,
                                    cc->decoder_use_flash);
    if (sb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    const ggml_fp16_t        z    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t        ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(static_cast<size_t>(max_n_kv), ninf);
    std::fill(step_mask.begin(), step_mask.begin() + T_prompt, z);

    const int32_t eos      = hp.eos_token_id;
    int           cur_past = T_prompt;
    while (next != eos && static_cast<int>(gen.size()) < max_new && cur_past < max_n_kv) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int32_t pos_val = cur_past;
        const int64_t kv_val  = cur_past;
        step_mask[cur_past]   = z;
        ggml_backend_tensor_set(sb.input_id_in, &next, 0, sizeof(int32_t));
        ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_val, 0, sizeof(int64_t));
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0, step_mask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(cc->sched, sb.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_get(sb.out, &next, 0, sizeof(int32_t));
        gen.push_back(next);
        ++cur_past;
        cc->kv_cache.n    = cur_past;
        cc->kv_cache.head = cur_past;
        if (next != eos && transcribe::stop_on_repetition(gen, "whisper_qwen run")) {
            repeated = true;
            break;
        }
    }
    if (!gen.empty() && gen.back() == eos) {
        gen.pop_back();
    } else if (!repeated) {
        truncated = true;
        transcribe::trim_repetition_at_budget_stop(gen, "whisper_qwen run");
    }
    cc->t_decode_us += ggml_time_us() - t_dec0;

    // Drop special tokens (skip_special_tokens=True in the reference).
    std::vector<int32_t> text_ids;
    text_ids.reserve(gen.size());
    for (int32_t id : gen) {
        if (!cm->tok.is_control(id)) {
            text_ids.push_back(id);
        }
    }
    const std::string raw = cm->tok.decode(text_ids.data(), static_cast<int>(text_ids.size()));
    if (!out_raw.empty()) {
        out_raw += ' ';
    }
    out_raw += raw;
    const std::string clean = collapse_repeats(raw);
    if (!clean.empty()) {
        if (!out_text.empty()) {
            out_text += ' ';
        }
        out_text += clean;
    }
    return TRANSCRIBE_OK;
}

transcribe_status run(transcribe_session * session, const float * pcm, int n_samples, const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<WqSession *>(session);
    auto * cm = static_cast<WqModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    transcribe::debug::init();

    const HParams &      hp = cm->hparams;
    std::vector<int32_t> controls;
    if (params != nullptr) {
        if (params->pnc == TRANSCRIBE_PNC_MODE_ON) {
            controls = { hp.tok_cased, hp.tok_punc };
        } else if (params->pnc == TRANSCRIBE_PNC_MODE_OFF) {
            controls = { hp.tok_nocase, hp.tok_nopunc };
        }
    }

    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;

    std::string text;
    std::string raw;
    bool        repeated  = false;
    bool        truncated = false;
    for (const auto & [a, b] : plan_chunks(pcm, n_samples, hp.fe_sample_rate)) {
        if (auto st = run_piece(cc, cm, pcm + a, b - a, controls, text, raw, repeated, truncated);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }

    cc->raw_text    = raw;
    cc->full_text   = text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_session::SegmentEntry seg{};
    seg.text  = text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / hp.fe_sample_rate;
    cc->segments.push_back(std::move(seg));

    if (repeated) {
        cc->mark_repetition_stop();
    } else if (truncated) {
        cc->was_truncated = true;
    }
    return cc->truncation_status();
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "whisper_qwen",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

}  // namespace transcribe::whisper_qwen
