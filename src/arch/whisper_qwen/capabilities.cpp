// arch/whisper_qwen/capabilities.cpp - family invariants.

#include "whisper_qwen.h"

namespace transcribe::whisper_qwen {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;
    caps.native_sample_rate        = 16000;
    caps.max_timestamp_kind        = TRANSCRIBE_TIMESTAMPS_NONE;
    caps.supports_translate        = false;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    // Casing + punctuation are steered by the <|cased|>/<|punc|> and
    // <|nocase|>/<|nopunc|> control tokens; DEFAULT lets the model decide.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_PNC, true);
}

}  // namespace transcribe::whisper_qwen
