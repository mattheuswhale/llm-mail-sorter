# Safety audit — v3

- [x] OAuth scope is exactly gmail.readonly.
- [x] Known Gmail write-capable scopes are rejected.
- [x] Gmail wrapper exposes only list/get.
- [x] No message modify call.
- [x] No trash/delete call.
- [x] No send/draft/reply call.
- [x] Raw message bodies are not serialized to output files.
- [x] Email text is treated as untrusted model input.
- [x] LLM is local and has no tools.
- [x] Model is stored under the program directory.
- [x] Transformers loads local model files only.
- [x] HF telemetry is disabled.
- [x] Once model download completes, HF Hub is put in offline mode.
- [x] LLM does not assign priority.
- [x] Priority is assigned by deterministic Python rules.
- [x] Output exposes the rule and extracted signals for auditability.
