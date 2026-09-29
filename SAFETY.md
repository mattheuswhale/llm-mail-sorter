# Safety audit

## Default mode

- [x] Normal runs use only `gmail.readonly`.
- [x] Normal runs reject a token containing `gmail.modify`.
- [x] No message is marked read unless `--mark-read` is explicitly supplied.
- [x] Email subjects are not printed to terminal logs unless `--show-subjects` is supplied.
- [x] Full extracted email bodies are not archived unless `--save-bodies` is supplied.
- [x] The LLM is local and has no Gmail tools.
- [x] Transformers loads local model files only after download.
- [x] Hugging Face telemetry is disabled.

## Optional `--mark-read` mode

- [x] Uses a separate `.secrets/token_modify.json` token by default.
- [x] Requests `gmail.modify` only for runs that explicitly use `--mark-read`.
- [x] The only implemented mailbox mutation is removing the `UNREAD` label from the processed message IDs.
- [x] The timestamped local archive is written before any Gmail mutation.
- [x] Manifest state changes to `attempting` before Gmail is modified.
- [x] No trash/delete/send/draft/reply/archive operations are implemented.

## Local archives

- [x] Every run is saved under `output/<local timestamp>/` by default.
- [x] `processed_messages.json` stores metadata/snippets/results by default.
- [x] Full bodies are opt-in with `--save-bodies`.
- [x] `List-Unsubscribe` is omitted unless `--save-bodies` is used because it may contain recipient-specific tokens.
- [x] `digest.json`, `digest.md`, and `manifest.json` are saved alongside it.
- [!] Output still contains private email metadata, snippets, summaries, and subjects. Keep it out of source control and protect it accordingly.

## Model identity

- [x] Models downloaded by MailSort record their Hugging Face repo in `.mailsort_model.json`.
- [x] An explicit `--model` that conflicts with recorded local metadata is rejected.
- [x] Existing untracked/manual local model directories can still be used with `--model-dir` alone.
