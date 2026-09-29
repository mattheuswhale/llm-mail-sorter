# Gmail Digest MVP v0.4.0

MailSort retrieves unread Gmail messages, analyzes them with a small local LLM
(default: Qwen3-0.6B), and assigns priority using deterministic Python rules.

By default, Gmail access remains **read-only**. An optional `--mark-read` flag can
be used to mark only the messages processed in that run as read after the local
archive has been written successfully.

## How it works

Currently only supports Gmail via the API. IMAP support is planned for later.

```text
Unread Gmail
    |
    v
hard evidence from headers/text
    |
    +----> LLM extracts objective signals + summary
                     |
                     v
               Python rules
                     |
                     v
              final priority
                     |
                     v
        timestamped local archive
                     |
                     +----> optional --mark-read
```

The JSON digest includes the extracted signals and the exact rule used for each
priority, so misclassifications are auditable.

## Priority policy

Rules are evaluated in this order:

1. Security/account/payment/service problems -> **URGENT**
2. Non-promotional action/reply explicitly due immediately -> **URGENT**
3. Promotions/newsletters -> **LOW**
4. Non-promotional action/reply/action deadline -> **HIGH**
5. Automated message with no action -> **LOW**
6. Everything else -> **NORMAL**

Critical account/payment/security problems override newsletter/promotional
evidence.

Commercial urgency ("sale ends today", "last chance", etc.) does not count as
a user deadline.

Edit `maildigest/priority.py` if you want to change this policy.

## Safety

Normal runs use Gmail's read-only OAuth scope:

```text
https://www.googleapis.com/auth/gmail.readonly
```

The normal token is stored separately at:

```text
.secrets/token_readonly.json
```

In normal mode:

- Gmail access is read-only
- the client lists and retrieves unread messages
- messages are not marked as read
- no labels are changed
- no archive, trash, delete, draft, reply, or send operations are implemented

### Optional mark-as-read mode

`--mark-read` is explicitly opt-in.

When enabled, MailSort uses a separate token:

```text
.secrets/token_modify.json
```

and requests:

```text
https://www.googleapis.com/auth/gmail.modify
```

The local timestamped archive is written **before** Gmail is modified. Only
after the archive succeeds does MailSort remove the `UNREAD` label from the
exact message IDs processed during that run.

MailSort does not implement archive, trash, delete, draft, reply, or send
operations.

`gmail.modify` is broader than only marking messages as read, so keep
`.secrets/token_modify.json` private.

### Local data

By default, full extracted email bodies are **not** written to disk.

The run history still contains private information such as:

- sender
- subject
- snippet
- generated summary
- classification results

Treat the entire output directory as private data.

Full extracted bodies are saved only when `--save-bodies` is explicitly used.

Email text is treated as untrusted LLM input, and the local LLM has no tools and
cannot call Gmail.

## Model privacy / local storage

Models are stored inside the project.

The default model keeps the legacy path:

```text
.models/
└── Qwen3-0.6B/
```

If a different model is selected without specifying `--model-dir`, MailSort
derives a model-specific directory automatically.

For example:

```bash
python run.py --model Qwen/Qwen3-1.7B
```

uses a directory similar to:

```text
.models/
└── Qwen__Qwen3-1.7B/
```

You can also choose your own directory:

```bash
python run.py \
  --model Qwen/Qwen3-1.7B \
  --model-dir ./models/qwen17b
```

After the model exists locally, `--model` is optional:

```bash
python run.py --offline --model-dir ./models/qwen17b
```

Models downloaded by MailSort contain `.mailsort_model.json`, which records the
Hugging Face repository ID. This allows MailSort to detect an explicit
`--model` / `--model-dir` mismatch instead of silently loading the wrong model.

Older or manually created local model directories can still be used with
`--model-dir` alone and are treated as local/unverified.

After a local model is available:

- `HF_HUB_OFFLINE=1` is set for the process
- `HF_HUB_DISABLE_TELEMETRY=1` is set
- Transformers loads with `local_files_only=True`
- email contents are processed locally and are not sent to Hugging Face

Use `--offline` to refuse any model download.

## Install

Create/activate a venv, then install a PyTorch build appropriate for your
machine **before** installing the remaining requirements.

For NVIDIA GPUs, install a CUDA-enabled PyTorch build from the official PyTorch
installation selector. Verify by running:

```bash
python check_gpu.py
```

You want:

```text
CUDA available: True
GPU: ...
```

Then:

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally does not install PyTorch or Accelerate, so it
does not accidentally replace your CUDA PyTorch with a CPU build.

## Google setup

1. Enable Gmail API in a Google Cloud project.
2. Configure OAuth consent.
3. Create a Desktop-app OAuth client.
4. Put the downloaded client file at `credentials.json`.

Normal runs request only `gmail.readonly`.

The first run with `--mark-read` performs a separate authorization using
`gmail.modify` and stores that token in `.secrets/token_modify.json`.

## Run

Basic read-only run:

```bash
python run.py --max-messages 30
```

Force CUDA:

```bash
python run.py --device cuda --max-messages 30
```

After the model has downloaded, enforce offline model operation:

```bash
python run.py --offline --device cuda --max-messages 30
```

Mark the processed messages as read after the archive succeeds:

```bash
python run.py \
  --device cuda \
  --max-messages 30 \
  --mark-read
```

Save full extracted email bodies in the run history:

```bash
python run.py \
  --max-messages 30 \
  --save-bodies
```

Show subjects in terminal output:

```bash
python run.py \
  --max-messages 30 \
  --show-subjects
```

By default, subjects are not printed while messages are processed to reduce
accidental leakage into terminal logs or transcripts.

## Usage

Run:

```bash
python run.py --help
```

to see the available options.

| Argument | Default | Description |
|---|---|---|
| `--credentials PATH` | `credentials.json` | Path to the Google OAuth desktop-app credentials file. |
| `--token PATH` | `.secrets/token_readonly.json` | OAuth token used for normal read-only runs. |
| `--modify-token PATH` | `.secrets/token_modify.json` | Separate OAuth token used only for `--mark-read` runs. |
| `--mark-read` | disabled | After the run archive is saved successfully, remove the `UNREAD` label from every processed message. |
| `--max-messages N` | `30` | Maximum number of unread inbox messages to retrieve and process. |
| `--model MODEL_ID` | optional | Hugging Face repository to download when the requested local model is missing. If both model arguments are omitted, the default model is `Qwen/Qwen3-0.6B`. |
| `--model-dir PATH` | optional | Local model directory. If omitted, a directory is derived from `--model`. An existing complete local model can be used with `--model-dir` alone. |
| `--device {auto,cuda,cpu}` | `auto` | Device used for inference. `auto` uses CUDA when available, otherwise CPU. |
| `--offline` | disabled | Prevent model downloads. The requested model must already exist locally. |
| `--save-bodies` | disabled | Include full extracted email bodies in `processed_messages.json`. |
| `--show-subjects` | disabled | Print email subjects while processing. |
| `--output-dir PATH` | `output` | Root directory for timestamped run archives. |

### Examples

Process up to 50 unread messages using the automatically selected device:

```bash
python run.py --max-messages 50
```

Force GPU inference:

```bash
python run.py --device cuda --max-messages 50
```

Run with an already downloaded local model:

```bash
python run.py \
  --offline \
  --device cuda \
  --model-dir ./models/qwen17b \
  --max-messages 50
```

Download a different compatible Hugging Face model:

```bash
python run.py \
  --model Qwen/Qwen3-1.7B \
  --device cuda
```

Download it to a custom directory instead:

```bash
python run.py \
  --model Qwen/Qwen3-1.7B \
  --model-dir ./models/qwen17b \
  --device cuda
```

After that download, the repository ID no longer needs to be supplied:

```bash
python run.py \
  --offline \
  --model-dir ./models/qwen17b \
  --device cuda
```

Process and then mark those exact messages as read:

```bash
python run.py \
  --offline \
  --model-dir ./models/qwen17b \
  --device cuda \
  --max-messages 50 \
  --mark-read
```

### Notes

`--max-messages` is only an upper limit. If there are fewer unread messages in
the Gmail inbox, only those messages are processed.

`--offline` affects model access only. Gmail still requires a network
connection because messages are retrieved from the Gmail API.

`--mark-read` affects only the messages successfully retrieved and processed in
that run.

`--save-bodies` stores significantly more sensitive data locally. It is
disabled by default.

If you use a custom `--output-dir`, make sure that directory is also excluded
from Git if you do not want private mail history committed accidentally.

## Output and run history

Every run is stored in its own timestamped subdirectory using local time.

For example:

```text
output/
└── 2026-09-29_11-17-32/
    ├── digest.md
    ├── digest.json
    ├── processed_messages.json
    └── manifest.json
```

If two runs happen within the same second, MailSort adds a numeric suffix to
avoid overwriting the previous run.

### `digest.md`

Human-readable priority digest.

### `digest.json`

Structured digest containing, per message:

- final priority
- category
- action-required flag
- summary
- priority rule
- extracted model signals

It also records basic information about the run and whether mailbox mutation
was requested.

### `processed_messages.json`

Stores the source message metadata together with the analysis result.

By default, it does **not** store:

- the full extracted email body
- the `List-Unsubscribe` header

The `List-Unsubscribe` header is omitted because it can contain a
recipient-specific token.

Use:

```bash
python run.py --save-bodies
```

to explicitly include full extracted bodies.

Even without `--save-bodies`, this file can contain sensitive sender, subject,
snippet, and summary information.

### `manifest.json`

Records information about the run, including:

- local and UTC generation time
- number of processed messages
- model repository/directory
- selected device
- whether full bodies were saved
- whether mark-as-read was requested
- how many messages were marked read
- mark-as-read status

For `--mark-read`, the status can progress through:

```text
pending
attempting
completed
```

or:

```text
failed
```

The manifest is updated to `attempting` **before** the Gmail modification is
made. This makes an interrupted mark-as-read operation easier to diagnose.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover priority behavior as well as the newer safety features,
including:

- promotional sale -> low even if the LLM mistakes "shop now" for an action
- payment failure -> urgent even with a newsletter footer
- immediate real action -> urgent
- ordinary requested action -> high
- automated no-action mail -> low
- routine personal informational mail -> normal
- string `"false"` from model output correctly parses as false
- read-only Gmail clients cannot call the mark-as-read operation
- `--mark-read` removes only the `UNREAD` label from the supplied message IDs
- run archives and body-saving behavior
- model repository/directory mismatch detection
