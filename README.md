# Gmail Read-Only Digest MVP v3

This version keeps Gmail strictly read-only, runs small sized llms(default Qwen3-0.6B) locally, and makes priority assignment deterministic.

## How it works

Currently only supports gmail via the api, Imap support planned for later.

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
              final priority + digest
```

The JSON digest includes the extracted signals and the exact rule used for each priority, so misclassifications are auditable.

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

The Gmail inbox remains read-only:

- OAuth scope is only `https://www.googleapis.com/auth/gmail.readonly`
- saved credentials containing known Gmail write scopes are rejected
- Gmail wrapper exposes only message `list` and `get`
- no mark-read, modify, label, archive, trash, delete, draft, reply, or send code
- raw email bodies are not saved to the output files
- email text is treated as untrusted input
- the local LLM model has no tools and cannot call Gmail

## Model privacy / local storage

The model is stored inside the project.

Example:
```text
gmail_readonly_digest_mvp_v3/
└── .models/
    └── LLM3-0.6B/
```

On the first run, if that directory does not contain a complete model,
`huggingface_hub.snapshot_download()` downloads the model into it.

After the local model is complete:

- `HF_HUB_OFFLINE=1` is set for the process
- `HF_HUB_DISABLE_TELEMETRY=1` is set
- Transformers loads with `local_files_only=True`
- email contents are processed locally and are not sent to Hugging Face

Use `--offline` to refuse any model download:

## Install

Create/activate a venv, then install a PyTorch build appropriate for your
machine **before** installing the remaining requirements.

For NVIDIA gpus, install a CUDA-enabled PyTorch build from the official PyTorch
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

The application itself asks only for `gmail.readonly`.

## Run

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

## Usage

Run:

```bash
python run.py --help
```

to see the available options.

| Argument | Default | Description |
|---|---|---|
| `--credentials PATH` | `credentials.json` | Path to the Google OAuth desktop-app credentials file. |
| `--token PATH` | `.secrets/token_readonly.json` | Path where the read-only Gmail OAuth token is stored. |
| `--max-messages N` | `30` | Maximum number of unread inbox messages to retrieve and process. |
| `--model MODEL_ID` | `Qwen/Qwen3-0.6B` | Hugging Face model repository to download if the model is not already available locally. |
| `--model-dir PATH` | `.models/Qwen3-0.6B` | Directory where the local model files are stored. |
| `--device {auto,cuda,cpu}` | `auto` | Device used for inference. `auto` uses CUDA when available, otherwise CPU. |
| `--offline` | disabled | Prevents model downloads and Hugging Face Hub access. The model must already exist locally. |
| `--output-dir PATH` | `output` | Directory where `digest.md` and `digest.json` are written. |

### Examples

Process up to 50 unread messages using the automatically selected device:

```bash
python run.py --max-messages 50
```

Force GPU inference:

```bash
python run.py --device cuda --max-messages 50
```

Run completely offline after the model has already been downloaded:

```bash
python run.py --offline --device cuda --max-messages 50
```

Use a different compatible Hugging Face model:

```bash
python run.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --model-dir .models/SmolLM2-1.7B-Instruct \
  --device cuda
```

When changing models, use a separate `--model-dir` for each model to avoid mixing model files.

### Notes

`--max-messages` is only an upper limit. If there are fewer unread messages in the Gmail inbox, only those messages are processed.

`--offline` affects model access only. Gmail still requires a network connection because unread messages are retrieved from the Gmail API.

The application requests Gmail's read-only OAuth scope and does not mark messages as read, archive them, modify labels, delete messages, or send mail.

## Output

```text
output/
├── digest.md
└── digest.json
```

The JSON contains, per message:

- final priority
- category
- action-required flag
- summary
- priority rule
- extracted model signals

It does not save the raw email body.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- promotional sale -> low even if LLM mistakes "shop now" for an action
- payment failure -> urgent even with a newsletter footer
- immediate real action -> urgent
- ordinary requested action -> high
- automated no-action mail -> low
- routine personal informational mail -> normal
- string `"false"` from model output correctly parses as false
