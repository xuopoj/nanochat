# Running nanochat Offline

## Step 1 — Download assets (on an internet-connected machine)

```bash
python -m scripts.download_offline
```

This downloads everything into two cache directories:

| Cache | Default path | Contents |
|---|---|---|
| `NANOCHAT_BASE_DIR` | `~/.cache/nanochat` | ClimbMix shards, eval bundle, identity conversations, spellingbee word list |
| `HF_DATASETS_CACHE` | `~/.cache/huggingface/datasets` | GSM8K, MMLU, ARC, HumanEval, SmolTalk |

The script downloads 8 ClimbMix shards by default (~800 MB), which is enough for
tokenizer training and a CPU/MPS run (`runs/runcpu.sh`). For a full GPU speedrun,
increase `NUM_CLIMBMIX_SHARDS` to 170 at the top of `scripts/download_offline.py`.

## Step 2 — Mount the cache in your offline environment

Mount (or copy) both cache directories into the offline machine. The layout inside
each directory must be preserved exactly — nanochat and HuggingFace both look for
files at specific relative paths.

Example with a shared volume mounted at `/data/nanochat-cache`:

```
/data/nanochat-cache/
├── nanochat/                  ← NANOCHAT_BASE_DIR
│   ├── base_data_climbmix/
│   │   ├── shard_00000.parquet
│   │   └── ...
│   ├── eval_bundle/
│   ├── identity_conversations.jsonl
│   └── words_alpha.txt
└── huggingface/
    └── datasets/              ← HF_DATASETS_CACHE
```

## Step 3 — Set environment variables and run

```bash
export NANOCHAT_BASE_DIR=/data/nanochat-cache/nanochat
export HF_DATASETS_CACHE=/data/nanochat-cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CPU/MPS run (no GPU needed):
bash runs/runcpu.sh
```

The `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` flags tell HuggingFace
libraries to never attempt network calls, so the run will fail fast if a dataset
is missing rather than hanging on a timeout.

## What each asset is used for

| Asset | Used by |
|---|---|
| ClimbMix shards | `tok_train.py` (tokenizer), `base_train.py` (pretraining) |
| eval bundle | `base_eval.py` (CORE metric) |
| identity conversations | `chat_sft.py` (personality fine-tuning) |
| GSM8K | `chat_sft.py`, `chat_eval.py` |
| MMLU | `chat_sft.py`, `chat_eval.py` |
| ARC | `chat_eval.py` |
| HumanEval | `chat_eval.py` |
| SmolTalk | `chat_sft.py` |
| words_alpha.txt | `chat_sft.py` (SpellingBee task) |
