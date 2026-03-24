"""
Download all assets needed to run nanochat in an offline environment.

This covers the minimal CPU run (runcpu.sh). For a full GPU speedrun, increase
NUM_CLIMBMIX_SHARDS to 170 (or more).

Usage:
    python -m scripts.download_offline

The script writes everything into NANOCHAT_BASE_DIR (default ~/.cache/nanochat)
and the HuggingFace cache (HF_DATASETS_CACHE, default ~/.cache/huggingface/datasets).
Mount both directories in your offline environment and set the env vars accordingly.
"""

import os
import sys
import zipfile

# ---- Config ------------------------------------------------------------------

# Number of ClimbMix train shards to download.
# 8 shards (~800MB) is enough to train the tokenizer and do a short CPU run.
# Use 170 for a full GPU speedrun (~17GB).
NUM_CLIMBMIX_SHARDS = 8

# ---------------------------------------------------------------------------

def download_climbmix(n):
    """Download n ClimbMix shards + the fixed validation shard."""
    # Import after project is on path
    from nanochat.dataset import DATA_DIR, download_single_file, MAX_SHARD
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(n)) + [MAX_SHARD]
    print(f"\n[1/5] Downloading {len(ids)} ClimbMix shards to {DATA_DIR} ...")
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        results = pool.map(download_single_file, ids)
    ok = sum(results)
    print(f"      {ok}/{len(ids)} shards downloaded.")
    if ok < len(ids):
        print("      WARNING: some shards failed. Re-run to retry.")


def download_eval_bundle():
    """Download and unzip the CORE eval bundle."""
    from nanochat.common import get_base_dir
    import urllib.request

    base_dir = get_base_dir()
    bundle_dir = os.path.join(base_dir, "eval_bundle")
    if os.path.isdir(bundle_dir):
        print("\n[2/5] eval_bundle already present, skipping.")
        return

    url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
    zip_path = os.path.join(base_dir, "eval_bundle.zip")
    print(f"\n[2/5] Downloading eval bundle ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir)
    os.remove(zip_path)
    print(f"      Extracted to {bundle_dir}")


def download_identity_conversations():
    """Download the identity conversations JSONL used in SFT."""
    from nanochat.common import get_base_dir, download_file_with_lock

    print("\n[3/5] Downloading identity conversations ...")
    url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    path = download_file_with_lock(url, "identity_conversations.jsonl")
    print(f"      Saved to {path}")


def download_hf_datasets():
    """Pre-download all HuggingFace task datasets into the HF cache."""
    print("\n[4/5] Downloading HuggingFace task datasets ...")
    from datasets import load_dataset

    tasks = [
        ("openai/gsm8k",              dict(name="main",        split="train")),
        ("openai/gsm8k",              dict(name="main",        split="test")),
        ("cais/mmlu",                 dict(name="all",         split="test")),
        ("allenai/ai2_arc",           dict(name="ARC-Easy",    split="test")),
        ("allenai/ai2_arc",           dict(name="ARC-Challenge",split="test")),
        ("openai/openai_humaneval",   dict(                    split="test")),
        ("HuggingFaceTB/smol-smoltalk", dict(                  split="train")),
    ]
    for name, kwargs in tasks:
        label = f"{name}[{kwargs.get('name', '')} {kwargs.get('split','')}]"
        try:
            load_dataset(name, **kwargs)
            print(f"      OK  {label}")
        except Exception as e:
            print(f"      ERR {label}: {e}")


def download_spellingbee_wordlist():
    """Download the English word list used by the SpellingBee task."""
    from nanochat.common import download_file_with_lock

    print("\n[5/5] Downloading spelling bee word list ...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    filename = "words_alpha.txt"
    path = download_file_with_lock(url, filename)
    print(f"      Saved to {path}")


def download_tiktoken():
    """Warm the tiktoken cache for gpt2 and cl100k_base encodings."""
    print("\n[+]  Warming tiktoken cache ...")
    import tiktoken
    for enc_name in ("gpt2", "cl100k_base"):
        tiktoken.get_encoding(enc_name)
        print(f"      OK  {enc_name}")


if __name__ == "__main__":
    # Make sure the project root is importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    print("=" * 60)
    print("nanochat offline asset downloader")
    print(f"  NANOCHAT_BASE_DIR : {base_dir}")
    print(f"  HF_DATASETS_CACHE : {hf_cache}")
    print(f"  ClimbMix shards   : {NUM_CLIMBMIX_SHARDS} train + 1 val")
    print("=" * 60)

    download_climbmix(NUM_CLIMBMIX_SHARDS)
    download_eval_bundle()
    download_identity_conversations()
    download_hf_datasets()
    download_spellingbee_wordlist()
    download_tiktoken()

    print("\nDone. Mount the two cache directories in your offline environment:")
    print(f"  {base_dir}")
    print(f"  {hf_cache}")
    print("\nThen set these env vars before running nanochat:")
    print(f"  export NANOCHAT_BASE_DIR=<mount-path>/nanochat")
    print(f"  export NANOCHAT_DATA_DIR=<mount-path>/nanochat/base_data_climbmix  # optional override")
    print(f"  export HF_DATASETS_OFFLINE=1")
    print(f"  export TRANSFORMERS_OFFLINE=1")
    print(f"  export HF_HUB_OFFLINE=1")
    print(f"  export HF_DATASETS_CACHE=<mount-path>/huggingface/datasets")
    print(f"\nFor Ascend NPU, install with:  uv sync --extra npu")
