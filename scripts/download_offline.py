"""
Download all assets needed to run nanochat in an offline environment.

This covers the minimal CPU run (runcpu.sh). For a full GPU speedrun, increase
NUM_CLIMBMIX_SHARDS to 170 (or more).

Usage:
    python -m scripts.download_offline          # download everything (default 8 shards)
    python -m scripts.download_offline --sft     # download only SFT datasets (no ClimbMix/eval)
    python -m scripts.download_offline -n 170    # download 170 ClimbMix shards for full GPU run

The script writes everything into NANOCHAT_BASE_DIR (default ~/.cache/nanochat)
and the HuggingFace cache (HF_DATASETS_CACHE, default ~/.cache/huggingface/datasets).
Mount both directories in your offline environment and set the env vars accordingly.

Dependencies: pip install datasets requests  (no torch needed)
"""

import os
import sys
import time
import argparse
import zipfile
import urllib.request

# ---- Lightweight helpers (no torch dependency) --------------------------------

def get_base_dir():
    """Return the nanochat base directory (mirrors nanochat.common.get_base_dir)."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        nanochat_dir = os.path.join(home_dir, ".cache", "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file(url, filepath):
    """Download url to filepath with retries, skip if already exists."""
    if os.path.exists(filepath):
        print(f"      Skipping {os.path.basename(filepath)} (already exists)")
        return True
    filename = os.path.basename(filepath)
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            import requests
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except Exception as e:
            print(f"      Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for p in [filepath + ".tmp", filepath]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"      Waiting {wait}s before retry...")
                time.sleep(wait)
    print(f"      FAILED to download {filename}")
    return False


# ---- ClimbMix (pretraining data) --------------------------------------------

CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542

def download_climbmix(n):
    """Download n ClimbMix train shards + the fixed validation shard."""
    base_dir = get_base_dir()
    data_dir = os.environ.get("NANOCHAT_DATA_DIR", os.path.join(base_dir, "base_data_climbmix"))
    os.makedirs(data_dir, exist_ok=True)
    ids = list(range(n)) + [CLIMBMIX_MAX_SHARD]
    print(f"\n[1/5] Downloading {len(ids)} ClimbMix shards to {data_dir} ...")
    ok = 0
    for idx in ids:
        filename = f"shard_{idx:05d}.parquet"
        filepath = os.path.join(data_dir, filename)
        url = f"{CLIMBMIX_BASE_URL}/{filename}"
        if download_file(url, filepath):
            ok += 1
    print(f"      {ok}/{len(ids)} shards downloaded.")
    if ok < len(ids):
        print("      WARNING: some shards failed. Re-run to retry.")


# ---- Eval bundle -------------------------------------------------------------

def download_eval_bundle():
    """Download and unzip the CORE eval bundle."""
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


# ---- Identity conversations --------------------------------------------------

def download_identity_conversations():
    """Download the identity conversations JSONL used in SFT."""
    base_dir = get_base_dir()
    print("\n[3/5] Downloading identity conversations ...")
    url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    download_file(url, filepath)
    print(f"      Saved to {filepath}")


# ---- HuggingFace task datasets -----------------------------------------------

def download_hf_datasets():
    """Pre-download all HuggingFace task datasets into the HF cache."""
    print("\n[4/5] Downloading HuggingFace task datasets ...")
    from datasets import load_dataset

    tasks = [
        # SFT training datasets
        ("openai/gsm8k",                dict(name="main",            split="train")),
        ("openai/gsm8k",                dict(name="main",            split="test")),
        ("cais/mmlu",                   dict(name="all",             split="test")),
        ("cais/mmlu",                   dict(name="auxiliary_train", split="train")),
        ("HuggingFaceTB/smol-smoltalk", dict(                        split="train")),
        ("HuggingFaceTB/smol-smoltalk", dict(                        split="test")),
        # Eval-only datasets
        ("allenai/ai2_arc",             dict(name="ARC-Easy",        split="test")),
        ("allenai/ai2_arc",             dict(name="ARC-Challenge",   split="test")),
        ("openai/openai_humaneval",     dict(                        split="test")),
    ]
    for name, kwargs in tasks:
        label = f"{name}[{kwargs.get('name', '')} {kwargs.get('split', '')}]"
        try:
            load_dataset(name, **kwargs)
            print(f"      OK  {label}")
        except Exception as e:
            print(f"      ERR {label}: {e}")


# ---- SpellingBee word list ---------------------------------------------------

def download_spellingbee_wordlist():
    """Download the English word list used by the SpellingBee task."""
    base_dir = get_base_dir()
    print("\n[5/5] Downloading spelling bee word list ...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    filepath = os.path.join(base_dir, "words_alpha.txt")
    download_file(url, filepath)
    print(f"      Saved to {filepath}")


# ---- Tiktoken cache ----------------------------------------------------------

def download_tiktoken():
    """Warm the tiktoken cache for gpt2 and cl100k_base encodings."""
    print("\n[+]  Warming tiktoken cache ...")
    try:
        import tiktoken
        for enc_name in ("gpt2", "cl100k_base"):
            tiktoken.get_encoding(enc_name)
            print(f"      OK  {enc_name}")
    except ImportError:
        print("      SKIP (tiktoken not installed)")


# ---- Main --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download nanochat offline assets")
    parser.add_argument("-n", "--num-shards", type=int, default=8,
                        help="Number of ClimbMix train shards to download (default: 8, use 170 for full GPU run)")
    parser.add_argument("--sft", action="store_true",
                        help="Download only SFT datasets (HF tasks, identity conversations, word list)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    print("=" * 60)
    print("nanochat offline asset downloader")
    print(f"  NANOCHAT_BASE_DIR : {base_dir}")
    print(f"  HF_DATASETS_CACHE : {hf_cache}")
    if not args.sft:
        print(f"  ClimbMix shards   : {args.num_shards} train + 1 val")
    print(f"  Mode              : {'SFT only' if args.sft else 'full'}")
    print("=" * 60)

    if args.sft:
        # SFT-only mode: skip ClimbMix and eval bundle
        download_identity_conversations()
        download_hf_datasets()
        download_spellingbee_wordlist()
    else:
        # Full mode: everything
        download_climbmix(args.num_shards)
        download_eval_bundle()
        download_identity_conversations()
        download_hf_datasets()
        download_spellingbee_wordlist()
        download_tiktoken()

    print("\n" + "=" * 60)
    print("Done. Mount the two cache directories in your offline environment:")
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
