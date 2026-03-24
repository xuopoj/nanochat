# Nanochat Study Journal

**Goal**: Understand the full nanochat codebase — architecture, training, data pipeline, inference.
**Period**: March 22–26, 2026 (~5 hours/day)

---

## Progress Tracker

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 1 | Mar 22 (Sun) | Orientation + Architecture | ⬜ Not started |
| 2 | Mar 23 (Mon) | Training & Optimization | ⬜ Not started |
| 3 | Mar 24 (Tue) | Data Pipeline + Evaluation | ⬜ Not started |
| 4 | Mar 25 (Wed) | Inference + Synthesis | ⬜ Not started |

Status: ⬜ Not started · 🔄 In progress · ✅ Done · ⏭ Skipped

---

## Day 1 — Sunday Mar 22: Orientation + Architecture

**Files**: `runs/speedrun.sh`, `scripts/base_train.py`, `nanochat/common.py`, `nanochat/gpt.py`, `nanochat/flash_attention.py`

### Checklist
- [ ] `runs/speedrun.sh` — map the full pipeline end-to-end
- [ ] `scripts/base_train.py` (skim) — understand the training entry point
- [ ] `nanochat/common.py` — dtype management, distributed setup
- [ ] `nanochat/gpt.py` — full model (RoPE, QK-norm, GQA, sliding window, ReLU²)
- [ ] `nanochat/flash_attention.py` — FA3 + SDPA fallback

### Notes

* detect_compute_type: determine if a device support bf16(have a huge performance boost)
* logging: only on rank 0, with better formatting
* detect and setup ddp
* get flops for device, need them to compute mfu

---

## Day 2 — Monday Mar 23: Training & Optimization

**Files**: `nanochat/optim.py`, `scripts/base_train.py` (deep read), `nanochat/common.py` (revisit)

### Checklist
- [ ] `nanochat/optim.py` — Muon optimizer, Polar Express, NorMuon
- [ ] `scripts/base_train.py` (deep read) — DDP, grad accumulation, LR schedule, training loop
- [ ] Revisit dtype management in `common.py` with training context

### Notes

---

## Day 3 — Tuesday Mar 24: Data Pipeline + Evaluation

**Files**: `nanochat/tokenizer.py`, `scripts/tok_train.py`, `nanochat/dataset.py`, `nanochat/dataloader.py`, `nanochat/loss_eval.py`, `nanochat/core_eval.py`, `tasks/arc.py`, `tasks/gsm8k.py`

### Checklist
- [ ] `nanochat/tokenizer.py` + `scripts/tok_train.py` — BPE tokenizer
- [ ] `nanochat/dataset.py` — data downloading, preprocessing
- [ ] `nanochat/dataloader.py` — BOS packing, distributed sharding
- [ ] `nanochat/loss_eval.py` — bits-per-byte metric
- [ ] `nanochat/core_eval.py` — DCLM CORE 22-task ensemble
- [ ] `tasks/arc.py`, `tasks/gsm8k.py` — benchmark structure

### Notes

---

## Day 4 — Wednesday Mar 25: Inference + Synthesis

**Files**: `nanochat/engine.py`, `scripts/chat_sft.py`, `runs/runcpu.sh`

### Checklist
- [ ] `nanochat/engine.py` — KV cache, token generation
- [ ] `scripts/chat_sft.py` — SFT finetuning (bonus)
- [ ] Run `runs/runcpu.sh` — watch a small model train end-to-end
- [ ] Trace a token from disk to loss through the full stack

### Notes

---

## Questions & Things to Revisit

<!-- Running list of things that confused you or that you want to dig into more -->

---

## Key Concepts Learned

<!-- Running glossary — add entries as you encounter new ideas -->

| Concept | One-line explanation |
|---------|---------------------|
| | |
