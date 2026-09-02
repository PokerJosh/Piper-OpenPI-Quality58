# Joint RTC H20 Quality58 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the frozen 58-episode Joint dataset, mask-aware H20 loader, training-time RTC loss, independent normalization, and an exact 5-step GPU smoke gate without starting formal training.

**Architecture:** Keep the existing LeRobot/Pi0.5 pipeline and Joint action semantics. Add a read-only manifest/mask dataset adapter that maps each legal `(episode, start)` to one H20 sample, a model-level training RTC option that conditions on a clean committed prefix and computes loss only on the postfix, and a separate config/assets identity for the 58-episode experiment. The smoke runner reuses the production train step but disables checkpointing, W&B, robot, and CAN.

**Tech Stack:** Python, PyTorch/LeRobot, JAX, Flax NNX, Optax, pytest, JSON/SHA256.

**Spec:** `<USER_HOME>/.codex/attachments/7f0f6b51-c0f6-4895-bf93-0d01ea2f758d/pasted-text.txt`

## Global Constraints

- Final data is exactly 47 nominal plus 11 manually passed recovery episodes.
- Final H20 window count is exactly 25336 and all sampling must use `FINAL_TRAIN_SAMPLE_MASK.json`.
- Joint actions are `[J1, J2, J3, J4, J5, J6, G]`; J1-J6 use `DeltaActions`, G remains absolute.
- New model horizon is 20, training RTC max delay is 4, planned execution horizon is 8.
- The old `pi05_piper_joint_recovery_mix_finetune` config and old checkpoint semantics remain unchanged.
- Norm stats are recomputed independently from the final masked H20 data.
- Smoke is exactly five full forward/loss/backward/update steps, batch 16 first and one batch-8 fallback only on CUDA OOM.
- No formal 3000-step training, checkpoint save, Orbax save, W&B logging, robot, CAN, source-data mutation, or old-checkpoint mutation.

### Task 1: Validate and freeze data inputs

**Files:**
- Create: `<DATA_QA_ROOT>/joint_rtc_expert_salvage_20260901T161500Z/RTC_QUALITY58_DATA_FREEZE.json`
- Read-only: `FINAL_TRAINING_MANIFEST.json`, `FINAL_TRAIN_SAMPLE_MASK.json`, review artifacts, nominal and recovery sources
- Test/utility: `scripts/validate_joint_rtc_quality58.py`

- [ ] Validate counts, uniqueness, source mappings, quality classes, legal ranges, H20 count, and diversity coverage.
- [ ] Record SHA256 and source fingerprints in the freeze metadata.
- [ ] Verify source trees are unchanged before and after all later steps.

### Task 2: Add the mask-aware data adapter

**Files:**
- Modify: `src/openpi/training/data_loader.py`
- Test: `src/openpi/training/data_loader_test.py`
- Optional utility: `src/openpi/training/joint_rtc_dataset.py`

- [ ] Write failing tests for legal-only sampling, H20 boundary safety, recovery pickle loading, fixed-seed determinism, and source read-only behavior.
- [ ] Implement a flat legal-window index and read-only adapters for LeRobot and recovery pickle episodes.
- [ ] Wire the adapter only through the new data config; preserve ordinary configs.

### Task 3: Add training-time RTC

**Files:**
- Modify: `src/openpi/models/pi0.py`
- Create/modify: `src/openpi/models/training_rtc.py`
- Test: `src/openpi/models/pi0_test.py` or `src/openpi/models/training_rtc_test.py`

- [ ] Write failing tests for disabled parity, prefix lengths 0/1/4, clean-prefix conditioning, postfix-only loss, H20 shapes, finite gradients, deterministic JIT, and 7D action semantics.
- [ ] Implement the minimal fixed-shape JAX loss path with committed clean prefix and postfix loss masking.
- [ ] Keep inference RTC guidance separate and preserve old loss behavior when disabled.

### Task 4: Add independent config and norm workflow

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `scripts/compute_norm_stats.py` or add a focused quality58 norm script
- Create: new assets directory under `openpi/assets/`
- Test: `src/openpi/training/config_test.py` and focused norm/transform tests

- [ ] Add `pi05_piper_joint_rtc_h20_quality58_finetune` with params-only init, fresh optimizer, H20, RTC settings, prompt, and frozen Joint semantics.
- [ ] Compute independent norm stats from all 25336 legal windows using the exact production transforms.
- [ ] Run finite and normalize/unnormalize consistency checks.

### Task 5: Verify params and run the conditional smoke

**Files:**
- Create: `scripts/smoke_joint_rtc_h20_quality58.py`
- Test: focused smoke helper tests where practical

- [ ] Verify the 2999 params-only loader against the new model shape.
- [ ] Record `nvidia-smi`, RAM, JAX backend/devices, and memory metrics.
- [ ] Run five full train steps on GPU with batch 16, and only retry batch 8 for CUDA OOM.
- [ ] Stop with an explicit failure report if GPU is unavailable; never substitute CPU for the RTX5090 gate.

### Verification

- [ ] Run the focused TDD tests and relevant existing model/data tests.
- [ ] Run the hard validation again after freeze generation.
- [ ] Confirm source fingerprints and old config/checkpoint hashes are unchanged.
- [ ] Report exact gate values and stop without formal training.
