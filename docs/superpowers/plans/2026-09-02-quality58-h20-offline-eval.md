# Quality58 H20 RTC final checkpoint offline evaluation

## Scope

Run a strictly offline, read-only evaluation of the final π0.5 Piper Direct-Joint Quality58 H20 RTC checkpoint against the existing baseline. Do not modify training/model/data/norm/RTC/checkpoint/runner code and do not touch hardware.

## Files

- `scripts/eval_joint_pi05_rtc_h20_quality58_final_offline.py`: new standalone evaluator only; loads frozen final/baseline checkpoints and final mask-aware dataset, computes physical 7D/H20/RTC/limit/determinism/latency/schema metrics, writes a unique report directory.
- `docs/superpowers/plans/2026-09-02-quality58-h20-offline-eval.md`: this execution plan.
- `<EVAL_ROOT>/...`: generated report artifacts only.

## Steps

1. Read-only audit existing config, dataset, transforms, checkpoint loading, and runner schema. Record exact APIs and confirm no hardware imports in the new path.
2. Add the evaluator under `scripts/` with explicit offline guards and no production-code edits. Reuse existing APIs; keep final and baseline observations/prompts/noise identical and evaluate physical 7D only.
3. Run a syntax/import/preflight check, then execute the evaluator over all 58 episodes with at least 300 real forwards, full dataset statistical metrics, paired RTC comparison, determinism, and RTX5090 latency.
4. Verify report files and safety fields, inspect failures/NaN/Inf/tracebacks, and report exact measured values. Stop without deployment or robot actions.
