# OpenPI π0.5 Piper Direct Joint RTC H20 Quality58

This repository is the final, selective archive of the Piper/OpenPI Quality58
project. It is an archive, not an active development branch.

## Final experiment

- Model: π0.5 Piper Direct Joint RTC H20 Quality58
- Physical action dimension: 7 (J1–J6 plus gripper)
- Latent/model action dimension: 32
- Action horizon: 20
- Dataset selection: 47 nominal + 11 recovery = 58 episodes
- Valid H20 windows: 25,336
- Training: 3,000 steps
- Final checkpoint logical step: 2,999
- Checkpoint reference: `pi05_piper_joint_rtc_h20_quality58_3k_20260902_135854/2999`

The checkpoint, dataset episodes, videos, images, logs, and runtime caches are
intentionally not included.

## Contents

- `src/openpi/`: Quality58 model/data/policy additions and tests.
- `src/openpi/rollouts/`: final pure Direct Joint safety core and tests.
- `scripts/`: final Direct Joint2999 runner, camera manifest verifier, offline
evaluator, mask/data utilities, and focused tests.
- `configs/`: sanitized camera configuration example.
- `examples/`: metadata-only sanitized manifest and sample-mask examples; no
episode payloads are present.
- `artifacts/`: the small final normalization statistics JSON and sanitized
provenance metadata.
- `patches/`: project-scoped recovery patches, sanitized of local paths and
hardware identifiers.
- `docs/`: project plans, design specification, and normalization notes.
- `provenance/`: source revisions and the explicitly preserved deployment
snapshot of shared files that differed from the main worktree.

## Reproducibility note

This is a selective snapshot and therefore requires the upstream OpenPI tree
for imports and the original compatible Python dependencies:
`REQUIRES_UPSTREAM_OPENPI_TREE`.

Before any offline use, replace the placeholders `<OPENPI_ROOT>`,
`<DEPLOYMENT_ROOT>`, `<CHECKPOINT_ROOT>`, `<DATASET_ROOT>`, and
`<DATA_QA_ROOT>` with paths in the local environment. The sanitized camera
configuration also requires a locally supplied device path and camera serial;
no real hardware identifier is archived here. See `REPRODUCIBILITY.md`.

No command in this archive should be used to enable a robot, write CAN, probe
cameras, train a model, or run a long GPU evaluation without an independent
safety review. The archived runner retains its fail-closed gates.

## Project outcome

The final real-run attempt did **not** complete the policy task. Preflight,
camera probe, feedback probe, and checkpoint load passed, but the runner held
on stale data before sending any policy joint command. The exact final status
is recorded in `PROJECT_FINAL_STATUS.md`.

Upstream copyright and license notices remain in `LICENSE`,
`LICENSE_GEMMA.txt`, and `THIRD_PARTY_NOTICE.md`.
