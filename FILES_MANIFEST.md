# Files Manifest

This is a selective, second-stage archive of the Piper zero-to-deployment workflow. The first archive commit remains unchanged; this update adds source snapshots and procedure documentation.

## Included

- `hardware/`: Piper SDK boundary, LeRobot robot adapter, Feetech SDK, installation templates.
- `calibration/`: leader servo ID/zero tools, mapping example, sanitized calibration example, launch/config notes.
- `teleop/`: Piper leader plugin, Feetech reader, mapping, active-handoff source, package metadata.
- `recording/`: guarded recorder, episode schema/trim/review/replay utilities, conversion entrypoint, recording launcher, review configs.
- `dataset_tools/`: selected v2.1 conversion and Joint recovery/Quality58 QA utilities.
- `configs/`: sanitized camera and recording templates.
- `tests/`: focused recording, mapping, Quality58, evaluator, and deployment tests.
- Existing `src/`, `scripts/`, `artifacts/`, `examples/`, `patches/`, `docs/`, and provenance files from the first archive.

## Deliberately excluded

- Dataset episodes and parquet/pickle payloads.
- Camera images, videos, raw captures, bags, CAN dumps, and runtime logs.
- Checkpoint weights, Orbax trees, serialized model state, and cache directories.
- `.venv`, conda environments, `__pycache__`, `.git` directories, and editable-install metadata.
- Hardware serials, local absolute paths, credentials, private keys, and shell history.
- Placeholder words such as `token` in upstream model/tokenizer code are not credentials; no credential values are present.
- Exploratory inference probes, visual review servers, watchdogs, backup copies, and unrelated robot experiments.

## Provenance rule

Copied source files retain their implementation structure and upstream headers where present. Path-bearing configs and hardware-bound calibration records were converted to examples with placeholders. The authoritative tracked file list is the Git tree after the second commit.
