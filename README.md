# Piper from Setup to π0.5 Deployment

This private repository is a historical, selective archive of the complete Piper/OpenPI workflow: hardware preparation, CAN and SDK integration, leader/follower teleoperation, camera setup, demonstration recording, dataset review and conversion, Quality58 RTC H20 training, offline evaluation, guarded Direct Joint deployment, safety design, and the final experiment result.

It is an archive, not an active development branch. No production behavior is changed here. No training, real-robot run, CAN operation, or GPU evaluation is performed by archive validation.

## Final Quality58 experiment

- Policy: OpenPI π0.5 Piper Direct Joint RTC H20 Quality58
- Physical output: 7 values, `[J1, J2, J3, J4, J5, J6, gripper]`
- Model/latent action width: 32
- Action horizon: 20
- Dataset selection: 47 nominal + 11 recovery = 58 episodes
- Valid H20 windows: 25,336
- Training: 3,000 steps; logical final checkpoint step: 2,999
- Checkpoint reference only: `pi05_piper_joint_rtc_h20_quality58_3k_20260902_135854/2999`

The real dataset, videos, images, raw recordings, CAN dumps, checkpoint weights, caches, and environments are intentionally absent. The examples and templates describe their schemas without containing payloads.

## Workflow map

1. [Overview](docs/00_overview.md)
2. [Environment setup](docs/01_environment_setup.md)
3. [CAN and SDK](docs/02_piper_can_sdk.md)
4. [Calibration](docs/03_calibration.md)
5. [Teleoperation](docs/04_teleoperation.md)
6. [Camera setup](docs/05_camera_setup.md)
7. [Data recording](docs/06_data_recording.md)
8. [Dataset processing](docs/07_dataset_processing.md)
9. [π0.5 training](docs/08_pi05_training.md)
10. [Offline evaluation](docs/09_offline_evaluation.md)
11. [Direct Joint deployment](docs/10_direct_joint_deployment.md)
12. [Final real-robot run](docs/11_real_robot_run.md)
13. [Safety design](docs/12_safety.md)
14. [Known issues](docs/13_known_issues.md)
15. [Command reference](docs/COMMAND_REFERENCE.md)

The source snapshots are organized under `hardware/`, `calibration/`, `teleop/`, `recording/`, `dataset_tools/`, `configs/`, `tests/`, and the existing `src/`, `scripts/`, `artifacts/`, and `patches/` directories.

## Final real-robot result

The final run was **not successful**:

```text
PREFLIGHT=PASS
CAMERA_PROBE=PASS
FEEDBACK_PROBE=PASS
CHECKPOINT_LOAD=PASS
ACTION_DIM=7
ACTION_HORIZON=20
MODEL_ACTION_DIM=32
ARM_ENABLE_COUNT=1
CAN_WRITE_COUNT=3
COMMAND_SEND_COUNT=0
SAFETY_REJECT_COUNT=1
STOP_REASON=STALE_DATA_HOLD
RUN_STATUS=FAIL
TASK_COMPLETE=NO
```

The model loaded and camera/feedback preflight passed. The runner entered `REAL_RUN`, but the freshness gate stopped execution before any policy joint command was sent. `CAN_WRITE_COUNT=3` includes enable/stop control writes and does not mean a policy rollout succeeded.

A cold-start JAX inference delay was observed. `FIRST_TARGET_TIMEOUT_SECONDS=20.0` was added, while the runtime freshness threshold remained `STALE_AFTER_SECONDS=0.50`. A proposal to warm up inference, discard that result, reacquire fresh camera and CAN state, and then enter runtime was **PROPOSED_NOT_IMPLEMENTED**. The project was archived at that point.

## Safety boundary

The archived code retains physical hard limits, gripper limits, raw policy-output rejection, safety aborts, executor re-checks, a maximum 1 degree/tick for J1–J6, and camera/state/target freshness checks. Offline evaluation observed a positive J3 overshoot tendency; these checks must not be removed to make a run proceed.

Any command that can open a device, enable an arm, write CAN, or start a camera is an operator-reviewed template only. Replace placeholders locally and perform an independent safety review first.

## Provenance

The project-specific additions are layered on upstream OpenPI/LeRobot interfaces and Piper/Feetech/RealSense SDK boundaries. See `THIRD_PARTY_NOTICE.md`, `FILES_MANIFEST.md`, and `provenance/source_revisions.txt`. The archive is selective and may require a compatible upstream OpenPI tree: `REQUIRES_UPSTREAM_OPENPI_TREE`.
