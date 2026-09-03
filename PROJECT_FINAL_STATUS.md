# Project Final Status

This project is formally archived. No feature work, training, hardware testing, or runtime fixes are part of this repository.

## Final real-robot status

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

Checkpoint loading and camera/CAN feedback preflight passed. The runner entered `REAL_RUN`, but the stale-data safety gate triggered before any policy joint command was sent. The policy task was not completed.

`CAN_WRITE_COUNT=3` records enable/stop control writes at the boundary. It is not a count of policy commands. `COMMAND_SEND_COUNT=0` is the authoritative result for policy action delivery.

## Known runtime behavior

Cold-start JAX inference can make the first observation or target old by the time the runtime freshness check evaluates it. The runner has `FIRST_TARGET_TIMEOUT_SECONDS=20.0`, but retains `STALE_AFTER_SECONDS=0.50` for normal runtime freshness. The proposed sequence below was not implemented:

1. Run a cold-start warmup inference.
2. Discard the warmup output.
3. Acquire fresh camera frames.
4. Acquire fresh Piper feedback.
5. Enter formal runtime.

Status: `PROPOSED_NOT_IMPLEMENTED`. The archive intentionally preserves the fail-closed behavior and does not claim a successful task.

## Retained safety design

- Physical hard joint limits.
- Gripper limits.
- Raw policy-output rejection.
- Safety abort.
- Executor re-check.
- Maximum 1 degree/tick for J1–J6.
- Stale camera, state, and target checks.
- Driver fault checks and explicit enable/shutdown boundaries.

Offline evaluation observed a positive J3 overshoot tendency. Hard-limit and raw-reject checks remain required.
