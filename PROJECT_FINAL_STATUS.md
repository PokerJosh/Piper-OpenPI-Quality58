# Project Final Status

## Scope

The Piper/OpenPI project is formally archived. No further feature work,
training, hardware testing, or runtime fixes are part of this repository.

## Final real-robot status

The project must not be described as a successful final robot task.

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

The checkpoint loaded and the camera/CAN feedback preflight passed. The runner
entered `REAL_RUN`, but the freshness/stale-data safety gate triggered
`STALE_DATA_HOLD` before any policy joint command was sent. The policy task was
therefore not completed.

A cold-start JAX inference delay was observed to make the first observation or
target old. `FIRST_TARGET_TIMEOUT_SECONDS=20.0` existed, while
`STALE_AFTER_SECONDS=0.50` remained. A later warmup-inference proposal (discard
warmup output, then refresh camera and CAN state) is explicitly
`PROPOSED_NOT_IMPLEMENTED`.

## Retained safety design

The archived implementation retains:

- physical hard joint limits;
- gripper limits;
- raw policy-output rejection;
- safety abort;
- executor re-check;
- maximum 1 degree per tick for J1–J6;
- stale camera, state, and target checks.

Offline evaluation observed a positive J3 overshoot tendency. Hard-limit and
raw-reject checks are consequently retained and must not be removed merely to
make a rollout proceed.
