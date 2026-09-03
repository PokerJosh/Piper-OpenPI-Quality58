# 11 — Final Real-Robot Run

The final run is recorded accurately and is not a success claim:

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

The model checkpoint loaded, camera probing passed, and CAN feedback/preflight passed. The runner entered `REAL_RUN`, but freshness checking detected stale observation/target data before any policy joint command was sent. `CAN_WRITE_COUNT=3` includes control writes such as enable/stop and does not represent successful policy execution.

A cold-start JAX inference delay was known to make the first observation/target old. `FIRST_TARGET_TIMEOUT_SECONDS=20.0` handled the first-target wait, but normal runtime freshness remained `STALE_AFTER_SECONDS=0.50`. The task was not completed and the project was archived at this point.
