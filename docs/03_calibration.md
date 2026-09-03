# 03 — Calibration

## What is calibrated

The follower Piper uses its own mechanical/SDK zero and hard-limit configuration. The leader kit has seven Feetech servos (IDs 1–7): six mapped arm joints and one gripper channel. Leader calibration is separate from follower calibration; both must be known before collecting demonstrations.

## Leader procedure

1. Connect one approved leader serial port, represented here by `<LEADER_SERIAL_PORT>`.
2. Ping/scan IDs 1–7 with `calibration/scripts/feetech_ping.py`.
3. Place the leader in the documented zero pose.
4. Run `calibration/scripts/feetech_servo_zero.py` in the locally reviewed environment. The script supports a dry-run/read-only inspection and a write-to-EEPROM calibration mode; writing EEPROM is not part of archive validation.
5. Verify after power cycling with the same tool's verification mode.
6. Generate a private calibration result and mapping file from the example schemas.

The expected conceptual center is 2047 ticks, not a universal claim about any particular device. `leader_zero_result.example.json` deliberately redacts measured raw positions and offsets. Never use its placeholders as calibration data.

## Mapping and failure modes

`leader_to_piper_mapping.example.json` documents tick wrapping, direction, student limits, and the gripper mapping. The teleoperator loads this mapping and applies clamp/reject behavior before producing LeRobot actions. Calibration files are consumed by the leader reader/mapping plugin, not by the π0.5 model itself.

A reset returns the runtime/scene to a known pose; zero/homing calibration changes the servo's reference relationship. They are not interchangeable. Bad zero or direction calibration shifts every demonstration, corrupts state/action alignment, and can drive a follower toward the wrong limit. Reset motion must remain outside a demonstration episode.
