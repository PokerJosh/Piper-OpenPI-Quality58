# 04 — Teleoperation

The final data path is a single Piper follower driven by a Feetech leader. The leader plugin in `teleop/lerobot_teleoperator_piper_leader/` reads servo ticks, applies the calibrated mapping, and emits `[J1.pos … J6.pos, G.pos]`. J1–J6 are degrees at the LeRobot/Piper boundary; G is gripper opening in metres.

The follower plugin in `hardware/lerobot_robot_piper/` consumes these actions over the Piper CAN SDK. The nominal loop is configured around 30–50 Hz depending on recording or runtime path; motion command duration and speed caps remain explicit config values.

Calibration is a prerequisite. The plugin supports hold-last/read-error behavior and clamp/reject policies. `require_all_servos` determines whether a partial leader read freezes the whole action.

`active_handoff.py` is retained as an experimental support path for pose synchronization; it is not required by the final Quality58 single-arm training pipeline. No dual-arm deployment claim is made.

Emergency procedure: stop issuing actions, use the physical emergency stop, keep the workspace clear, then disconnect according to the locally reviewed Piper procedure. Do not rely on a process exit to make an unsafe pose safe.
