# 02 — Piper CAN and SDK

The follower Piper is accessed through `hardware/lerobot_robot_piper/`. `PiperHardware` is the hardware boundary: it connects to `piper_sdk`, reads six joint positions and gripper opening, enables/disables the arm, and sends joint/gripper targets.

- CAN interface: `<CAN_INTERFACE>`.
- Nominal bitrate: 1,000,000 bit/s, subject to the local Piper setup.
- Joint units at the LeRobot boundary: degrees.
- Gripper unit: metres, with a 0–0.07 m software range.
- Joint output order: J1 through J6.
- Motion enable, home-on-disconnect, and torque behavior are configuration-controlled.

A local operator may configure CAN only after hardware review. The archive never runs `ip link`, opens the SDK, enables motion, or writes CAN. The final Direct Joint runner has a separate read-only preflight and explicit acknowledgement/write-adapter gates; see `docs/10_direct_joint_deployment.md` and `docs/12_safety.md`.

The SDK adapter counts calls at the hardware boundary, but a returned SDK call is not a CAN acknowledgement. Do not infer task success from a write counter.
