# 01 — Environment Setup

Use a dedicated Python 3.12 environment for the LeRobot/Piper integration and a compatible OpenPI environment for π0.5 training/evaluation. Install the local LeRobot Piper robot adapter and Piper leader teleoperator from `hardware/lerobot_robot_piper/` and `teleop/lerobot_teleoperator_piper_leader/`.

The archived install wrapper is `hardware/piper_upper_pc/install_piper_upper_pc.sh`. Set `LEROBOT_V2_ROOT=<PIPER_PROJECT_ROOT>` in a local working copy. The wrapper expects the upstream LeRobot tree plus the two local plugins.

Typical host prerequisites are git, FFmpeg, CAN utilities, V4L2 tools, serial permissions, and the vendor Python SDKs. Exact versions should be pinned in the local environment; they are not embedded here.

Before touching hardware, verify emergency-stop access, clear workspace, arm mounting, cable routing, and that all device identifiers are local placeholders. `REQUIRES_UPSTREAM_OPENPI_TREE` applies to selective archive imports.
