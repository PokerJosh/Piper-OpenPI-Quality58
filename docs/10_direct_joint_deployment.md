# 10 — Direct Joint Deployment

The final runner is `scripts/pi05_joint2999_rollout.py`, with pure safety primitives in `src/openpi/rollouts/direct_joint_core.py` and camera verification in `scripts/verify_piper_joint2999_cameras.py`.

The deployment contract is:

- model output: 32-dimensional latent/model action;
- physical action: 7 values `[J1,J2,J3,J4,J5,J6,gripper]`;
- J1–J6: degrees;
- gripper: metres;
- action horizon: 20.

The runner has a read-only preflight, explicit config/checkpoint identity, camera and feedback freshness checks, and a separate explicit write-adapter/acknowledgement requirement for real mode. Its default dry-run does not construct a robot writer. The sanitized camera configuration requires local device paths and serials.

A local operator must independently review the start pose, workspace, emergency stop, driver health, and action semantics. This archive does not recommend running the real mode; it records the final historical result in `docs/11_real_robot_run.md`.
