# Direct Joint2999 Real Runner and Camera Restoration Design

**Date:** 2026-09-01  
**Status:** Approved for implementation

## Goal

Prepare a dedicated, auditable Direct Joint runner for Joint-Recovery2999 and restore the two-camera mapping so that a real rollout can be performed only after a successful read-only preflight and dry-run. The current task does not execute a real rollout or write CAN commands.

## Observed hardware and checkpoint facts

- CAN is already configured and healthy; the runner must not call `ip link`, `CreateCanBus`, bitrate setup, or any CAN interface state transition during preflight or dry-run.
- TOP is the Hy RGB Camera (`1bcf:2ced`), serial `<TOP_CAMERA_DEVICE_ID>`. Its stable V4L2 index-0 by-id path currently resolves to `/dev/video6` and supports `640x480 MJPG@30`.
- WRIST is the RealSense D435 (`8086:0b07`), current serial `<REDACTED>`. The existing `<REDACTED>` value is stale and must not be used.
- The selected checkpoint is:
  `<CHECKPOINT_ROOT>/pi05_piper_joint_recovery_mix_finetune/pi05_piper_joint_recovery_mix_103ep_3k_20260831_133508_fastfinish/2999`
- Norm statistics are loaded from the checkpoint asset and validated as seven state dimensions and seven action dimensions.
- The policy/config source semantics are absolute `[J1..J6,G]` targets, with no TCP decoding or Cartesian conversion.

## Camera restoration

Use identity-based addressing rather than a guessed numeric device index:

- TOP: `<TOP_CAMERA_DEVICE_PATH>`, OpenCV, `640x480`, `30 FPS`, `MJPG`.
- WRIST: RealSense by serial `<REDACTED>`, `640x480`, `30 FPS`, RGB output semantics matching the existing policy preprocessing.

The camera configuration change is limited to the stale wrist serial, the stable TOP path, and the TOP capture format needed to make `640x480` deterministic. No exposure, white balance, gain, or other image-processing parameters are changed. The runner records the resolved device identity, requested/actual shape, pixel order, and freshness age before it can become READY.

## Direct Joint execution architecture

The new runner is a separate entry point and must not import or call TCP, FK, IK, hybrid IK, branch selection, or `tcp_branch_fallback` code. The only command path is:

```text
fresh cameras + Piper feedback
    -> joint policy observation [J1..J6,G]
    -> policy.infer()
    -> actions [H,7]
    -> actions[0]
    -> absolute [J1..J6,G] target
    -> raw joint production safety
    -> latest-only validated-target mailbox
    -> PhaseAwareExecutor at 50 Hz
    -> final per-send safety
    -> Piper absolute Joint command
```

The runner uses the Joint-Recovery2999 config, checkpoint, norm, and prompt. It prints and validates `JOINT_CONFIG`, `JOINT_CHECKPOINT`, `ACTION_HORIZON`, `ACTION_DIM`, `PROMPT`, and `ABSOLUTE_JOINT_DEG_J1_J6_PLUS_ABSOLUTE_GRIPPER` before the dry-run proceeds.

## Timing and target policy

- Inference is N1 and consumes only `actions[0]`; N5, synchronized N10, RTC, 5 Hz action-0, TCP/IK, and new parameter changes are prohibited.
- The executor runs at 50 Hz in `NORMAL` mode, `wrist_scale=1.0`, `ramp_sec=0.50`, existing phase-aware acceleration limits, existing `vmax`, and `1 deg/tick` maximum executor step.
- Planner/inference and executor are decoupled. The mailbox has one slot; a new validated target overwrites the old target, and the executor never drains a backlog.
- Feedback, camera, and plan timestamps are carried with every target. Missing or stale feedback/camera/plan causes deceleration toward HOLD and an immediate safety stop/no-send decision rather than replaying an old trajectory.
- Joint targets are not re-anchored to measured feedback on every inference. The 5 degree raw gate compares each candidate with the previous accepted planned target; the first candidate uses fresh measured feedback as its reference.
- Gripper absolute targets use the existing driver cadence and range checks and bypass joint anti-jitter filtering, while remaining subject to final finite/range/driver safety checks.

## Safety gates

Raw candidate validation rejects without mailbox publication when any condition fails:

- non-finite action, malformed shape, or wrong dimension;
- physical joint hard limit;
- existing deployment soft limit;
- existing J2/J3 safety and J3 abnormality rule;
- gripper range;
- 5 degree continuity gate relative to the prior accepted planned target.

Every 50 Hz command is independently checked for finite values, physical limits, 1 degree/tick step, driver velocity cap, J3 safety, and gripper safety. Any rejected command is not sent. All existing hard/soft/J3 abort conditions remain active; no safety reject may be bypassed to maintain cadence.

The real path requires explicit real-run arguments and an operator acknowledgement distinct from dry-run. Camera/feedback/policy/preflight failure, stale data, driver fault, abnormal J3, command rejection, exception, operator stop, or timeout stops the run immediately. The maximum duration is 60 seconds; the runner may stop earlier.

## Dry-run and verification

Dry-run uses real camera reads, read-only Piper feedback, checkpoint/norm load, policy inference, output transform, raw gate, mailbox, and executor preview. It must set `REAL_CAN_WRITE=NO`, keep arm enable and command counters at zero, and never invoke a Piper write method.

Tests and checks include:

1. Pure unit tests for action shape/absolute semantics, raw safety gates, 5 degree reference behavior, latest-only overwrite, stale HOLD, executor 1 degree/tick behavior, and gripper bypass/range checks.
2. Static checks that the Direct Joint runner contains no TCP/FK/IK/branch imports or calls and that dry-run cannot reach CAN write methods.
3. Offline replay using captured joint observations and the fixed checkpoint/output transform.
4. Python compilation/lint checks.
5. Actual-camera/read-only-feedback dry-run, including resolved identities and freshness.

## Logging and final report

Each planner/executor cycle logs timestamp, feedback joints/gripper, raw action-0, raw target, raw-gate result/reason, previous accepted target, executor command, sent target, joint deltas/velocities, gripper command, inference latency, target age, camera ages, CAN state, and driver state.

The final report must include the requested READY/preflight fields and, for any eventual real run, `REAL_ROBOT_EXECUTED`, duration, stop reason, task phase, startup smoothness, J3 abnormality, visible jitter, stop-and-go, policy progress, gripper behavior, reject counters, verdict, and artifact paths. If behavior is clearly poor, it records `FREEZE_CURRENT_EXPERIMENT=YES`; otherwise `NO`.

## Non-goals

- No real rollout in this implementation/preflight task.
- No CAN interface reconfiguration, enable/disable transition, or command write during dry-run.
- No TCP policy compatibility layer, Cartesian fallback, new safety thresholds, or parameter tuning.
