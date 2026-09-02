# Direct Joint2999 Real Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a camera-restored, Direct Joint `[J1..J6,G]` Joint-Recovery2999 runner whose dry-run is read-only and whose real path is protected by explicit operator and safety gates.

**Architecture:** Add the missing Piper policy/config transforms to OpenPI, a local identity-based camera manifest, and a pure joint execution core with no robot/TCP imports. The runner composes that core with the existing Piper camera/read interfaces and policy factory; inference publishes only the newest validated absolute target, while a 50 Hz executor performs final safety checks before any future Piper command.

**Tech Stack:** Python 3.11+, NumPy, OpenPI transforms/policy factory, existing Piper/LeRobot camera APIs, OpenCV/V4L2, RealSense, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-direct-joint2999-real-runner-design.md`

## Global Constraints

- Do not run `ip link`, `CreateCanBus`, bitrate setup, CAN down/up, or any CAN interface reconfiguration.
- Dry-run must perform no CAN write, arm enable, gripper command, or robot motion command.
- Use `pi05_piper_joint_recovery_mix_finetune` checkpoint `.../fastfinish/2999`, its checkpoint norm stats, and the verified absolute `[J1..J6,G]` output semantics.
- Use N1 and `actions[0]`; do not enable N5, N10 synchronization, RTC, 5 Hz action-0, TCP, FK, IK, hybrid IK, branch selection, or `tcp_branch_fallback`.
- Preserve NORMAL, 50 Hz, `wrist_scale=1.0`, `ramp_sec=0.50`, existing vmax, 5 degree raw gate, 1 degree/tick executor gate, and existing hard/soft/J3 safety.
- Use TOP by-id identity `usb-<TOP_CAMERA_DEVICE_ID>-video-index0` with `640x480 MJPG@30` and WRIST RealSense serial `<REDACTED>` at `640x480@30`.
- Do not modify the original dirty checkout; all tracked implementation changes occur on `codex/direct-joint2999`.

---

### Task 1: Add the Piper joint policy transforms and training config

**Files:**
- Create: `src/openpi/policies/piper_policy.py`
- Create: `src/openpi/policies/piper_policy_test.py`
- Modify: `src/openpi/training/config.py` at the policy imports, `DataConfigFactory` subclasses, and `_CONFIGS` list

**Interfaces:**
- Produces `PiperInputs.__call__(data) -> dict`, `PiperOutputs.__call__(data) -> dict`, and `LeRobotPiperDataConfig`.
- `PiperInputs` accepts images `top`/`wrist` as `[C,H,W]`, state `[7]`, optional actions `[H,7]`, and prompt; it returns model keys `image`, `image_mask`, `state`, optional `actions`, and optional `prompt`.
- `PiperOutputs` accepts model output actions with at least seven columns and returns exactly `{"actions": float32[N,7]}`.
- `LeRobotPiperDataConfig` applies first-six-joint `DeltaActions` and inverse `AbsoluteActions`, leaves gripper dimension absolute, and uses the Piper image transform.
- `get_config("pi05_piper_joint_recovery_mix_finetune")` returns a Pi0.5 config with action horizon 10, action dimension 32 at model level, local Piper joint dataset, and the checkpoint asset id.

- [ ] **Step 1: Write the failing policy/config tests**

```python
def test_piper_inputs_uses_top_and_wrist_and_masks_missing_right_wrist():
    data = PiperInputs()({
        "images": {
            "top": np.zeros((3, 4, 5), dtype=np.uint8),
            "wrist": np.ones((3, 4, 5), dtype=np.uint8),
        },
        "state": np.arange(7, dtype=np.float32),
        "prompt": "pick up the battery and place it into the target location",
    })
    assert data["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert data["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert bool(data["image_mask"]["right_wrist_0_rgb"]) is False
    np.testing.assert_array_equal(data["state"], np.arange(7, dtype=np.float32))


def test_piper_outputs_returns_absolute_physical_prefix():
    result = PiperOutputs()({"actions": np.arange(32, dtype=np.float32).reshape(4, 8)})
    assert result["actions"].shape == (4, 7)
    np.testing.assert_array_equal(result["actions"], np.arange(28, dtype=np.float32).reshape(4, 7))


def test_joint_recovery_config_declares_joint_representation():
    config = training_config.get_config("pi05_piper_joint_recovery_mix_finetune")
    assert config.model.action_horizon == 10
    assert config.data.repo_id == "local/piper_joint_recovery_mix_103ep"
    assert config.data.__class__.__name__ == "LeRobotPiperDataConfig"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest -q src/openpi/policies/piper_policy_test.py`

Expected: collection or assertion failure because the Piper policy/config symbols do not yet exist on the clean `HEAD`.

- [ ] **Step 3: Implement the minimal transforms and config**

Implement `_convert_image` as `[C,H,W] -> [H,W,C]`, preserve native degree/meter units, create the three model image slots with an explicit false mask for the absent right wrist, and add the joint config using the existing `DataConfigFactory`, `ModelTransformFactory`, `DeltaActions`, and `AbsoluteActions` APIs. Do not add TCP classes or TCP imports to this task.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `pytest -q src/openpi/policies/piper_policy_test.py`

Expected: PASS in an environment with OpenPI test dependencies. If the environment still lacks `pynvml`, record that collection blocker and run the same assertions through the deployment interpreter after the module is compiled.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/policies/piper_policy.py src/openpi/policies/piper_policy_test.py src/openpi/training/config.py
git commit -m "feat: add Piper joint policy transforms"
```

### Task 2: Add identity-based camera manifest and verification utility

**Files:**
- Create: `configs/piper_joint2999_camera.yaml`
- Create: `scripts/verify_piper_joint2999_cameras.py`
- Create: `scripts/test_verify_piper_joint2999_cameras.py`

**Interfaces:**
- Manifest keys are `top.kind=opencv`, `top.path=<TOP_CAMERA_DEVICE_PATH>`, `top.fourcc=MJPG`, `top.width=640`, `top.height=480`, `top.fps=30`; `wrist.kind=intelrealsense`, `wrist.serial=<REDACTED>`, `wrist.width=640`, `wrist.height=480`, `wrist.fps=30`.
- `verify_piper_joint2999_cameras.py` exposes `load_camera_manifest(path)`, `verify_manifest_shape(manifest)`, and `probe_cameras(manifest, read_frames=True) -> dict`; probing reports resolved identity, actual shape, pixel order, frame timestamp, and freshness age without enabling the arm or writing CAN.
- The script exits nonzero on missing identity, wrong shape, failed frame, or stale frame and never falls back from the declared by-id path to a numeric index.

- [ ] **Step 1: Write manifest/verification tests**

```python
def test_manifest_declares_stable_top_and_current_wrist_identity(tmp_path):
    manifest = load_camera_manifest(Path("configs/piper_joint2999_camera.yaml"))
    verify_manifest_shape(manifest)
    assert manifest["top"]["path"].endswith("video-index0")
    assert manifest["top"]["fourcc"] == "MJPG"
    assert manifest["wrist"]["serial"] == "<REDACTED>"


def test_manifest_rejects_stale_wrist_serial():
    with pytest.raises(ValueError, match="<REDACTED>"):
        verify_manifest_shape({"top": VALID_TOP, "wrist": {**VALID_WRIST, "serial": "<REDACTED>"}})
```

The test module defines the fixtures used above exactly as follows:

```python
VALID_TOP = {"kind": "opencv", "path": "<TOP_CAMERA_DEVICE_PATH>", "fourcc": "MJPG", "width": 640, "height": 480, "fps": 30}
VALID_WRIST = {"kind": "intelrealsense", "serial": "<REDACTED>", "width": 640, "height": 480, "fps": 30}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q scripts/test_verify_piper_joint2999_cameras.py`

Expected: FAIL because the manifest and utility do not exist.

- [ ] **Step 3: Implement the manifest and read-only probe**

Parse YAML with the project dependency, validate exact identity and dimensions, configure OpenCV with `MJPG` before width/height, and use the RealSense serial rather than a `/dev/videoN` guess. The probe must close cameras in `finally` and emit JSON-safe results.

- [ ] **Step 4: Run tests and a hardware read-only probe**

Run: `pytest -q scripts/test_verify_piper_joint2999_cameras.py`.

Then run: `<USER_HOME>/.conda/envs/pi05_piper_deploy/bin/python scripts/verify_piper_joint2999_cameras.py --manifest configs/piper_joint2999_camera.yaml`.

Expected: tests pass; hardware output identifies the Hy RGB device and D435 serial, both frames are `640x480`, and no CAN or arm command appears.

- [ ] **Step 5: Commit**

```bash
git add configs/piper_joint2999_camera.yaml scripts/verify_piper_joint2999_cameras.py scripts/test_verify_piper_joint2999_cameras.py
git commit -m "feat: restore Joint2999 camera identity mapping"
```

### Task 3: Implement pure Direct Joint safety, mailbox, and 50 Hz executor core

**Files:**
- Create: `src/openpi/rollouts/__init__.py`
- Create: `src/openpi/rollouts/direct_joint_core.py`
- Create: `src/openpi/rollouts/test_direct_joint_core.py`

**Interfaces:**
- `validate_raw_target(raw_target_7, reference_7) -> tuple[bool, str | None]` checks shape, finite values, physical limits, deployment soft/J2/J3 safety, gripper range, and the 5 degree joint continuity gate.
- `validate_executor_command(q_next_6, previous_q_6, dt, driver_speed_cap_deg_s) -> tuple[bool, str | None]` checks finite values, physical limits, `max(abs(delta)) <= 1`, and per-joint velocity cap.
- `LatestOnlyTargetMailbox.put(target: Target) -> None`, `.take() -> Target | None`, and `.dropped_count -> int`; `put` overwrites the pending item and never queues.
- `PhaseAwareExecutor(control_hz=50.0, ramp_sec=0.50).step(commanded_6, target_6, phase="NORMAL", wrist_scale=1.0) -> np.ndarray` uses `vmax=[10,12,12,3,3,3]`, acceleration `vmax/ramp_sec`, braking/no-overshoot, and a one-degree/tick cap.
- `gripper_send_decision(target_m, last_sent_m, last_sent_ts, now) -> tuple[bool, bool]` preserves material-change `0.0005 m` and five-second keepalive semantics without joint anti-jitter filtering.

- [ ] **Step 1: Write failing pure tests**

```python
def test_raw_gate_uses_previous_accepted_target_not_live_feedback():
    feedback = np.zeros(7)
    first = np.array([4.0, 0, 0, 0, 0, 0, 0.02])
    second = np.array([8.5, 0, 0, 0, 0, 0, 0.02])
    assert validate_raw_target(first, feedback) == (True, None)
    assert validate_raw_target(second, first) == (True, None)
    assert validate_raw_target(second, feedback)[0] is False


def test_executor_never_exceeds_one_degree_per_tick():
    executor = PhaseAwareExecutor(50.0, 0.50)
    previous = np.zeros(6)
    for _ in range(100):
        current = executor.step(previous, np.full(6, 30.0))
        assert np.max(np.abs(current - previous)) <= 1.0 + 1e-9
        previous = current


def test_mailbox_overwrites_without_backlog():
    mailbox = LatestOnlyTargetMailbox()
    mailbox.put(Target(np.zeros(7), 1.0, 1))
    mailbox.put(Target(np.ones(7), 2.0, 2))
    assert mailbox.take().generation_id == 2
    assert mailbox.take() is None
    assert mailbox.dropped_count == 1
```

- [ ] **Step 2: Run pure tests to verify they fail**

Run: `pytest -q src/openpi/rollouts/test_direct_joint_core.py`

Expected: FAIL because the core module does not exist.

- [ ] **Step 3: Implement the core without robot/policy/camera imports**

Define explicit seven-dimensional target dataclasses and constants in this module. Port only the existing NORMAL PhaseAwareExecutor behavior and safety limits; do not import `piper_tcp_kinematics`, `piper_tcp_convert`, `tcp_branch_fallback`, FK, IK, or any robot SDK. The continuity reference advances only after a target passes raw validation.

- [ ] **Step 4: Run pure tests and a static import audit**

Run: `pytest -q src/openpi/rollouts/test_direct_joint_core.py`.

Run: `rg -n "tcp|fk|ik|branch|Piper|can|camera" src/openpi/rollouts/direct_joint_core.py`.

Expected: tests pass; the import audit returns no forbidden implementation dependency.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/rollouts
git commit -m "feat: add pure Direct Joint safety executor"
```

### Task 4: Build the read-only preflight and Direct Joint2999 runner

**Files:**
- Create: `scripts/pi05_joint2999_rollout.py`
- Create: `scripts/test_pi05_joint2999_rollout.py`

**Interfaces:**
- `build_joint_observation(top_rgb, wrist_rgb, joint_deg_6, gripper_m, prompt) -> dict` returns the exact raw input expected by `PiperInputs`.
- `load_joint_policy(checkpoint_dir, config_name, prompt) -> tuple[Policy, dict]` calls `training_config.get_config`, loads the checkpoint norm stats through `create_trained_policy`, and returns metadata including config, checkpoint, horizon, action dim, norm path/hash, and prompt.
- `run_preflight(args) -> dict` reports all requested preflight fields and requires `OPERATOR_ONSITE=YES`, `ESTOP_READY=YES`, and `WORKSPACE_CLEAR=YES` without enabling motion.
- `run_dry_run(args) -> dict` runs camera reads, read-only CAN feedback, policy inference, raw validation, mailbox, executor preview, and logging with `REAL_CAN_WRITE=NO`, `ARM_ENABLE_COUNT=0`, and `COMMAND_SEND_COUNT=0`.
- `main()` accepts `--dry-run`, `--manifest`, `--checkpoint`, `--prompt`, `--max-time`, `--output-dir`, and a separate `--real` mode that additionally requires `--operator-ack` and the exact acknowledgement string `I_UNDERSTAND_JOINT2999_REAL_RUN`.

- [ ] **Step 1: Write failing runner contract tests**

```python
def test_observation_is_joint_native_and_has_no_tcp_fields():
    obs = build_joint_observation(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640, 3), dtype=np.uint8),
        np.zeros(6), 0.03,
        "pick up the battery and place it into the target location",
    )
    assert obs["state"].shape == (7,)
    assert set(obs) == {"images", "state", "prompt"}
    assert not any("tcp" in key.lower() for key in obs)


def test_dry_run_contract_has_zero_write_counters(tmp_path):
    report = run_dry_run(
        argparse.Namespace(dry_run=True, max_time=0.05, output_dir=tmp_path),
        dependencies=FakeReadOnlyDependencies(),
    )
    assert report["REAL_CAN_WRITE"] == "NO"
    assert report["ARM_ENABLE_COUNT"] == 0
    assert report["COMMAND_SEND_COUNT"] == 0


def test_real_mode_requires_distinct_acknowledgement():
    with pytest.raises(SystemExit, match="I_UNDERSTAND_JOINT2999_REAL_RUN"):
        parse_args(["--real"])
```

The runner tests define `FakeReadOnlyDependencies` with a clock returning monotonic values, a camera source returning one `640x480x3` TOP/WRIST pair, a feedback source returning seven finite values, a policy whose `infer` returns `{"actions": np.zeros((10, 7), dtype=np.float32)}`, and a command sink whose `send_joint`/`send_gripper` methods raise `AssertionError`. The dry-run must therefore finish with zero calls to either command method.

- [ ] **Step 2: Run contract tests to verify they fail**

Run: `pytest -q scripts/test_pi05_joint2999_rollout.py`

Expected: FAIL because the new runner and dependency-injection seams do not exist.

- [ ] **Step 3: Implement policy/camera/feedback composition**

Use the local joint config and camera manifest. Construct observations from only six joint angles plus gripper and RGB images; call `policy.infer` once per fresh observation, select `actions[0]`, require shape `[7]` after the output transform, and pass the absolute target to `validate_raw_target` and the mailbox. Use a read-only CAN feedback reader/socket for preflight and dry-run so no SDK initialization path can alter the already-configured interface. Keep all hardware-dependent objects behind injected interfaces so tests can prove no write method is called.

- [ ] **Step 4: Implement 50 Hz dry-run loop and guarded real loop**

At each executor tick read fresh feedback, consume only the latest target, hold/decelerate when plan/camera/feedback age exceeds `0.50 s`, run final safety, and in dry-run record a preview instead of invoking a command method. The real path must refuse to start unless all preflight fields pass, the operator acknowledgement is present, and the command adapter reports the host interface is already UP/ERROR-ACTIVE at 1 Mbps without attempting to configure it. The real path writes only through the Piper absolute Joint adapter after the final safety gate.

- [ ] **Step 5: Add structured logging and final report fields**

Write JSONL records containing feedback, raw action, raw target, gate reason, prior accepted target, executor command, send decision, deltas/velocities, gripper state, inference latency, target/camera/CAN ages, and driver status. Emit all requested `PIPER_*`, `ARM_STATUS`, `DRIVER_FAULT`, camera, GPU/JAX, checkpoint, norm, prompt, rollout, safety reject, artifact, and verdict fields. Dry-run emits `REAL_ROBOT_EXECUTED=NO`, `STOP_REASON=DRY_RUN_COMPLETE`, and zero sends.

- [ ] **Step 6: Run runner contract tests and static forbidden-path audit**

Run: `pytest -q scripts/test_pi05_joint2999_rollout.py`.

Run: `rg -n "piper_tcp|tcp_branch|decode_tcp|fk_|ik_|inverse_kinematics|branch" scripts/pi05_joint2999_rollout.py`.

Expected: tests pass; the forbidden-path audit returns no matches.

- [ ] **Step 7: Commit**

```bash
git add scripts/pi05_joint2999_rollout.py scripts/test_pi05_joint2999_rollout.py
git commit -m "feat: add guarded Direct Joint2999 runner"
```

### Task 5: Run checkpoint, camera, policy, and offline replay verification

**Files:**
- Modify: `scripts/pi05_joint2999_rollout.py` only if verification exposes a defect
- Create: `scripts/replay_direct_joint2999.py`
- Create: `scripts/test_replay_direct_joint2999.py`

**Interfaces:**
- `replay_direct_joint2999(input_jsonl, output_jsonl, checkpoint_dir) -> dict` runs the same policy-output and pure safety path with recorded observations and no hardware imports.
- Replay output includes per-frame raw action, accepted/rejected reason, executor preview, and reject counters.

- [ ] **Step 1: Write replay tests with finite, malformed, and stale cases**

```python
def test_replay_rejects_nonfinite_action_and_keeps_send_count_zero(tmp_path):
    input_path = tmp_path / "nan.jsonl"
    input_path.write_text(json.dumps({"timestamp": 1.0, "state": [0, 0, 0, 0, 0, 0, 0.03], "action": [float("nan")] * 7}) + "\n")
    report = replay_direct_joint2999(input_path, tmp_path / "out.jsonl", tmp_path / "fake-checkpoint")
    assert report["SAFETY_REJECT_COUNT"] == 1
    assert report["COMMAND_SEND_COUNT"] == 0


def test_replay_latest_only_path_has_no_backlog(tmp_path):
    input_path = tmp_path / "two-targets.jsonl"
    rows = [
        {"timestamp": 1.0, "generation_id": 1, "state": [0, 0, 0, 0, 0, 0, 0.03], "action": [1, 0, 0, 0, 0, 0, 0.03]},
        {"timestamp": 1.001, "generation_id": 2, "state": [0, 0, 0, 0, 0, 0, 0.03], "action": [2, 0, 0, 0, 0, 0, 0.03]},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = replay_direct_joint2999(input_path, tmp_path / "out.jsonl", tmp_path / "fake-checkpoint")
    assert report["MAX_CONSECUTIVE_REJECTS"] == 0
    assert report["LATEST_TARGET_GENERATION"] == 2
```

The replay test module imports `json`, `numpy as np`, and the replay function. The replay implementation accepts the temporary checkpoint path in these pure tests without opening it; checkpoint loading is covered separately by the real load-only verification.

- [ ] **Step 2: Run replay tests to verify they fail**

Run: `pytest -q scripts/test_replay_direct_joint2999.py`

Expected: FAIL because the replay utility does not exist.

- [ ] **Step 3: Implement offline replay using the pure core**

Do not import cameras, Piper SDK, CAN, or TCP modules. Reuse the same raw/final gates and executor constants as the runner, and make the checkpoint path an explicit argument. The replay must preserve generation IDs and latest-only overwrite behavior.

- [ ] **Step 4: Run compile, lint, focused tests, and replay**

Run:

```bash
python -m py_compile src/openpi/policies/piper_policy.py src/openpi/rollouts/direct_joint_core.py scripts/pi05_joint2999_rollout.py scripts/replay_direct_joint2999.py
pytest -q src/openpi/rollouts/test_direct_joint_core.py scripts/test_verify_piper_joint2999_cameras.py scripts/test_pi05_joint2999_rollout.py scripts/test_replay_direct_joint2999.py
ruff check src/openpi/policies/piper_policy.py src/openpi/rollouts scripts/pi05_joint2999_rollout.py scripts/replay_direct_joint2999.py
```

Expected: compile and pure tests pass. If collection is blocked by missing `pynvml`, record it explicitly and run compile plus deployment-interpreter smoke assertions; do not install packages as part of the robot safety task.

- [ ] **Step 5: Load the real checkpoint and verify GPU/JAX**

Run the runner’s load-only mode with the deployment interpreter and the exact `2999` checkpoint. Confirm `CHECKPOINT_LOAD=PASS`, norm dimension/hash, `ACTION_HORIZON=10`, post-transform `ACTION_DIM=7`, `JAX_GPU=PASS`, and output semantics `ABSOLUTE_JOINT_DEG_J1_J6_PLUS_ABSOLUTE_GRIPPER`.

- [ ] **Step 6: Commit replay and verification fixes**

```bash
git add scripts/replay_direct_joint2999.py scripts/test_replay_direct_joint2999.py scripts/pi05_joint2999_rollout.py
git commit -m "test: verify Direct Joint2999 replay safety"
```

### Task 6: Execute read-only camera/Piper dry-run and package READY report

**Files:**
- Create: `scripts/report_direct_joint2999_preflight.py`
- Modify: `scripts/pi05_joint2999_rollout.py` only for verified dry-run defects
- Generated outside Git: `/tmp/direct_joint2999_preflight/` or an explicitly supplied output directory

**Interfaces:**
- `report_direct_joint2999_preflight(report_json, log_jsonl, video_path=None) -> dict` validates that no dry-run write counter is nonzero and formats the requested final fields.

- [ ] **Step 1: Run the camera-only read-only probe**

Run: `<USER_HOME>/.conda/envs/pi05_piper_deploy/bin/python scripts/verify_piper_joint2999_cameras.py --manifest configs/piper_joint2999_camera.yaml`.

Expected: TOP and WRIST identity, shape, and freshness PASS; no arm enable and no command write.

- [ ] **Step 2: Run load-only policy preflight**

Run the runner in load-only mode with the 2999 checkpoint and fixed prompt. Expected: config/checkpoint/norm/prompt/GPU/JAX fields PASS and no camera or CAN writes.

- [ ] **Step 3: Run full dry-run with real cameras and read-only feedback**

Run: `<USER_HOME>/.conda/envs/pi05_piper_deploy/bin/python scripts/pi05_joint2999_rollout.py --dry-run --manifest configs/piper_joint2999_camera.yaml --max-time 60 --output-dir /tmp/direct_joint2999_preflight`.

Expected: `REAL_ROBOT_EXECUTED=NO`, `REAL_CAN_WRITE=NO`, `COMMAND_SEND_COUNT=0`, `ARM_ENABLE_COUNT=0`; all preflight gates PASS or a concrete STOP_REASON is reported. No `can0` state transition or bitrate configuration is attempted.

- [ ] **Step 4: Validate the generated report and artifacts**

Run: `<USER_HOME>/.conda/envs/pi05_piper_deploy/bin/python scripts/report_direct_joint2999_preflight.py --report /tmp/direct_joint2999_preflight/report.json --log /tmp/direct_joint2999_preflight/rollout.jsonl`.

Expected: report contains every requested field, exact config freeze, safety reject counters, and `VIDEO`, `LOG`, `REPORT` paths (or explicit `N/A` for video when not recorded).

- [ ] **Step 5: Final verification and handoff**

Run `git status --short --branch`, `git log -3 --oneline`, `git diff --check`, and repeat the original checkout status check. Confirm only the isolated branch changed, the original dirty checkout remains byte-for-byte at its prior status, and no real rollout was executed.

Expected final handoff states: `REAL_ROBOT_EXECUTED=NO`, `TASK_COMPLETE=NO` for the unexecuted real task, `TASK_PHASE_REACHED=PREPARED_DRY_RUN` when dry-run passes, and `FREEZE_CURRENT_EXPERIMENT` based only on observed dry-run behavior.

- [ ] **Step 6: Commit the READY report utility**

```bash
git add scripts/report_direct_joint2999_preflight.py
git commit -m "chore: report Direct Joint2999 preflight"
```
