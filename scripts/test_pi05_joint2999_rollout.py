from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

import scripts.pi05_joint2999_rollout as rollout
from scripts.pi05_joint2999_rollout import REAL_ACKNOWLEDGEMENT
from scripts.pi05_joint2999_rollout import build_joint_observation
from scripts.pi05_joint2999_rollout import parse_args
from scripts.pi05_joint2999_rollout import run_dry_run
from scripts.pi05_joint2999_rollout import run_preflight
from scripts.pi05_joint2999_rollout import run_real


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        time.sleep(0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CommandSinkThatMustNotSend:
    def __init__(self) -> None:
        self.previews: list[dict] = []

    def preview(self, command: dict) -> None:
        self.previews.append(command)

    def send_arm_absolute(self, *_args, **_kwargs) -> None:
        raise AssertionError("dry-run must not write arm commands")

    def send_gripper(self, *_args, **_kwargs) -> None:
        raise AssertionError("dry-run must not write gripper commands")

    def hold_stop(self, *_args, **_kwargs) -> None:
        raise AssertionError("dry-run must not write hold/stop commands")

    def enable(self) -> None:
        raise AssertionError("dry-run must not enable the arm")


class RecordingWriteAdapter:
    explicit_real_write_adapter = True

    def __init__(self, event_log=None) -> None:
        self.event_log = event_log
        self.enable_count = 0
        self.arm_commands: list[np.ndarray] = []
        self.gripper_commands: list[float] = []
        self.stop_reasons: list[str] = []
        self.successful_can_write_count = 0

    def ready(self) -> bool:
        return True

    def enable(self) -> None:
        if self.event_log is not None:
            self.event_log.append("enable")
        self.enable_count += 1
        self.successful_can_write_count += 2

    def send_arm_absolute(self, command) -> None:
        if self.event_log is not None:
            self.event_log.append("send_arm_absolute")
        self.arm_commands.append(np.asarray(command).copy())
        self.successful_can_write_count += 3

    def send_gripper(self, target_m: float) -> None:
        if self.event_log is not None:
            self.event_log.append("send_gripper")
        self.gripper_commands.append(float(target_m))
        self.successful_can_write_count += 1

    def hold_stop(self, reason: str) -> None:
        if self.event_log is not None:
            self.event_log.append("hold_stop")
        self.stop_reasons.append(reason)
        self.successful_can_write_count += 1


class PartialEnableWriteAdapter(RecordingWriteAdapter):
    def enable(self) -> None:
        self.enable_count += 1
        self.successful_can_write_count += 1
        raise RuntimeError("second enable CAN frame failed")


class PartialArmWriteAdapter(RecordingWriteAdapter):
    def send_arm_absolute(self, command) -> None:
        self.successful_can_write_count += 1
        raise RuntimeError("second arm CAN frame failed")


class DeadlineJumpingPreviewSink(CommandSinkThatMustNotSend):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.preview_timestamps: list[float] = []

    def preview(self, command: dict) -> None:
        super().preview(command)
        self.preview_timestamps.append(self.clock.now)
        if len(self.preview_timestamps) == 1:
            self.clock.advance(3.5 * rollout.CONTROL_DT)


class FakeRuntimeDependencies:
    """Read-only camera, feedback, and policy boundary with deterministic time."""

    def __init__(
        self,
        *,
        frame_keys=(1,),
        inference_delay=0.0,
        action=None,
        driver_statuses=None,
        event_log=None,
        fault_after_feedback_reads=None,
    ) -> None:
        self.clock = FakeClock()
        self.command_sink = CommandSinkThatMustNotSend()
        self.frame_keys = list(frame_keys)
        self.inference_delay = inference_delay
        self.action = np.zeros(7, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
        self.driver_statuses = list(driver_statuses or [])
        self.event_log = event_log
        self.fault_after_feedback_reads = fault_after_feedback_reads
        self.camera_probe_count = 0
        self.feedback_probe_count = 0
        self.camera_read_count = 0
        self.feedback_read_count = 0
        self.infer_count = 0
        self._frame_timestamps: dict[object, float] = {}

    def monotonic(self) -> float:
        return self.clock.monotonic()

    def sleep(self, seconds: float) -> None:
        self.clock.sleep(seconds)

    def read_camera_pair(self):
        if self.event_log is not None:
            self.event_log.append("read_camera_pair")
        index = min(self.camera_read_count, len(self.frame_keys) - 1)
        frame_key = self.frame_keys[index]
        self.camera_read_count += 1
        self._frame_timestamps.setdefault(frame_key, self.clock.now)
        return (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8),
            self._frame_timestamps[frame_key],
            frame_key,
        )

    def read_feedback(self):
        if self.event_log is not None:
            self.event_log.append("read_feedback")
        self.feedback_read_count += 1
        return np.zeros(7, dtype=np.float64), self.clock.now

    def infer(self, _observation):
        if self.event_log is not None:
            self.event_log.append("infer")
        self.infer_count += 1
        self.clock.advance(self.inference_delay)
        return {"actions": np.tile(self.action, (10, 1))}

    def probe_cameras(self, manifest):
        self.camera_probe_count += 1
        return {
            "top": {
                "resolved_identity": manifest["top"]["path"],
                "actual_shape": {"width": 640, "height": 480, "channels": 3},
                "freshness_verifiable": True,
                "freshness_age_seconds": 0.01,
                "freshness_contract": "bounded_host_receipt_sequence_liveness",
                "timestamp_source": "host_receipt_monotonic_not_sensor_capture_time",
                "source_capture_timestamp_available": False,
                "host_receipt_monotonic_seconds": self.clock.now,
                "worker_sequence": 2,
                "bounded_buffer_capacity": 1,
            },
            "wrist": {
                "resolved_identity": manifest["wrist"]["serial"],
                "actual_shape": {"width": 640, "height": 480, "channels": 3},
                "freshness_verifiable": True,
                "freshness_age_seconds": 0.01,
                "timestamp_source": "realsense_system_time_mapped_to_host_monotonic",
                "source_capture_timestamp_available": True,
            },
        }

    def probe_feedback(self):
        self.feedback_probe_count += 1
        return {
            "feedback": np.zeros(7, dtype=np.float64),
            "feedback_timestamp": self.clock.now,
            "feedback_age_seconds": 0.0,
            "driver_status": healthy_driver_status(self.clock.now),
        }

    def driver_status(self):
        if self.event_log is not None:
            self.event_log.append("driver_status")
        if (
            self.fault_after_feedback_reads is not None
            and self.feedback_read_count >= self.fault_after_feedback_reads
        ):
            return faulted_driver_status(self.clock.now)
        if self.driver_statuses:
            return self.driver_statuses.pop(0)
        return healthy_driver_status(self.clock.now)


class PostEnableFailureDependencies(FakeRuntimeDependencies):
    def read_feedback(self):
        if self.feedback_read_count >= 2:
            raise RuntimeError("post-enable feedback failed")
        return super().read_feedback()


class BlockingCameraDependencies(FakeRuntimeDependencies):
    def __init__(self) -> None:
        super().__init__()
        self.camera_blocked = threading.Event()
        self.release_camera = threading.Event()
        self.camera_returned = threading.Event()
        self.close_count = 0

    def read_camera_pair(self):
        self.camera_blocked.set()
        self.release_camera.wait(timeout=2.0)
        self.camera_returned.set()
        raise RuntimeError("camera read cancelled")

    def close(self) -> None:
        self.close_count += 1


class ThreadCheckedRuntimeDependencies(FakeRuntimeDependencies):
    def read_camera_pair(self):
        assert threading.current_thread().name == "joint2999-camera-inference"
        return super().read_camera_pair()

    def infer(self, observation):
        assert threading.current_thread().name == "joint2999-camera-inference"
        return super().infer(observation)


def healthy_driver_status(timestamp: float) -> dict:
    return {
        "healthy": True,
        "timestamp": timestamp,
        "arm_status": 0,
        "arm_error_code": 0,
        "driver_fault_codes": [0, 0, 0, 0, 0, 0],
        "gripper_status_code": 0,
    }


def faulted_driver_status(timestamp: float) -> dict:
    status = healthy_driver_status(timestamp)
    status.update({"healthy": False, "driver_fault_codes": [0, 0, 32, 0, 0, 0]})
    return status


def make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "dry_run": True,
        "real": False,
        "manifest": Path("configs/piper_joint2999_camera.yaml"),
        "checkpoint": rollout.CHECKPOINT,
        "config_name": rollout.CONFIG_NAME,
        "prompt": rollout.PROMPT,
        "max_time": 0.05,
        "output_dir": tmp_path,
        "operator_ack": None,
        "write_adapter": None,
        "can_interface": "can0",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def preflight_environment(monkeypatch):
    monkeypatch.setenv("OPERATOR_ONSITE", "YES")
    monkeypatch.setenv("ESTOP_READY", "YES")
    monkeypatch.setenv("WORKSPACE_CLEAR", "YES")


def test_observation_is_joint_native_and_has_no_tcp_fields():
    """Removing native joint state or adding pose fields must break this contract."""
    obs = build_joint_observation(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640, 3), dtype=np.uint8),
        np.zeros(6),
        0.03,
        "pick up the battery and place it into the target location",
    )

    assert obs["state"].shape == (7,)
    assert set(obs) == {"images", "state", "prompt"}
    assert not any("tcp" in key.lower() for key in obs)


def test_policy_decode_uses_only_n1_actions_zero_as_absolute_joint_target():
    """Flattening the horizon or selecting a later row would change the physical target."""
    actions = np.array(
        [
            [1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 0.02],
            [11.0, 12.0, -13.0, 14.0, 15.0, 16.0, 0.06],
        ],
        dtype=np.float32,
    )

    decoded = rollout._first_action({"actions": actions})

    np.testing.assert_array_equal(decoded, actions[0])
    assert decoded.shape == (7,)


def test_dry_run_runs_bounded_multi_tick_loop_without_any_write(tmp_path):
    """Removing the bounded executor loop would reduce three 50 Hz previews to one."""
    dependencies = FakeRuntimeDependencies(frame_keys=(1,) * 3)

    report = run_dry_run(make_args(tmp_path), dependencies=dependencies)

    assert report["REAL_CAN_WRITE"] == "NO"
    assert report["ARM_ENABLE_COUNT"] == 0
    assert report["COMMAND_SEND_COUNT"] == 0
    assert report["REAL_ROBOT_EXECUTED"] == "NO"
    assert report["preview_count"] == 3
    assert dependencies.feedback_read_count >= 2 * report["preview_count"] + 1
    assert (tmp_path / "joint2999_dry_run.jsonl").is_file()


def test_missed_50hz_deadline_rebases_without_catch_up_burst(tmp_path):
    """Leaving next_tick in the past would dispatch several previews at one timestamp."""
    dependencies = FakeRuntimeDependencies(frame_keys=(1,))
    sink = DeadlineJumpingPreviewSink(dependencies.clock)
    dependencies.command_sink = sink

    run_dry_run(make_args(tmp_path, max_time=0.14), dependencies=dependencies)

    assert len(sink.preview_timestamps) >= 3
    gaps = np.diff(sink.preview_timestamps)
    assert np.all(gaps >= rollout.CONTROL_DT - 1e-9), sink.preview_timestamps


def test_first_target_wait_allows_cold_inference_longer_than_stale_window():
    """Cold first inference gets its bounded startup window, not the runtime stale window."""
    assert rollout.FIRST_TARGET_TIMEOUT_SECONDS == 20.0
    assert rollout.FIRST_TARGET_TIMEOUT_SECONDS > rollout.STALE_AFTER_SECONDS

    mailbox = rollout._InferencePublicationMailbox()
    assert mailbox.wait_for_pending(rollout.FIRST_TARGET_TIMEOUT_SECONDS) is False


def test_inference_runs_once_per_fresh_camera_snapshot(tmp_path):
    """Removing frame identity deduplication would infer three times, not twice."""
    dependencies = FakeRuntimeDependencies(frame_keys=(1, 1, 2))

    report = run_dry_run(make_args(tmp_path), dependencies=dependencies)

    assert dependencies.infer_count == 2
    assert report["LATEST_TARGET_GENERATION"] == 2
    assert report["preview_count"] == 3


def test_executor_loop_never_reads_cameras_or_runs_inference_inline(tmp_path):
    """Calling camera or policy code on the executor thread would violate the 50 Hz boundary."""
    dependencies = ThreadCheckedRuntimeDependencies(frame_keys=(1, 1, 2))

    report = run_dry_run(make_args(tmp_path), dependencies=dependencies)

    assert dependencies.camera_read_count > 0
    assert dependencies.infer_count > 0
    assert report["preview_count"] == 3


def test_camera_inference_worker_deduplicates_stable_worker_sequence(tmp_path):
    """Reprocessing one worker sequence would run policy inference twice for one pair."""
    dependencies = FakeRuntimeDependencies(frame_keys=(7, 7, 8))
    feedback_mailbox = rollout._LatestFeedbackMailbox()
    publication_mailbox = rollout._InferencePublicationMailbox()
    feedback_mailbox.put(np.zeros(7), dependencies.clock.now)
    worker = rollout.CameraInferenceWorker(
        make_args(tmp_path),
        dependencies,
        feedback_mailbox,
        publication_mailbox,
    )

    assert worker.process_once() == "PUBLISHED"
    first = publication_mailbox.take()
    assert first.target is not None
    assert first.camera_worker_sequence == 7
    assert worker.process_once() == "DUPLICATE"
    assert publication_mailbox.take() is None
    assert worker.process_once() == "PUBLISHED"
    second = publication_mailbox.take()

    assert second.target is not None
    assert second.camera_worker_sequence == 8
    assert dependencies.infer_count == 2


@pytest.mark.parametrize(
    "camera_pair",
    [
        (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            100.0,
        ),
        (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            100.0,
            100.001,
        ),
    ],
)
def test_camera_pair_requires_explicit_stable_worker_sequence(camera_pair):
    """Falling back to a receipt timestamp would make dedupe depend on read timing."""
    with pytest.raises(ValueError, match="worker/source sequence"):
        rollout._unpack_camera_pair(camera_pair)


def test_camera_inference_worker_rejects_post_inference_stale_data(tmp_path):
    """Publishing a target after slow inference would bypass the post-inference age gate."""
    dependencies = FakeRuntimeDependencies(inference_delay=0.6)
    feedback_mailbox = rollout._LatestFeedbackMailbox()
    publication_mailbox = rollout._InferencePublicationMailbox()
    feedback_mailbox.put(np.zeros(7), dependencies.clock.now)
    worker = rollout.CameraInferenceWorker(
        make_args(tmp_path),
        dependencies,
        feedback_mailbox,
        publication_mailbox,
    )

    assert worker.process_once() == "REJECTED"
    publication = publication_mailbox.take()

    assert publication.target is None
    assert publication.raw_gate == "SKIPPED"
    assert publication.safety_reason == "STALE_DATA_HOLD"
    assert publication.inference_latency_seconds == pytest.approx(0.6)


def test_stale_after_inference_dispatches_stop_without_normal_send(
    tmp_path, preflight_environment
):
    """Removing the publication-time age check would send after a 0.6 s inference."""
    dependencies = FakeRuntimeDependencies(inference_delay=0.6)
    adapter = RecordingWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=1.0,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(args, dependencies, write_adapter=adapter)

    assert adapter.arm_commands == []
    assert adapter.gripper_commands == []
    assert adapter.stop_reasons == ["STALE_DATA_HOLD"]
    assert report["COMMAND_SEND_COUNT"] == 0
    assert report["STOP_DISPATCH_COUNT"] == 1
    assert report["STOP_REASON"] == "STALE_DATA_HOLD"


def test_post_inference_feedback_drain_exposes_fault_before_command(
    tmp_path, preflight_environment
):
    """Using pre-inference CAN state would hide a fault that arrives while inference runs."""
    dependencies = FakeRuntimeDependencies(fault_after_feedback_reads=4)
    adapter = RecordingWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=0.05,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(
        args,
        dependencies,
        write_adapter=adapter,
        preflight_report={"PREFLIGHT": "PASS"},
    )

    assert dependencies.infer_count == 1
    assert dependencies.feedback_read_count >= 4
    assert adapter.arm_commands == []
    assert adapter.stop_reasons == ["DRIVER_FAULT_HOLD"]
    assert report["COMMAND_SEND_COUNT"] == 0


def test_final_send_is_immediately_preceded_by_feedback_drain_and_driver_poll(
    tmp_path, preflight_environment
):
    """A cached final gate would leave no feedback drain between inference and send."""
    events = []
    dependencies = FakeRuntimeDependencies(event_log=events)
    adapter = RecordingWriteAdapter(event_log=events)
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=0.05,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    run_real(
        args,
        dependencies,
        write_adapter=adapter,
        preflight_report={"PREFLIGHT": "PASS"},
    )

    send_index = events.index("send_arm_absolute")
    assert events[send_index - 2 : send_index + 1] == [
        "read_feedback",
        "driver_status",
        "send_arm_absolute",
    ]
    last_infer = max(index for index, event in enumerate(events[:send_index]) if event == "infer")
    assert events[last_infer:send_index].count("read_feedback") >= 2


def test_partial_enable_failure_forces_stop_and_counts_successful_can_frames(
    tmp_path, preflight_environment
):
    """A failure after the first enable frame must still stop and report both successful writes."""
    dependencies = FakeRuntimeDependencies()
    adapter = PartialEnableWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=0.05,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(
        args,
        dependencies,
        write_adapter=adapter,
        preflight_report={"PREFLIGHT": "PASS"},
    )

    assert adapter.stop_reasons == ["WRITE_FAILURE:ENABLE"]
    assert report["ARM_ENABLE_COUNT"] == 0
    assert report["COMMAND_SEND_COUNT"] == 0
    assert report["STOP_DISPATCH_COUNT"] == 1
    assert report["CAN_WRITE_COUNT"] == 2
    assert report["REAL_CAN_WRITE"] == "YES"
    assert report["REAL_ROBOT_EXECUTED"] == "YES"
    assert report["STOP_REASON"] == "WRITE_FAILURE:ENABLE"
    assert report["RUN_STATUS"] == "FAIL"


def test_partial_arm_write_failure_forces_stop_and_counts_successful_can_frames(
    tmp_path, preflight_environment
):
    """A partially written arm command must not be reported as a completed command."""
    dependencies = FakeRuntimeDependencies()
    adapter = PartialArmWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=0.05,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(
        args,
        dependencies,
        write_adapter=adapter,
        preflight_report={"PREFLIGHT": "PASS"},
    )

    assert adapter.stop_reasons == ["WRITE_FAILURE:ARM_COMMAND"]
    assert report["ARM_ENABLE_COUNT"] == 1
    assert report["COMMAND_SEND_COUNT"] == 0
    assert report["STOP_DISPATCH_COUNT"] == 1
    assert report["CAN_WRITE_COUNT"] == 4
    assert report["REAL_CAN_WRITE"] == "YES"
    assert report["REAL_ROBOT_EXECUTED"] == "YES"
    assert report["STOP_REASON"] == "WRITE_FAILURE:ARM_COMMAND"
    assert report["RUN_STATUS"] == "FAIL"


def test_post_enable_exception_reports_successful_frames_and_stop_attempt(
    tmp_path, preflight_environment
):
    """An exception after enable must not erase physical-write evidence or its reason."""
    dependencies = PostEnableFailureDependencies()
    adapter = RecordingWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        max_time=0.05,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(args, dependencies, write_adapter=adapter)

    assert adapter.stop_reasons == ["RUNTIME_EXCEPTION:post-enable feedback failed"]
    assert report["STOP_REASON"] == "RUNTIME_EXCEPTION:post-enable feedback failed"
    assert report["STOP_ATTEMPT_COUNT"] == 1
    assert report["STOP_DISPATCH_COUNT"] == 1
    assert report["CAN_WRITE_COUNT"] == 3
    assert report["REAL_CAN_WRITE"] == "YES"
    assert report["REAL_ROBOT_EXECUTED"] == "YES"
    assert report["RUN_STATUS"] == "FAIL"


def test_stale_dry_run_tick_previews_executor_deceleration(tmp_path):
    """Skipping executor.step on stale data would omit the required HOLD/deceleration preview."""
    dependencies = FakeRuntimeDependencies(frame_keys=(1,), action=np.array([4.0, 0, 0, 0, 0, 0, 0]))
    args = make_args(tmp_path, max_time=0.55)

    run_dry_run(args, dependencies=dependencies)
    rows = [json.loads(line) for line in (tmp_path / "joint2999_dry_run.jsonl").read_text().splitlines()]
    stale_rows = [row for row in rows if row.get("safety_reason") == "STALE_DATA_HOLD"]

    assert stale_rows
    assert all(row["send_decision"] == "HOLD_PREVIEW" for row in stale_rows)
    assert all("executor_command" in row for row in stale_rows)
    assert all("joint_velocities_deg_s" in row for row in stale_rows)


def test_every_runtime_log_row_records_raw_gate_result(tmp_path):
    """Rows emitted while reusing an accepted target must not omit its raw-gate result."""
    dependencies = FakeRuntimeDependencies(frame_keys=(1, 1, 1))

    run_dry_run(make_args(tmp_path), dependencies=dependencies)
    rows = [json.loads(line) for line in (tmp_path / "joint2999_dry_run.jsonl").read_text().splitlines()]

    assert rows
    assert all(row.get("raw_gate") in {"PASS", "REJECT", "SKIPPED", "NOT_RUN"} for row in rows)


@pytest.mark.parametrize(
    ("driver_statuses", "action", "reason"),
    [
        ([faulted_driver_status(100.0)], np.zeros(7), "DRIVER_FAULT_HOLD"),
        ([], np.array([6.0, 0, 0, 0, 0, 0, 0]), "SAFETY_ABORT:RAW_CONTINUITY"),
    ],
)
def test_real_fault_or_reject_stops_without_normal_send(
    tmp_path, preflight_environment, driver_statuses, action, reason
):
    """Removing fault/reject gating would emit a normal absolute command."""
    dependencies = FakeRuntimeDependencies(driver_statuses=driver_statuses, action=action)
    adapter = RecordingWriteAdapter()
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    report = run_real(args, dependencies, write_adapter=adapter)

    assert adapter.arm_commands == []
    assert adapter.gripper_commands == []
    assert adapter.stop_reasons == [reason]
    assert report["COMMAND_SEND_COUNT"] == 0
    assert report["STOP_DISPATCH_COUNT"] == 1


def test_preflight_invokes_camera_feedback_and_driver_health_probes(
    tmp_path, preflight_environment
):
    """Returning after manifest validation would leave both probe counters at zero."""
    dependencies = FakeRuntimeDependencies()

    report = run_preflight(make_args(tmp_path), dependencies=dependencies)

    assert dependencies.camera_probe_count == 1
    assert dependencies.feedback_probe_count == 1
    assert report["CAMERA_PROBE"] == "PASS"
    assert report["CAMERA_TOP"]["resolved_identity"].startswith(str(Path("/dev") / "v4l" / "by-id") + "/")
    assert report["CAMERA_TOP"]["actual_shape"] == {"width": 640, "height": 480, "channels": 3}
    assert report["CAMERA_TOP"]["freshness_contract"] == "bounded_host_receipt_sequence_liveness"
    assert report["CAMERA_TOP"]["timestamp_source"] == "host_receipt_monotonic_not_sensor_capture_time"
    assert report["CAMERA_WRIST"]["resolved_identity"] == "<REDACTED>"
    assert report["CAMERA_WRIST"]["actual_shape"] == {"width": 640, "height": 480, "channels": 3}
    assert report["FEEDBACK_PROBE"] == "PASS"
    assert report["PIPER_CONNECTION"] == "READ_ONLY_PASS"
    assert report["DRIVER_FAULT"] == "NO"
    assert report["DRIVER_STATUS"]["healthy"] is True
    assert report["ARM_ENABLE_COUNT"] == 0
    assert report["COMMAND_SEND_COUNT"] == 0


def test_preflight_fails_closed_without_top_host_sequence_liveness(
    tmp_path, preflight_environment
):
    """Trusting a bare camera PASS would admit an unusable OpenCV freshness probe."""
    dependencies = FakeRuntimeDependencies()
    valid_probe = dependencies.probe_cameras

    def probe_without_liveness(manifest):
        report = valid_probe(manifest)
        report["top"].update(
            {
                "freshness_verifiable": False,
                "freshness_contract": "unavailable",
                "timestamp_source": "unavailable_from_opencv_live_capture",
            }
        )
        return report

    dependencies.probe_cameras = probe_without_liveness

    with pytest.raises(rollout.PreflightError, match="CAMERA_PROBE") as error:
        run_preflight(make_args(tmp_path), dependencies=dependencies)

    assert error.value.report["CAMERA_PROBE"] == "FAIL"
    assert error.value.report["ARM_ENABLE_COUNT"] == 0
    assert error.value.report["COMMAND_SEND_COUNT"] == 0


@pytest.mark.parametrize(("camera_name", "identity"), [("top", "/dev/video9"), ("wrist", "wrong")])
def test_preflight_fails_closed_on_camera_identity_mismatch(
    tmp_path, preflight_environment, camera_name, identity
):
    """A live frame from an undeclared device must not satisfy identity preflight."""
    dependencies = FakeRuntimeDependencies()
    valid_probe = dependencies.probe_cameras

    def probe_wrong_identity(manifest):
        report = valid_probe(manifest)
        report[camera_name]["resolved_identity"] = identity
        return report

    dependencies.probe_cameras = probe_wrong_identity

    with pytest.raises(rollout.PreflightError, match="CAMERA_PROBE"):
        run_preflight(make_args(tmp_path), dependencies=dependencies)


def test_real_mode_requires_distinct_acknowledgement():
    """Accepting real mode without the exact acknowledgement is unsafe."""
    with pytest.raises(SystemExit, match="I_UNDERSTAND_JOINT2999_REAL_RUN"):
        parse_args(["--real"])


def test_real_mode_requires_explicit_write_adapter(tmp_path, preflight_environment):
    """Falling back to the preview sink in real mode would bypass explicit write selection."""
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    with pytest.raises(RuntimeError, match="explicit absolute Joint write adapter"):
        run_real(args, FakeRuntimeDependencies(), write_adapter=None)


def test_real_mode_requires_per_frame_can_write_accounting(tmp_path, preflight_environment):
    """Without an adapter frame counter, a partial write could be reported as zero."""
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )
    adapter = RecordingWriteAdapter()
    adapter.successful_can_write_count = None

    with pytest.raises(RuntimeError, match="CAN-write accounting"):
        run_real(args, FakeRuntimeDependencies(), write_adapter=adapter)

    assert adapter.enable_count == 0
    assert adapter.stop_reasons == []


def test_real_mode_ignores_passing_report_and_builds_writer_after_fresh_preflight(
    tmp_path, preflight_environment
):
    """A caller-supplied PASS must not bypass probes or construct the writer early."""
    dependencies = FakeRuntimeDependencies()
    adapter = RecordingWriteAdapter()
    factory_probe_counts = []
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    def build_writer():
        factory_probe_counts.append(
            (dependencies.camera_probe_count, dependencies.feedback_probe_count)
        )
        return adapter

    report = run_real(
        args,
        dependencies,
        write_adapter_factory=build_writer,
        preflight_report={"PREFLIGHT": "PASS"},
    )

    assert factory_probe_counts == [(1, 1)]
    assert report["PREFLIGHT"] == "PASS"


def test_real_mode_rejects_supplied_pass_when_fresh_preflight_fails(
    tmp_path, preflight_environment
):
    """Only the fresh report from the actual runtime dependencies may authorize a writer."""
    dependencies = FakeRuntimeDependencies()
    dependencies.probe_feedback = lambda: (_ for _ in ()).throw(RuntimeError("fresh probe failed"))
    writer_built = False
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
    )

    def build_writer():
        nonlocal writer_built
        writer_built = True
        return RecordingWriteAdapter()

    with pytest.raises(rollout.PreflightError, match="fresh probe failed"):
        run_real(
            args,
            dependencies,
            write_adapter_factory=build_writer,
            preflight_report={"PREFLIGHT": "PASS"},
        )

    assert dependencies.camera_probe_count == 1
    assert writer_built is False


def test_wrong_runtime_ack_blocks_preflight_and_adapter(tmp_path, preflight_environment):
    """Moving acknowledgement after preflight would touch hardware on an unacknowledged run."""
    dependencies = FakeRuntimeDependencies()
    adapter = RecordingWriteAdapter()
    args = make_args(tmp_path, dry_run=False, real=True, operator_ack="wrong")

    with pytest.raises(RuntimeError, match=REAL_ACKNOWLEDGEMENT):
        run_real(args, dependencies, write_adapter=adapter)

    assert dependencies.camera_probe_count == 0
    assert dependencies.feedback_probe_count == 0
    assert adapter.enable_count == 0
    assert adapter.stop_reasons == []


def test_main_builds_deployable_default_readonly_dependencies(monkeypatch, tmp_path):
    """Reintroducing the unavailable facade would bypass the concrete default factory."""
    args = make_args(tmp_path)
    created = []
    dependencies = FakeRuntimeDependencies()
    policy = object()

    monkeypatch.setattr(rollout, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(rollout, "run_preflight", lambda _args, dependencies: {"PREFLIGHT": "PASS"})
    monkeypatch.setattr(rollout, "load_joint_policy", lambda *_args: (policy, {"CHECKPOINT_LOAD": "PASS"}))

    def build_default(*, policy, manifest_path, can_interface):
        created.append((policy, manifest_path, can_interface))
        return dependencies

    monkeypatch.setattr(rollout, "DefaultRuntimeDependencies", build_default)
    monkeypatch.setattr(
        rollout,
        "run_dry_run",
        lambda _args, dependencies: {
            "REAL_CAN_WRITE": "NO",
            "REAL_ROBOT_EXECUTED": "NO",
            "dependencies": dependencies is not None,
        },
    )

    assert rollout.main([]) == 0
    assert created == [(policy, args.manifest, "can0")]


def test_main_preserves_real_runtime_write_counters(monkeypatch, tmp_path, capsys):
    """Merging preflight last would overwrite real runtime writes with zero counters."""
    args = make_args(
        tmp_path,
        dry_run=False,
        real=True,
        operator_ack=REAL_ACKNOWLEDGEMENT,
        write_adapter=rollout.SOCKETCAN_WRITE_ADAPTER,
    )
    dependencies = FakeRuntimeDependencies()
    adapter = RecordingWriteAdapter()

    monkeypatch.setattr(rollout, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(rollout, "load_joint_policy", lambda *_args: (object(), {"CHECKPOINT_LOAD": "PASS"}))
    monkeypatch.setattr(rollout, "DefaultRuntimeDependencies", lambda **_kwargs: dependencies)
    monkeypatch.setattr(
        rollout,
        "run_preflight",
        lambda _args, dependencies: {
            "PREFLIGHT": "PASS",
            "REAL_CAN_WRITE": "NO",
            "ARM_ENABLE_COUNT": 0,
            "COMMAND_SEND_COUNT": 0,
        },
    )
    monkeypatch.setattr(rollout, "_build_explicit_write_adapter", lambda _args: adapter)
    monkeypatch.setattr(
        rollout,
        "run_real",
        lambda *_args, **_kwargs: {
            "REAL_CAN_WRITE": "YES",
            "REAL_ROBOT_EXECUTED": "YES",
            "ARM_ENABLE_COUNT": 1,
            "COMMAND_SEND_COUNT": 2,
        },
    )

    assert rollout.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["REAL_CAN_WRITE"] == "YES"
    assert report["ARM_ENABLE_COUNT"] == 1
    assert report["COMMAND_SEND_COUNT"] == 2


def test_main_does_not_close_dependencies_while_camera_worker_is_blocked(
    monkeypatch, tmp_path, capsys, preflight_environment
):
    """Closing camera/CAN dependencies under a live worker creates a use-after-close race."""
    args = make_args(tmp_path, max_time=0.03)
    dependencies = BlockingCameraDependencies()

    monkeypatch.setattr(rollout, "STALE_AFTER_SECONDS", 0.01)
    monkeypatch.setattr(rollout, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        rollout,
        "load_joint_policy",
        lambda *_args: (object(), {"CHECKPOINT_LOAD": "PASS"}),
    )
    monkeypatch.setattr(rollout, "DefaultRuntimeDependencies", lambda **_kwargs: dependencies)

    try:
        assert rollout.main([]) == 1
        report = json.loads(capsys.readouterr().out)
        assert dependencies.camera_blocked.is_set()
        assert report["CAMERA_INFERENCE_WORKER_STOPPED"] == "NO"
        assert report["RUN_STATUS"] == "FAIL"
        assert report["STOP_REASON"] == "CAMERA_INFERENCE_WORKER_STOP_TIMEOUT"
        assert dependencies.close_count == 0
    finally:
        dependencies.release_camera.set()
        assert dependencies.camera_returned.wait(timeout=1.0)


def test_quality58_h20_final_checkpoint_runner_contract():
    """The runner defaults must match the final Quality58 H20 deployment artifact."""
    from openpi.rollouts import direct_joint_core
    from openpi.training import config as training_config

    expected_checkpoint = Path(
        "<CHECKPOINT_ROOT>/"
        "pi05_piper_joint_rtc_h20_quality58_finetune/"
        "pi05_piper_joint_rtc_h20_quality58_3k_20260902_135854/2999"
    )

    assert rollout.CONFIG_NAME == "pi05_piper_joint_rtc_h20_quality58_finetune"
    assert rollout.CHECKPOINT == expected_checkpoint

    config = training_config.get_config(rollout.CONFIG_NAME)
    assert config.model.action_horizon == 20
    assert config.model.action_dim == 32
    assert config.data.create(config.assets_dirs, config.model).action_sequence_keys == ("action",)

    norm_path = expected_checkpoint / "assets" / "local/piper_joint_rtc_quality58" / "norm_stats.json"
    assert norm_path.is_file()  # checkpoint-local norm was validated by the prior offline gate
    assert rollout._first_action({"actions": np.zeros((20, 7), dtype=np.float32)}).shape == (7,)

    assert np.array_equal(direct_joint_core.PHYSICAL_HARD_LIMITS[2], np.array([-170.0, 0.0]))
    assert direct_joint_core.EXECUTOR_STEP_DEG == 1.0
