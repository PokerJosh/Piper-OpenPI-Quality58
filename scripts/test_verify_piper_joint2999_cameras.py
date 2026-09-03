from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.verify_piper_joint2999_cameras as camera_probe
from scripts.verify_piper_joint2999_cameras import FRAME_STALE_AFTER_SECONDS
from scripts.verify_piper_joint2999_cameras import FreshnessUnverifiableError
from scripts.verify_piper_joint2999_cameras import _bounded_host_receipt_sequence
from scripts.verify_piper_joint2999_cameras import _frame_report
from scripts.verify_piper_joint2999_cameras import _probe_top
from scripts.verify_piper_joint2999_cameras import _probe_wrist
from scripts.verify_piper_joint2999_cameras import load_camera_manifest
from scripts.verify_piper_joint2999_cameras import probe_cameras
from scripts.verify_piper_joint2999_cameras import verify_manifest_shape


VALID_TOP = {
    "kind": "opencv",
    "path": "/dev/v4l/by-id/usb-ZJ-240719-ZW_Hy_RGB_Came_01.00.00-video-index0",
    "fourcc": "MJPG",
    "width": 640,
    "height": 480,
    "fps": 30,
}
VALID_WRIST = {
    "kind": "intelrealsense",
    "serial": "<WRIST_CAMERA_SERIAL>",
    "width": 640,
    "height": 480,
    "fps": 30,
}


def test_manifest_declares_stable_top_and_current_wrist_identity():
    manifest = load_camera_manifest(Path("configs/piper_joint2999_camera.yaml"))
    verify_manifest_shape(manifest)
    assert manifest["top"]["path"].endswith("video-index0")
    assert manifest["top"]["fourcc"] == "MJPG"
    assert manifest["wrist"]["serial"] == "<WRIST_CAMERA_SERIAL>"


def test_manifest_rejects_stale_wrist_serial():
    with pytest.raises(ValueError, match="<WRIST_CAMERA_SERIAL>"):
        verify_manifest_shape({"top": VALID_TOP, "wrist": {**VALID_WRIST, "serial": "<TOP_CAMERA_SERIAL>"}})


def test_frame_report_rejects_old_monotonic_source_timestamp():
    old_source_timestamp = time.monotonic() - FRAME_STALE_AFTER_SECONDS - 0.1

    with pytest.raises(RuntimeError, match="camera frame is stale"):
        _frame_report(
            np.zeros((480, 640, 3), dtype=np.uint8),
            source_timestamp_monotonic=old_source_timestamp,
            source_timestamp_kind="injected-test-source",
        )


@pytest.mark.parametrize("source_timestamp", [float("nan"), float("inf"), float("-inf")])
def test_frame_report_marks_nonfinite_source_timestamp_unverifiable(source_timestamp):
    report = _frame_report(
        np.zeros((480, 640, 3), dtype=np.uint8),
        source_timestamp_monotonic=source_timestamp,
        source_timestamp_kind="injected-test-source",
    )

    assert report["freshness_verifiable"] is False
    assert report["freshness_age_seconds"] is None


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False
        self.settings = []

    def isOpened(self):
        return True

    def set(self, property_id, value):
        self.settings.append((property_id, value))

    def read(self):
        return True, self.frame

    def release(self):
        self.released = True


def _install_fake_cv2(monkeypatch, capture):
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FOURCC=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        CAP_PROP_FPS=4,
        VideoCapture=lambda path: capture,
        VideoWriter_fourcc=lambda *fourcc: 99,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(camera_probe.Path, "exists", lambda path: True)


def test_probe_top_establishes_bounded_host_receipt_sequence_liveness(monkeypatch):
    capture = FakeCapture(np.zeros((480, 640, 3), dtype=np.uint8))
    _install_fake_cv2(monkeypatch, capture)

    report = _probe_top(VALID_TOP, read_frames=True)

    assert report["resolved_identity"] == VALID_TOP["path"]
    assert report["actual_shape"] == {"width": 640, "height": 480, "channels": 3}
    assert report["freshness_verifiable"] is True
    assert report["freshness_contract"] == "bounded_host_receipt_sequence_liveness"
    assert report["timestamp_source"] == "host_receipt_monotonic_not_sensor_capture_time"
    assert report["source_capture_timestamp_available"] is False
    assert report["bounded_buffer_capacity"] == 1
    assert report["worker_sequence"] >= 2
    assert report["freshness_age_seconds"] <= FRAME_STALE_AFTER_SECONDS
    assert "frame_timestamp_monotonic_seconds" not in report
    assert capture.released


def test_top_host_receipt_liveness_fails_closed_when_sequence_stalls():
    release_second_read = threading.Event()

    class CaptureThatStallsAfterOneFrame(FakeCapture):
        def __init__(self):
            super().__init__(np.zeros((480, 640, 3), dtype=np.uint8))
            self.read_count = 0

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return True, self.frame
            release_second_read.wait(0.1)
            return True, self.frame

    capture = CaptureThatStallsAfterOneFrame()
    try:
        with pytest.raises(FreshnessUnverifiableError) as error:
            _bounded_host_receipt_sequence(capture, timeout_seconds=0.005)
    finally:
        release_second_read.set()

    assert error.value.report["freshness_verifiable"] is False
    assert error.value.report["timestamp_source"] == "host_receipt_monotonic_not_sensor_capture_time"
    assert error.value.report["source_capture_timestamp_available"] is False


def test_probe_top_releases_capture_when_frame_shape_is_wrong(monkeypatch):
    capture = FakeCapture(np.zeros((240, 320, 3), dtype=np.uint8))
    _install_fake_cv2(monkeypatch, capture)

    with pytest.raises(RuntimeError, match="top frame shape 320x240"):
        _probe_top(VALID_TOP, read_frames=True)

    assert capture.released


class FakeColorFrame:
    def __init__(self, frame, timestamp_ms, timestamp_domain):
        self.frame = frame
        self.timestamp_ms = timestamp_ms
        self.timestamp_domain = timestamp_domain

    def get_data(self):
        return self.frame

    def get_timestamp(self):
        return self.timestamp_ms

    def get_frame_timestamp_domain(self):
        return self.timestamp_domain


def _install_fake_realsense(monkeypatch, color_frame):
    system_time = object()
    global_time = object()

    class FakePipeline:
        def __init__(self):
            self.started = False
            self.stopped = False

        def start(self, config):
            self.started = True

        def wait_for_frames(self, timeout_ms):
            return SimpleNamespace(get_color_frame=lambda: color_frame)

        def stop(self):
            self.stopped = True

    pipeline = FakePipeline()
    fake_rs = SimpleNamespace(
        camera_info=SimpleNamespace(serial_number=object()),
        stream=SimpleNamespace(color=object()),
        format=SimpleNamespace(bgr8=object()),
        timestamp_domain=SimpleNamespace(system_time=system_time, global_time=global_time),
        context=lambda: SimpleNamespace(
            query_devices=lambda: [SimpleNamespace(get_info=lambda info: VALID_WRIST["serial"])]
        ),
        pipeline=lambda context: pipeline,
        config=lambda: SimpleNamespace(enable_device=lambda serial: None, enable_stream=lambda *args: None),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    return pipeline, system_time, global_time


def test_probe_wrist_rejects_stale_system_timestamp_and_stops_pipeline(monkeypatch):
    timestamp_domain = object()
    color_frame = FakeColorFrame(
        np.zeros((480, 640, 3), dtype=np.uint8),
        (time.time() - FRAME_STALE_AFTER_SECONDS - 0.1) * 1000,
        timestamp_domain,
    )
    pipeline, system_time, _ = _install_fake_realsense(monkeypatch, color_frame)
    color_frame.timestamp_domain = system_time

    with pytest.raises(RuntimeError, match="camera frame is stale"):
        _probe_wrist(VALID_WRIST, read_frames=True)

    assert pipeline.stopped


def test_probe_wrist_stops_pipeline_when_frame_shape_is_wrong(monkeypatch):
    color_frame = FakeColorFrame(np.zeros((240, 320, 3), dtype=np.uint8), time.time() * 1000, None)
    pipeline, system_time, _ = _install_fake_realsense(monkeypatch, color_frame)
    color_frame.timestamp_domain = system_time

    with pytest.raises(RuntimeError, match="wrist frame shape 320x240"):
        _probe_wrist(VALID_WRIST, read_frames=True)

    assert pipeline.stopped


def test_probe_wrist_accepts_global_time_with_bounded_host_receipt(monkeypatch):
    color_frame = FakeColorFrame(
        np.zeros((480, 640, 3), dtype=np.uint8),
        time.time() * 1000,
        object(),
    )
    pipeline, _, global_time = _install_fake_realsense(monkeypatch, color_frame)
    color_frame.timestamp_domain = global_time

    report = _probe_wrist(VALID_WRIST, read_frames=True)

    assert report["freshness_verifiable"] is True
    assert report["freshness_contract"] == "bounded_host_receipt_sequence_liveness"
    assert report["timestamp_source"] == "host_receipt_monotonic_not_sensor_capture_time"
    assert report["source_capture_timestamp_available"] is False
    assert pipeline.stopped


def test_probe_cameras_combines_backend_reports(monkeypatch):
    top_report = {"resolved_identity": "top", "freshness_verifiable": True}
    wrist_report = {"resolved_identity": "wrist", "freshness_verifiable": True}
    monkeypatch.setattr(camera_probe, "_probe_top", lambda camera, read_frames: top_report)
    monkeypatch.setattr(camera_probe, "_probe_wrist", lambda camera, read_frames: wrist_report)

    assert probe_cameras({"top": VALID_TOP, "wrist": VALID_WRIST}) == {"top": top_report, "wrist": wrist_report}
