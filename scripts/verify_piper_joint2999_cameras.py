"""Read-only camera identity and frame verification for the Joint2999 runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np
import yaml


TOP_PATH = "<TOP_CAMERA_DEVICE_PATH>"
WRIST_SERIAL = "<REDACTED>"
FRAME_STALE_AFTER_SECONDS = 2.0
HOST_RECEIPT_SEQUENCE_CONTRACT = "bounded_host_receipt_sequence_liveness"
HOST_RECEIPT_TIMESTAMP_KIND = "host_receipt_monotonic_not_sensor_capture_time"
HOST_RECEIPT_BUFFER_CAPACITY = 1


class FreshnessUnverifiableError(RuntimeError):
    """Raised when a backend cannot prove a frame's capture freshness."""

    def __init__(self, camera_name: str, report: dict[str, Any]):
        super().__init__(f"{camera_name} frame freshness cannot be verified")
        self.report = report


def load_camera_manifest(path: Path | str) -> dict[str, Any]:
    """Load a YAML camera manifest without accessing devices."""
    with Path(path).open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("camera manifest must be a mapping")
    return manifest


def verify_manifest_shape(manifest: dict[str, Any]) -> None:
    """Reject manifests that do not name the exact Joint2999 cameras."""
    top = manifest.get("top")
    wrist = manifest.get("wrist")
    if not isinstance(top, dict) or not isinstance(wrist, dict):
        raise ValueError("camera manifest requires top and wrist mappings")

    expected_top = {
        "kind": "opencv",
        "path": TOP_PATH,
        "fourcc": "MJPG",
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    expected_wrist = {
        "kind": "intelrealsense",
        "serial": WRIST_SERIAL,
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    for name, camera, expected in (("top", top, expected_top), ("wrist", wrist, expected_wrist)):
        for field, value in expected.items():
            if camera.get(field) != value:
                raise ValueError(f"{name}.{field} must be {value!r}")


def _frame_report(
    frame: Any,
    *,
    source_timestamp_monotonic: float | None,
    source_timestamp_kind: str,
) -> dict[str, Any]:
    """Report a frame using a source timestamp mapped to host monotonic time."""
    if frame is None or len(frame.shape) < 2:
        raise RuntimeError("camera returned an invalid frame")
    height, width = frame.shape[:2]
    channels = int(frame.shape[2]) if len(frame.shape) > 2 else 1
    report: dict[str, Any] = {
        "actual_shape": {"width": int(width), "height": int(height), "channels": channels},
        "pixel_order": "BGR" if channels == 3 else "GRAY",
        "timestamp_source": source_timestamp_kind,
    }
    if source_timestamp_monotonic is None or not math.isfinite(source_timestamp_monotonic):
        report.update({"freshness_verifiable": False, "freshness_age_seconds": None})
        return report

    freshness_age = time.monotonic() - source_timestamp_monotonic
    if freshness_age < 0:
        report.update({"freshness_verifiable": False, "freshness_age_seconds": None})
        return report
    if freshness_age > FRAME_STALE_AFTER_SECONDS:
        raise RuntimeError(f"camera frame is stale ({freshness_age:.3f}s old)")
    report.update(
        {
            "frame_timestamp_monotonic_seconds": source_timestamp_monotonic,
            "freshness_age_seconds": freshness_age,
            "freshness_verifiable": True,
        }
    )
    return report


def _verify_dimensions(camera_name: str, report: dict[str, Any], camera: dict[str, Any]) -> None:
    shape = report["actual_shape"]
    if shape["width"] != camera["width"] or shape["height"] != camera["height"]:
        raise RuntimeError(
            f"{camera_name} frame shape {shape['width']}x{shape['height']} does not match "
            f"{camera['width']}x{camera['height']}"
        )


def _require_verifiable_freshness(camera_name: str, report: dict[str, Any]) -> None:
    if not report["freshness_verifiable"]:
        raise FreshnessUnverifiableError(camera_name, report)


def _put_latest(mailbox: queue.Queue, item: Any) -> None:
    try:
        mailbox.put_nowait(item)
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        mailbox.put_nowait(item)


def _bounded_host_receipt_sequence(capture, timeout_seconds: float) -> tuple[Any, float, int, float]:
    """Read two deliveries through a latest-only slot within a host-time bound."""
    mailbox: queue.Queue = queue.Queue(maxsize=HOST_RECEIPT_BUFFER_CAPACITY)
    stop_event = threading.Event()
    started_at = time.monotonic()

    def acquire() -> None:
        first_receipt = None
        for sequence in (1, 2):
            if stop_event.is_set():
                return
            ok, frame = capture.read()
            receipt = time.monotonic()
            if not ok or frame is None:
                _put_latest(mailbox, ("error", RuntimeError("top camera failed to return a frame")))
                return
            if first_receipt is None:
                first_receipt = receipt
            _put_latest(mailbox, ("frame", frame, receipt, sequence, first_receipt))

    worker = threading.Thread(target=acquire, name="top-camera-liveness-probe", daemon=True)
    worker.start()
    deadline = started_at + float(timeout_seconds)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                report = {
                    "freshness_verifiable": False,
                    "freshness_age_seconds": None,
                    "freshness_contract": HOST_RECEIPT_SEQUENCE_CONTRACT,
                    "timestamp_source": HOST_RECEIPT_TIMESTAMP_KIND,
                    "source_capture_timestamp_available": False,
                    "bounded_buffer_capacity": HOST_RECEIPT_BUFFER_CAPACITY,
                }
                raise FreshnessUnverifiableError("top", report)
            try:
                item = mailbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if item[0] == "error":
                raise item[1]
            _, frame, receipt, sequence, first_receipt = item
            if sequence >= 2:
                return frame, receipt, sequence, receipt - first_receipt
    finally:
        stop_event.set()


def _probe_top(camera: dict[str, Any], read_frames: bool) -> dict[str, Any]:
    import cv2

    declared_path = Path(camera["path"])
    if not declared_path.exists():
        raise FileNotFoundError(f"top camera identity is missing: {declared_path}")
    capture = cv2.VideoCapture(str(declared_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open top camera identity: {declared_path}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera["fourcc"]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera["width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["height"])
        capture.set(cv2.CAP_PROP_FPS, camera["fps"])
        result: dict[str, Any] = {"resolved_identity": str(declared_path.resolve())}
        if read_frames:
            frame, receipt, sequence, receipt_interval = _bounded_host_receipt_sequence(
                capture,
                FRAME_STALE_AFTER_SECONDS,
            )
            result.update(_frame_report(frame, source_timestamp_monotonic=None, source_timestamp_kind="unavailable"))
            freshness_age = time.monotonic() - receipt
            if freshness_age < 0.0 or freshness_age > FRAME_STALE_AFTER_SECONDS:
                raise RuntimeError(f"top camera host receipt is stale ({freshness_age:.3f}s old)")
            result.update(
                {
                    "freshness_verifiable": True,
                    "freshness_age_seconds": freshness_age,
                    "freshness_contract": HOST_RECEIPT_SEQUENCE_CONTRACT,
                    "timestamp_source": HOST_RECEIPT_TIMESTAMP_KIND,
                    "source_capture_timestamp_available": False,
                    "host_receipt_monotonic_seconds": receipt,
                    "worker_sequence": sequence,
                    "host_receipt_interval_seconds": receipt_interval,
                    "bounded_buffer_capacity": HOST_RECEIPT_BUFFER_CAPACITY,
                }
            )
            _verify_dimensions("top", result, camera)
            _require_verifiable_freshness("top", result)
        return result
    finally:
        capture.release()


def _system_timestamp_to_monotonic(source_timestamp_ms: float) -> float:
    """Map a RealSense system-clock timestamp to host monotonic time."""
    wall_now = time.time()
    monotonic_now = time.monotonic()
    return monotonic_now - (wall_now - source_timestamp_ms / 1000.0)


def _probe_wrist(camera: dict[str, Any], read_frames: bool) -> dict[str, Any]:
    import pyrealsense2 as rs

    context = rs.context()
    serials = [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]
    if camera["serial"] not in serials:
        raise RuntimeError(f"wrist camera identity is missing: {camera['serial']}")
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(camera["serial"])
    config.enable_stream(rs.stream.color, camera["width"], camera["height"], rs.format.bgr8, camera["fps"])
    started = False
    try:
        pipeline.start(config)
        started = True
        result: dict[str, Any] = {"resolved_identity": camera["serial"]}
        if read_frames:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("wrist camera failed to return a color frame")
            source_timestamp_ms = float(color_frame.get_timestamp())
            host_receipt_monotonic = time.monotonic()
            timestamp_domain = color_frame.get_frame_timestamp_domain()
            source_timestamp_monotonic = None
            source_timestamp_kind = f"unmappable_realsense_timestamp_domain:{timestamp_domain}"
            if timestamp_domain == rs.timestamp_domain.system_time:
                source_timestamp_monotonic = _system_timestamp_to_monotonic(source_timestamp_ms)
                source_timestamp_kind = "realsense_system_time_mapped_to_host_monotonic"
            elif timestamp_domain == getattr(rs.timestamp_domain, "global_time", object()):
                # Some D435 firmware reports global_time while the Python API
                # does not expose a host-clock mapping. Use the bounded host
                # receipt contract; do not present the device timestamp as a
                # host-mapped capture timestamp.
                source_timestamp_monotonic = host_receipt_monotonic
                source_timestamp_kind = HOST_RECEIPT_TIMESTAMP_KIND
            result.update(
                _frame_report(
                    np.asanyarray(color_frame.get_data()),
                    source_timestamp_monotonic=source_timestamp_monotonic,
                    source_timestamp_kind=source_timestamp_kind,
                )
            )
            result["source_capture_timestamp_available"] = (
                source_timestamp_kind == "realsense_system_time_mapped_to_host_monotonic"
            )
            result["frame_source_timestamp_milliseconds"] = source_timestamp_ms
            if source_timestamp_kind == HOST_RECEIPT_TIMESTAMP_KIND:
                result.update(
                    {
                        "freshness_contract": HOST_RECEIPT_SEQUENCE_CONTRACT,
                        "host_receipt_monotonic_seconds": host_receipt_monotonic,
                        "bounded_buffer_capacity": HOST_RECEIPT_BUFFER_CAPACITY,
                    }
                )
            _verify_dimensions("wrist", result, camera)
            _require_verifiable_freshness("wrist", result)
        return result
    finally:
        if started:
            pipeline.stop()


def probe_cameras(manifest: dict[str, Any], read_frames: bool = True) -> dict[str, Any]:
    """Probe declared camera identities and frames without robot interaction."""
    verify_manifest_shape(manifest)
    return {
        "top": _probe_top(manifest["top"], read_frames),
        "wrist": _probe_wrist(manifest["wrist"], read_frames),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_camera_manifest(args.manifest)
        verify_manifest_shape(manifest)
        print(json.dumps(probe_cameras(manifest), sort_keys=True))
    except FreshnessUnverifiableError as error:
        print(json.dumps({"error": str(error), "camera": error.report}, sort_keys=True))
        return 1
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
