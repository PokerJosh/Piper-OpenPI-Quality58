"""Pure safety and execution primitives for absolute seven-value targets."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

ARM_DIM = 6
TARGET_DIM = 7
RAW_CONTINUITY_DEG = 5.0
EXECUTOR_STEP_DEG = 1.0
PHASE_VMAX_DEG_S = np.array([10.0, 12.0, 12.0, 3.0, 3.0, 3.0], dtype=np.float64)
PHYSICAL_HARD_LIMITS = np.array(
    [(-150.0, 124.22), (0.0, 179.91), (-170.0, 0.0),
     (-99.98, 99.98), (-69.90, 69.90), (-120.0, 120.0)],
    dtype=np.float64,
)
DEPLOYMENT_SOFT_LIMITS = np.array(
    [(-149.0, 123.22), (1.0, 178.91), (-169.0, -0.30),
     (-98.98, 98.98), (-68.90, 68.90), (-119.0, 119.0)],
    dtype=np.float64,
)
GRIPPER_LIMITS_M = (0.0, 0.07)
GRIPPER_CHANGE_M = 0.0005
GRIPPER_KEEPALIVE_S = 5.0


@dataclass(frozen=True)
class Target:
    """One absolute target in [J1, J2, J3, J4, J5, J6, gripper] order."""

    values: np.ndarray
    timestamp: float
    generation_id: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (TARGET_DIM,):
            raise ValueError("target must have shape (7,)")
        snapshot = values.copy()
        snapshot.setflags(write=False)
        object.__setattr__(self, "values", snapshot)


class LatestOnlyTargetMailbox:
    """Thread-safe single-slot mailbox with overwrite accounting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Target | None = None
        self._dropped_count = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def put(self, target: Target) -> None:
        if not isinstance(target, Target):
            raise TypeError("target must be a Target")
        with self._lock:
            if self._pending is not None:
                self._dropped_count += 1
            self._pending = target

    def take(self) -> Target | None:
        with self._lock:
            target = self._pending
            self._pending = None
            return target


def _vector(value: object, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,):
        return None
    if not np.isfinite(array).all():
        return None
    return array


def _outside_direction(target: np.ndarray, reference: np.ndarray) -> str | None:
    for index, (lower, upper) in enumerate(DEPLOYMENT_SOFT_LIMITS):
        if reference[index] < lower and target[index] < reference[index] - 1e-9:
            return f"J{index + 1}_FURTHER_OUTSIDE_LOW"
        if reference[index] > upper and target[index] > reference[index] + 1e-9:
            return f"J{index + 1}_FURTHER_OUTSIDE_HIGH"
    return None


def validate_raw_target(raw_target_7, reference_7) -> tuple[bool, str | None]:
    """Validate an absolute policy target against the accepted-target reference."""

    target = _vector(raw_target_7, TARGET_DIM)
    reference = _vector(reference_7, TARGET_DIM)
    if target is None:
        try:
            shape = np.asarray(raw_target_7).shape
        except (TypeError, ValueError):
            shape = None
        return (False, "TARGET_SHAPE" if shape != (TARGET_DIM,) else "NONFINITE_TARGET")
    if reference is None:
        try:
            shape = np.asarray(reference_7).shape
        except (TypeError, ValueError):
            shape = None
        return (False, "REFERENCE_SHAPE" if shape != (TARGET_DIM,) else "NONFINITE_REFERENCE")
    if np.any(target[:ARM_DIM] < PHYSICAL_HARD_LIMITS[:, 0]) or np.any(target[:ARM_DIM] > PHYSICAL_HARD_LIMITS[:, 1]):
        return False, "HARD_LIMIT"
    if not GRIPPER_LIMITS_M[0] <= target[6] <= GRIPPER_LIMITS_M[1]:
        return False, "GRIPPER_LIMIT"
    direction_reason = _outside_direction(target, reference)
    if direction_reason is not None:
        return False, direction_reason
    if np.max(np.abs(target[:ARM_DIM] - reference[:ARM_DIM])) > RAW_CONTINUITY_DEG:
        return False, "RAW_CONTINUITY"
    return True, None


def validate_executor_command(q_next_6, previous_q_6, dt, driver_speed_cap_deg_s) -> tuple[bool, str | None]:
    """Validate one absolute six-joint command before it is handed off."""

    q_next = _vector(q_next_6, ARM_DIM)
    previous = _vector(previous_q_6, ARM_DIM)
    if q_next is None:
        try:
            shape = np.asarray(q_next_6).shape
        except (TypeError, ValueError):
            shape = None
        return (False, "COMMAND_SHAPE" if shape != (ARM_DIM,) else "NONFINITE_COMMAND")
    if previous is None:
        try:
            shape = np.asarray(previous_q_6).shape
        except (TypeError, ValueError):
            shape = None
        return (False, "PREVIOUS_SHAPE" if shape != (ARM_DIM,) else "NONFINITE_PREVIOUS")
    try:
        dt = float(dt)
        speed_cap = float(driver_speed_cap_deg_s)
    except (TypeError, ValueError):
        return False, "BAD_TIMING"
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(speed_cap) or speed_cap <= 0.0:
        return False, "BAD_TIMING"
    if np.any(q_next < PHYSICAL_HARD_LIMITS[:, 0]) or np.any(q_next > PHYSICAL_HARD_LIMITS[:, 1]):
        return False, "HARD_LIMIT"
    direction_reason = _outside_direction(q_next, previous)
    if direction_reason is not None:
        return False, direction_reason
    delta = np.abs(q_next - previous)
    if np.max(delta) > EXECUTOR_STEP_DEG + 1e-9:
        return False, "EXECUTOR_STEP"
    if np.max(delta / dt) > speed_cap + 1e-9:
        return False, "DRIVER_SPEED_CAP"
    return True, None


class PhaseAwareExecutor:
    """Acceleration-limited, braking-aware six-joint 50 Hz command generator."""

    def __init__(self, control_hz: float = 50.0, ramp_sec: float = 0.50) -> None:
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if not np.isfinite(ramp_sec) or ramp_sec <= 0.0:
            raise ValueError("ramp_sec must be positive")
        self.dt = 1.0 / float(control_hz)
        self.ramp_sec = float(ramp_sec)
        self.v_cmd = np.zeros(ARM_DIM, dtype=np.float64)

    def step(self, commanded_6, target_6, phase: str = "NORMAL", wrist_scale: float = 1.0) -> np.ndarray:
        if phase != "NORMAL":
            raise ValueError("unsupported phase; only NORMAL is implemented")
        commanded = _vector(commanded_6, ARM_DIM)
        target = _vector(target_6, ARM_DIM)
        if commanded is None or target is None:
            raise ValueError("commanded_6 and target_6 must be finite shape (6,)")
        if not np.isfinite(wrist_scale) or wrist_scale < 0.0:
            raise ValueError("wrist_scale must be nonnegative")
        vmax = PHASE_VMAX_DEG_S.copy()
        vmax[3:] *= float(wrist_scale)
        amax = vmax / self.ramp_sec
        error = target - commanded
        absolute_error = np.abs(error)
        acceleration_tick = amax * self.dt
        stopping_velocity = np.sqrt(acceleration_tick**2 + 2.0 * amax * absolute_error) - acceleration_tick
        desired_velocity = np.sign(error) * np.minimum(vmax, stopping_velocity)
        self.v_cmd += np.clip(desired_velocity - self.v_cmd, -acceleration_tick, acceleration_tick)
        return commanded + np.clip(self.v_cmd * self.dt, -EXECUTOR_STEP_DEG, EXECUTOR_STEP_DEG)


def gripper_send_decision(target_m, last_sent_m, last_sent_ts, now) -> tuple[bool, bool]:
    """Return whether a gripper target is material or its keepalive is due."""

    target = float(target_m)
    if not np.isfinite(target):
        raise ValueError("target_m must be finite")
    changed = last_sent_m is None or abs(target - float(last_sent_m)) >= GRIPPER_CHANGE_M
    keepalive = last_sent_ts is not None and float(now) - float(last_sent_ts) >= GRIPPER_KEEPALIVE_S
    return bool(changed or keepalive), bool(keepalive)
