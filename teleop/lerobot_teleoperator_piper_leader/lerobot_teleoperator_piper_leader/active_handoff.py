"""ACTIVE LEADER MIRROR HANDOFF for the Feetech Piper leader arm.

Semantics (task spec — torque stays ON after a converged mirror):
    POLICY → PAUSE → leader torque ON → smooth mirror to the follower pose →
    convergence gate (≤1.0 deg equivalent error, 3 consecutive frames) →
    LEADER_SYNCED = YES, torque REMAINS ON (the operator holds the leader) →
    only on formally entering CORRECTING is torque released (disable_torque()).

The follower/Piper NEVER moves for alignment (follower-slide is NOT used).

Components:
  * FollowerToLeaderMap  — PURE inverse of the kit's forward mapping
    (leader ticks → piper rad/m). Uses the exact same JointMap/GripperMap
    tables from piper_leader_kit/config/leader_to_piper_mapping.example.json, so no
    mapping is re-invented. A follower pose is mirrorable only when the
    inverted ticks land inside the leader's travel range (no silent clipping).
  * LeaderBus protocol   — the single serial owner must provide:
    read_ticks(), write_goal(ticks, speed, acc), enable_torque(),
    disable_torque(). FeetechLeaderReader (feetech_reader.py) implements it
    on the SAME port/packet handlers used for reading — never a second owner.
  * ActiveHandoff        — the mirror procedure, fully fake-testable by
    injecting a fake bus. On SUCCESS the leader torque is left ON
    (disable_after_sync=False default) so the arm holds; on FAILURE it is
    released to a hand-moveable safe state and the report says so.
"""
from __future__ import annotations

import math
import time
from typing import Protocol

from .mapping_cache import LeaderMappingCache
from .mapping_bridge import _PIPER_JOINT_TO_LEROBOT

LEROBOT_TO_JOINT = {v: k for k, v in _PIPER_JOINT_TO_LEROBOT.items()}
SYNC_TOL_DEG = 1.0        # max follower-equivalent joint error
SYNC_CONSECUTIVE = 3      # consecutive good frames required
MIRROR_FPS = 30


def _wrap(value: float, zero: float, wrap_ticks: float) -> float:
    if wrap_ticks <= 0.0:
        return value
    half = wrap_ticks / 2.0
    return zero + ((value - zero + half) % wrap_ticks) - half


class LeaderBus(Protocol):
    def read_ticks(self) -> dict[int, int]: ...
    def write_goal(self, ticks_by_id: dict[int, int], speed: int = 1500, acc: int = 50) -> None: ...
    def enable_torque(self, ids: list[int] | None = None) -> None: ...
    def disable_torque(self, ids: list[int] | None = None) -> None: ...


class FollowerToLeaderMap:
    """Analytic inverse of the kit mapping: Piper joints (deg) + gripper (m)
    -> leader servo ticks, with strict range validation (no clipping)."""

    def __init__(self, cache: LeaderMappingCache):
        self.cache = cache
        self.joint_maps = {jm.student_joint: jm for jm in cache.joint_maps}
        self.gripper_map = cache.gripper_map

    # ---- per-axis inverse -----------------------------------------------------
    def joint_deg_to_ticks(self, piper_joint: str, deg: float) -> int:
        jm = self.joint_maps[piper_joint]
        rad = math.radians(float(deg))
        if not (jm.student_min - 1e-9 <= rad <= jm.student_max + 1e-9):
            raise ValueError(
                f"{piper_joint} {deg:+.3f} deg outside student range "
                f"[{math.degrees(jm.student_min):+.2f},{math.degrees(jm.student_max):+.2f}]")
        scale_ticks = jm.teacher_scale_ticks
        if scale_ticks <= 1e-9:
            neg = max(0.0, jm.teacher_zero - jm.teacher_min)
            pos = max(0.0, jm.teacher_max - jm.teacher_zero)
            scale_ticks = max(neg, pos, 1.0)
        student_scale = max(abs(jm.student_min - jm.student_zero),
                            abs(jm.student_max - jm.student_zero))
        if student_scale <= 1e-9:
            raise ValueError(f"{piper_joint} degenerate student scale")
        delta = (rad - jm.student_zero) / student_scale * scale_ticks
        ticks = _wrap(jm.teacher_zero + jm.direction * delta,
                      jm.teacher_zero, jm.teacher_wrap_ticks)
        if not (jm.teacher_min - 1e-9 <= ticks <= jm.teacher_max + 1e-9):
            raise ValueError(
                f"{piper_joint} {deg:+.3f} deg maps outside leader travel "
                f"(ticks {ticks:.0f} not in [{jm.teacher_min:.0f},{jm.teacher_max:.0f}])")
        return int(round(ticks))

    def gripper_m_to_ticks(self, opening_m: float) -> int:
        gm = self.gripper_map
        if gm is None:
            raise ValueError("no gripper mapping")
        if not (gm.opening_min - 1e-9 <= opening_m <= gm.opening_max + 1e-9):
            raise ValueError(f"gripper {opening_m:.5f} m outside "
                             f"[{gm.opening_min:.5f},{gm.opening_max:.5f}]")
        delta = (opening_m - gm.opening_zero) / (gm.opening_max - gm.opening_min) \
            * max(gm.teacher_scale_ticks, 1.0)
        ticks = _wrap(gm.teacher_zero + gm.direction * delta,
                      gm.teacher_zero, gm.teacher_wrap_ticks)
        if not (gm.teacher_min - 1e-9 <= ticks <= gm.teacher_max + 1e-9):
            raise ValueError(f"gripper {opening_m:.5f} m maps outside leader travel")
        return int(round(ticks))

    def follower_pose_to_ticks(self, joint_deg: dict[str, float],
                               gripper_m: float | None = None) -> dict[int, int]:
        out = {}
        for lerobot_key, deg in joint_deg.items():
            joint = LEROBOT_TO_JOINT.get(lerobot_key, lerobot_key.replace(".pos", ""))
            jm = self.joint_maps.get(f"joint{joint[-1]}" if joint.startswith("J") else joint)
            if jm is None:
                raise ValueError(f"no mapping for {lerobot_key}")
            out[jm.teacher_id] = self.joint_deg_to_ticks(jm.student_joint, deg)
        if gripper_m is not None and self.gripper_map is not None:
            out[self.gripper_map.teacher_id] = self.gripper_m_to_ticks(gripper_m)
        return out

    # ---- forward (for the convergence check) ---------------------------------
    def ticks_to_follower_deg(self, ticks: dict[int, int]) -> dict[str, float]:
        """Leader ticks -> follower-equivalent J1..J6 deg (forward mapping)."""
        out = {}
        for jm in self.cache.joint_maps:
            if jm.teacher_id not in ticks:
                continue
            raw = ticks[jm.teacher_id]
            value = _wrap(raw, jm.teacher_zero, jm.teacher_wrap_ticks)
            delta = jm.direction * (value - jm.teacher_zero)
            scale_ticks = jm.teacher_scale_ticks
            if scale_ticks <= 1e-9:
                neg = max(0.0, jm.teacher_zero - jm.teacher_min)
                pos = max(0.0, jm.teacher_max - jm.teacher_zero)
                scale_ticks = max(neg, pos, 1.0)
            student_scale = max(abs(jm.student_min - jm.student_zero),
                                abs(jm.student_max - jm.student_zero))
            rad = jm.student_zero + (delta / scale_ticks) * student_scale
            rad = max(jm.student_min, min(jm.student_max, rad))
            out[f"J{jm.student_joint[-1]}.pos"] = math.degrees(rad)
        return out


class ActiveHandoff:
    """The mirror procedure. All robot/Piper interaction is OUT of scope here:
    the caller passes the follower pose; this class only touches the leader bus."""

    def __init__(self, bus: LeaderBus, fmap: FollowerToLeaderMap):
        self.bus = bus
        self.fmap = fmap
        self.last_error: str | None = None

    def sync_to_robot_pose(self, joint_deg: dict[str, float], gripper_m: float | None,
                           duration_s: float = 2.0, disable_after_sync: bool = False) -> dict:
        """Active mirror handoff. Returns a report dict; NEVER raises past this
        boundary — every failure path returns synced=False + reason + torque
        released (hand-moveable). A successful, converged mirror leaves the
        leader torque ON by default (the operator holds it; the caller releases
        torque only when formally entering CORRECTING)."""
        rep = {"synced": False, "reason": None, "final_error_deg": None,
               "consecutive_good": 0, "steps": 0, "torque_on": None}
        try:
            target = self.fmap.follower_pose_to_ticks(joint_deg, gripper_m)
        except ValueError as e:
            rep["reason"] = f"MIRROR_TARGET_OUT_OF_RANGE:{e}"
            rep["torque_on"] = False
            self.last_error = rep["reason"]
            return rep
        try:
            self.bus.enable_torque()                      # 1. torque ON
            start = self.bus.read_ticks()                 # 2. current position
            n_steps = max(2, int(duration_s * MIRROR_FPS))
            for k in range(1, n_steps + 1):               # 3. smooth interpolation
                alpha = k / n_steps
                interp = {sid: int(round(s + alpha * (t - s)))
                          for sid, t in target.items()
                          for s in [start.get(sid, t)]}
                self.bus.write_goal(interp)               # never a single jump
                rep["steps"] = k
                # per-tick feedback check (comm failure aborts)
                if not self.bus.read_ticks():
                    raise IOError("leader feedback stale during mirror")
            good = 0
            for _ in range(30):                           # 4. convergence gate
                fb = self.bus.read_ticks()
                if not fb:
                    raise IOError("leader feedback stale during convergence")
                equiv = self.fmap.ticks_to_follower_deg(fb)
                err = max(abs(equiv[k] - float(joint_deg[k])) for k in joint_deg)
                rep["final_error_deg"] = err
                good = good + 1 if err <= SYNC_TOL_DEG else 0
                rep["consecutive_good"] = good
                if good >= SYNC_CONSECUTIVE:
                    # 5. SAFE: leave torque ON so the arm HOLDs — the caller
                    # (e.g. Tab -> CORRECTING) decides when to release it.
                    if disable_after_sync:
                        self.bus.disable_torque()
                    rep["torque_on"] = not disable_after_sync
                    rep["synced"] = True
                    return rep
            rep["reason"] = (f"MIRROR_NOT_CONVERGED (final {rep['final_error_deg']:.3f} deg, "
                             f"need <= {SYNC_TOL_DEG} x {SYNC_CONSECUTIVE})")
        except Exception as e:                            # §11: any failure rejects
            rep["reason"] = f"TAKEOVER_REJECTED:{type(e).__name__}:{e}"
        try:
            self.bus.disable_torque()   # failure path: hand-moveable, never stuck ON
            rep["torque_on"] = False
        except Exception:
            pass
        self.last_error = rep["reason"]
        return rep
