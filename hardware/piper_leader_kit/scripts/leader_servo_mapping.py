#!/usr/bin/env python3
"""主臂飞特 ticks -> 弧度 -> 松灵 PiPER 关节（含限位与方向）。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KIT_ROOT = Path(__file__).resolve().parents[2] / "piper_leader_kit"
DEFAULT_MAPPING = KIT_ROOT / "config" / "leader_to_piper_mapping.json"


@dataclass(frozen=True)
class JointMap:
    teacher_id: int
    student_joint: str
    teacher_min: float
    teacher_zero: float
    teacher_max: float
    student_min: float
    student_zero: float
    student_max: float
    direction: float
    teacher_wrap_ticks: float
    mode: str
    teacher_scale_ticks: float


@dataclass(frozen=True)
class GripperMap:
    teacher_id: int
    teacher_zero: float
    teacher_min: float
    teacher_max: float
    teacher_wrap_ticks: float
    direction: float
    teacher_scale_ticks: float
    opening_zero: float
    opening_min: float
    opening_max: float
    joint7: str
    joint8: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def unwrap_ticks(value: float, zero: float, wrap_ticks: float) -> float:
    if wrap_ticks <= 0.0:
        return value
    half = wrap_ticks / 2.0
    return zero + ((value - zero + half) % wrap_ticks) - half


def raw_ticks_to_teacher_rad(
    raw_ticks: int | float,
    *,
    teacher_zero: float = 2047.0,
    direction: float = 1.0,
    ticks_per_revolution: float = 4096.0,
    wrap_ticks: float = 4096.0,
) -> float:
    """编码器 ticks -> 相对零位的弧度（未做从臂限位）。"""
    unwrapped = unwrap_ticks(float(raw_ticks), teacher_zero, wrap_ticks)
    delta_ticks = direction * (unwrapped - teacher_zero)
    return delta_ticks / ticks_per_revolution * math.tau


def teacher_in_travel_range(
    raw_ticks: float,
    joint_map: JointMap,
) -> bool:
    value = unwrap_ticks(raw_ticks, joint_map.teacher_zero, joint_map.teacher_wrap_ticks)
    return joint_map.teacher_min <= value <= joint_map.teacher_max


def map_joint_delta(
    raw_ticks: float,
    joint_map: JointMap,
    *,
    out_of_range_policy: str = "clamp",
) -> float | None:
    """单关节：ticks -> 松灵弧度。reject 时超主臂行程返回 None。"""
    value = unwrap_ticks(raw_ticks, joint_map.teacher_zero, joint_map.teacher_wrap_ticks)
    if out_of_range_policy == "reject" and not (joint_map.teacher_min <= value <= joint_map.teacher_max):
        return None

    delta = joint_map.direction * (value - joint_map.teacher_zero)
    scale_ticks = joint_map.teacher_scale_ticks
    if scale_ticks <= 1e-9:
        neg_span = max(0.0, joint_map.teacher_zero - joint_map.teacher_min)
        pos_span = max(0.0, joint_map.teacher_max - joint_map.teacher_zero)
        scale_ticks = max(neg_span, pos_span, 1.0)

    student_scale = max(
        abs(joint_map.student_min - joint_map.student_zero),
        abs(joint_map.student_max - joint_map.student_zero),
    )
    target = joint_map.student_zero + (delta / scale_ticks) * student_scale
    return _clamp(target, joint_map.student_min, joint_map.student_max)


def map_gripper_opening(
    raw_ticks: float,
    gripper: GripperMap,
    *,
    out_of_range_policy: str = "clamp",
) -> float | None:
    value = unwrap_ticks(raw_ticks, gripper.teacher_zero, gripper.teacher_wrap_ticks)
    if out_of_range_policy == "reject" and not (gripper.teacher_min <= value <= gripper.teacher_max):
        return None
    delta = gripper.direction * (value - gripper.teacher_zero)
    opening = gripper.opening_zero + (
        delta / max(gripper.teacher_scale_ticks, 1.0)
    ) * (gripper.opening_max - gripper.opening_min)
    return _clamp(opening, gripper.opening_min, gripper.opening_max)


def load_mapping(path: str | Path = DEFAULT_MAPPING) -> tuple[list[JointMap], GripperMap | None, dict[str, Any]]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    joint_maps: list[JointMap] = []
    for item in raw.get("joint_maps", []):
        neg_span = max(0.0, float(item["teacher_zero"]) - float(item["teacher_min"]))
        pos_span = max(0.0, float(item["teacher_max"]) - float(item["teacher_zero"]))
        joint_maps.append(
            JointMap(
                teacher_id=int(item["teacher_id"]),
                student_joint=str(item["student_joint"]),
                teacher_min=float(item["teacher_min"]),
                teacher_zero=float(item["teacher_zero"]),
                teacher_max=float(item["teacher_max"]),
                student_min=float(item["student_min"]),
                student_zero=float(item["student_zero"]),
                student_max=float(item["student_max"]),
                direction=float(item["direction"]),
                teacher_wrap_ticks=float(item.get("teacher_wrap_ticks", 4096.0)),
                mode=str(item.get("mode", "delta")),
                teacher_scale_ticks=float(
                    item.get("teacher_scale_ticks", max(neg_span, pos_span, 1.0))
                ),
            )
        )

    gripper_raw = raw.get("gripper_map")
    gripper_map = None
    if gripper_raw is not None:
        gripper_map = GripperMap(
            teacher_id=int(gripper_raw["teacher_id"]),
            teacher_zero=float(gripper_raw["teacher_zero"]),
            teacher_min=float(gripper_raw["teacher_min"]),
            teacher_max=float(gripper_raw["teacher_max"]),
            teacher_wrap_ticks=float(gripper_raw.get("teacher_wrap_ticks", 4096.0)),
            direction=float(gripper_raw.get("direction", 1.0)),
            teacher_scale_ticks=float(gripper_raw["teacher_scale_ticks"]),
            opening_zero=float(gripper_raw.get("opening_zero", 0.0)),
            opening_min=float(gripper_raw.get("opening_min", 0.0)),
            opening_max=float(gripper_raw.get("opening_max", 0.07)),
            joint7=str(gripper_raw.get("joint7", "joint7")),
            joint8=str(gripper_raw.get("joint8", "joint8")),
        )

    return joint_maps, gripper_map, raw


def map_raw_dict_to_piper(
    raw_by_id: dict[int, int | float],
    joint_maps: list[JointMap],
    gripper_map: GripperMap | None = None,
    *,
    out_of_range_policy: str = "clamp",
    last_positions: dict[str, float] | None = None,
) -> dict[str, float]:
    """将 {servo_id: ticks} 转为 {joint_name: rad_or_m}。"""
    out = dict(last_positions or {})
    for jm in joint_maps:
        raw = raw_by_id.get(jm.teacher_id)
        if raw is None or int(raw) < 0:
            continue
        value = map_joint_delta(float(raw), jm, out_of_range_policy=out_of_range_policy)
        if value is None:
            continue
        out[jm.student_joint] = value

    if gripper_map is not None:
        raw = raw_by_id.get(gripper_map.teacher_id)
        if raw is not None and int(raw) >= 0:
            opening = map_gripper_opening(
                float(raw), gripper_map, out_of_range_policy=out_of_range_policy
            )
            if opening is not None:
                out[gripper_map.joint7] = opening
                out[gripper_map.joint8] = -opening
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试主臂 ticks -> 松灵关节映射")
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING),
        help="映射 JSON 路径",
    )
    parser.add_argument(
        "--ticks",
        default="2047,2047,2047,2047,2047,2047,2047",
        help="7 个舵机 ticks，逗号分隔",
    )
    parser.add_argument(
        "--policy",
        choices=("clamp", "reject"),
        default="clamp",
    )
    args = parser.parse_args()

    joint_maps, gripper_map, meta = load_mapping(args.mapping)
    ticks = [int(x) for x in args.ticks.split(",")]
    raw_by_id = {i + 1: t for i, t in enumerate(ticks)}
    result = map_raw_dict_to_piper(
        raw_by_id, joint_maps, gripper_map, out_of_range_policy=args.policy
    )
    print("policy:", args.policy)
    print("encoder:", meta.get("encoder", {}))
    for name in sorted(result):
        print(f"  {name}: {result[name]:.6f}")
