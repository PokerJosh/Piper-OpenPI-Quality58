"""主臂 ticks → LeRobot 动作（J1～J6 度，G 米），含限位饱和（clamp）。"""

from __future__ import annotations

import math
from pathlib import Path

from .features import ACTION_POS_KEYS, GRIPPER_POS_KEY
from .mapping_cache import LeaderMappingCache

# joint1..joint6 → J1..J6
_PIPER_JOINT_TO_LEROBOT: dict[str, str] = {
    f"joint{i}": f"J{i}" for i in range(1, 7)
}


def saturate_to_limits(
    value: float,
    low: float,
    high: float,
) -> float:
    """将标量限制在 [low, high]；超出时输出边界值（从臂不再被继续推离限位）。"""
    return max(low, min(high, value))


def map_ticks_to_lerobot_action(
    raw_by_id: dict[int, int],
    *,
    leader_kit_path: Path | None = None,
    mapping_json: Path | None = None,
    mapping_cache: LeaderMappingCache | None = None,
    out_of_range_policy: str = "clamp",
    last_piper_positions: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, bool]]:
    """
    返回:
      - action: LeRobot 键 J1.pos..G.pos
      - mapped_piper: 中间量 joint1..joint7（弧度/米）
      - saturated_flags: 各关节是否已顶在从臂软限位上
    """
    if mapping_cache is None:
        if leader_kit_path is None:
            raise ValueError("leader_kit_path or mapping_cache is required")
        mapping_path = mapping_json or (
            leader_kit_path / "config" / "leader_to_piper_mapping.json"
        )
        mapping_cache = LeaderMappingCache.load(leader_kit_path, mapping_path)

    joint_maps = mapping_cache.joint_maps
    gripper_map = mapping_cache.gripper_map
    mapped = mapping_cache.map_raw_dict_to_piper(
        raw_by_id,
        joint_maps,
        gripper_map,
        out_of_range_policy=out_of_range_policy,
        last_positions=last_piper_positions,
    )

    action: dict[str, float] = {}
    saturated_flags: dict[str, bool] = {}

    for jm in joint_maps:
        key = jm.student_joint
        if key not in mapped:
            continue
        rad = float(mapped[key])
        clamped_rad = saturate_to_limits(rad, jm.student_min, jm.student_max)
        saturated_flags[key] = abs(clamped_rad - rad) > 1e-9 or abs(
            clamped_rad - jm.student_min
        ) < 1e-6 or abs(clamped_rad - jm.student_max) < 1e-6

        lerobot_joint = _PIPER_JOINT_TO_LEROBOT.get(key)
        if lerobot_joint is None:
            continue
        action[f"{lerobot_joint}.pos"] = math.degrees(clamped_rad)

    if gripper_map is not None:
        opening_key = gripper_map.joint7
        if opening_key in mapped:
            opening_m = float(mapped[opening_key])
            clamped_m = saturate_to_limits(
                opening_m, gripper_map.opening_min, gripper_map.opening_max
            )
            saturated_flags[opening_key] = abs(clamped_m - opening_m) > 1e-9
            action[GRIPPER_POS_KEY] = clamped_m

    # 保证 LeRobot 从臂 send_action 需要的键齐全
    last = last_piper_positions or {}
    for jm in joint_maps:
        lerobot_joint = _PIPER_JOINT_TO_LEROBOT.get(jm.student_joint)
        if lerobot_joint is None:
            continue
        lerobot_key = f"{lerobot_joint}.pos"
        if lerobot_key not in action and jm.student_joint in last:
            action[lerobot_key] = math.degrees(float(last[jm.student_joint]))

    return action, mapped, saturated_flags


def default_action_features() -> dict[str, type]:
    return {key: float for key in ACTION_POS_KEYS + [GRIPPER_POS_KEY]}
