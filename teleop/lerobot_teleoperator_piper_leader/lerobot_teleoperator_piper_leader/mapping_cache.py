"""连接时加载一次映射，避免每帧 import leader_servo_mapping。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

def _import_kit_mapping(leader_kit_path: Path) -> tuple:
    scripts = leader_kit_path / "scripts"
    scripts_str = str(scripts.resolve())
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    from leader_servo_mapping import load_mapping, map_raw_dict_to_piper

    return load_mapping, map_raw_dict_to_piper


@dataclass
class LeaderMappingCache:
    leader_kit_path: Path
    mapping_json: Path
    joint_maps: tuple[Any, ...]
    gripper_map: Any
    map_raw_dict_to_piper: Callable[..., dict[str, float]]

    @classmethod
    def load(cls, leader_kit_path: Path, mapping_json: Path) -> LeaderMappingCache:
        load_mapping, map_raw_dict_to_piper = _import_kit_mapping(leader_kit_path)
        joint_maps, gripper_map, _meta = load_mapping(mapping_json)
        return cls(
            leader_kit_path=leader_kit_path,
            mapping_json=mapping_json,
            joint_maps=joint_maps,
            gripper_map=gripper_map,
            map_raw_dict_to_piper=map_raw_dict_to_piper,
        )
