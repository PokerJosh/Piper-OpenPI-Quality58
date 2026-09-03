from dataclasses import dataclass, field
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

DEFAULT_LEADER_KIT_PATH = Path("<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit")


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderTeleopConfig(TeleoperatorConfig):
    """Piper 主臂（Feetech 串口）→ 与从臂一致的 J1～J6 / G 动作键。"""

    # 主臂串口
    port: str = "<LEADER_SERIAL_PORT>"
    baudrate: int = 1_000_000
    servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # piper_leader_kit 根目录（含 config/、scripts/leader_servo_mapping.py、sdk/）
    leader_kit_path: str = str(DEFAULT_LEADER_KIT_PATH)

    # 映射 JSON；默认使用 kit 内 leader_to_piper_mapping.json
    mapping_json: str | None = None

    # 飞特 SDK 路径；默认 leader_kit_path/sdk
    scservo_sdk_path: str | None = None

    # 超限策略：clamp=输出钳制到限位边界；reject=该关节保持上一帧（不推动从臂）
    out_of_range_policy: str = "clamp"

    # 单帧 tick 跳变超过该值则沿用上一帧；0=关闭滤波
    max_tick_jump: int = 0
    hold_last_on_read_error: bool = True
    # False：允许部分舵机失败时仍遥操作；True：7 路全齐才更新（一路掉线则整帧冻结）
    require_all_servos: bool = False

    def resolved_leader_kit_path(self) -> Path:
        return Path(self.leader_kit_path).expanduser().resolve()

    def resolved_mapping_json(self) -> Path:
        if self.mapping_json:
            return Path(self.mapping_json).expanduser().resolve()
        return self.resolved_leader_kit_path() / "config" / "leader_to_piper_mapping.json"

    def resolved_scservo_sdk_path(self) -> Path:
        if self.scservo_sdk_path:
            return Path(self.scservo_sdk_path).expanduser().resolve()
        return self.resolved_leader_kit_path() / "sdk"
