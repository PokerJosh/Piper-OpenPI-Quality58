from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_leader import PiperLeaderTeleopConfig
from .feetech_reader import FeetechLeaderReader
from .features import ALL_POS_KEYS
from .mapping_bridge import default_action_features, map_ticks_to_lerobot_action
from .mapping_cache import LeaderMappingCache

logger = logging.getLogger(__name__)


class PiperLeaderTeleop(Teleoperator):
    """
    飞特 DIY 主臂遥操作：串口读 ticks → piper_leader_kit 映射 → Piper 从臂 J1～J6 (度) + G (米)。

    主臂超出 teacher 行程或从臂超出 student 限位时，默认 ``clamp``：输出固定在边界，
    从臂无法被继续推过该限位（等价于对子变量做饱和/切除）。
    """

    config_class = PiperLeaderTeleopConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self.reader = FeetechLeaderReader(
            port=config.port,
            baudrate=config.baudrate,
            servo_ids=config.servo_ids,
            scservo_sdk_path=config.resolved_scservo_sdk_path(),
            max_tick_jump=config.max_tick_jump,
            hold_last_on_read_error=config.hold_last_on_read_error,
            require_all_servos=config.require_all_servos,
        )
        self._last_action: dict[str, float] = {}
        self._last_piper_positions: dict[str, float] = {}
        self._mapping_cache: LeaderMappingCache | None = None
        self._unchanged_action_frames = 0
        self._handoff = None   # ActiveHandoff (built lazily on connect)

    @property
    def action_features(self) -> dict[str, type]:
        return default_action_features()

    @property
    def feedback_features(self) -> dict[str, type]:
        # The leader is ACTUATED (torque + goal write, see active_handoff.py),
        # so LeRobot's DAgger `_teleop_supports_feedback` sees it as drivable.
        return default_action_features()

    @property
    def is_connected(self) -> bool:
        return self.reader.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        kit = self.config.resolved_leader_kit_path()
        if not kit.is_dir():
            raise FileNotFoundError(f"leader_kit_path 不存在: {kit}")
        mapping = self.config.resolved_mapping_json()
        if not mapping.is_file():
            raise FileNotFoundError(f"mapping_json 不存在: {mapping}")
        self.reader.connect()
        self._mapping_cache = LeaderMappingCache.load(kit, mapping)
        raw = self.reader.read_raw_by_id_or_raise()
        logger.info(
            "%s connected on %s (kit=%s, policy=%s) 首帧 ticks=%s",
            self,
            self.config.port,
            kit,
            self.config.out_of_range_policy,
            raw,
        )
        action, _, _ = map_ticks_to_lerobot_action(
            raw,
            mapping_cache=self._mapping_cache,
            out_of_range_policy=self.config.out_of_range_policy,
        )
        logger.info("首帧映射 → LeRobot 动作: %s", action)
        self._last_action = dict(action)
        self._last_piper_positions = {}

        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        """标定在 piper_leader_kit 中完成（mapping JSON + 零位文件）。"""
        kit = self.config.resolved_leader_kit_path()
        mapping_ok = self.config.resolved_mapping_json().is_file()
        zero_file = kit / "calibration" / "leader_zero_result.json"
        return mapping_ok and zero_file.is_file()

    def calibrate(self) -> None:
        logger.info(
            "主臂硬件标定请使用 piper_leader_kit: "
            "python %s/scripts/feetech_servo_zero.py",
            self.config.resolved_leader_kit_path(),
        )

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        raw_by_id = self.reader.read_raw_by_id()

        if raw_by_id is None:
            if self._last_action:
                logger.warning(
                    "主臂本帧读数失败，沿用上一帧（从臂不会跟新动作）。"
                    "请查看上条 Feetech 警告。"
                )
                return dict(self._last_action)
            raise RuntimeError("主臂读数失败且无上一帧缓存")

        action, mapped, saturated = map_ticks_to_lerobot_action(
            raw_by_id,
            mapping_cache=self._mapping_cache,
            out_of_range_policy=self.config.out_of_range_policy,
            last_piper_positions=self._last_piper_positions or None,
        )

        # reject 策略下未更新的关节：沿用上一帧 LeRobot 动作
        for key in ALL_POS_KEYS:
            if key not in action and key in self._last_action:
                action[key] = self._last_action[key]

        missing_keys = [k for k in ALL_POS_KEYS if k not in action]
        if missing_keys:
            logger.warning("映射缺少键 %s，从臂可能收不全指令", missing_keys)

        if saturated and any(saturated.values()):
            logger.debug("限位饱和关节: %s", {k: v for k, v in saturated.items() if v})

        if self._last_action and action == self._last_action:
            self._unchanged_action_frames += 1
            # 只在300帧（10秒）时警告一次，避免频繁打印
            if self._unchanged_action_frames == 300:
                logger.warning(
                    "主臂动作已连续 %d 帧未变化（ticks=%s）。"
                    "若正在扳动主臂：检查 max_tick_jump、串口是否被占用、舵机是否上电。",
                    self._unchanged_action_frames,
                    raw_by_id,
                )
        else:
            self._unchanged_action_frames = 0

        self._last_action = dict(action)
        self._last_piper_positions = dict(mapped)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s get_action %.1f ms", self, dt_ms)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """ACTIVE-HANDOFF: drive the leader toward the follower pose.

        Accepts {"joints": {J1.pos..J6.pos deg}, "gripper_m": float} (the
        format produced by sync_to_robot_pose) or a flat LeRobot-style dict of
        J*.pos/G.pos. The leader is ACTUATED (torque + goal write on the same
        serial owner as reading); the follower/Piper never moves for sync.
        """
        if self._handoff is None:
            raise NotImplementedError("active handoff not initialized (not connected?)")
        joints = feedback.get("joints", {k: v for k, v in feedback.items()
                                         if k.startswith("J") and k.endswith(".pos")})
        gripper = feedback.get("gripper_m", feedback.get("G.pos"))
        self._handoff.bus.write_goal(
            self._handoff.fmap.follower_pose_to_ticks(joints, gripper))

    # ---- ACTIVE LEADER MIRROR HANDOFF --------------------------------------
    def enable_torque(self) -> None:
        self.reader.enable_torque()

    def disable_torque(self) -> None:
        self.reader.disable_torque()

    def sync_to_robot_pose(self, follower_joint_deg: dict[str, float],
                           gripper_m: float | None = None,
                           duration_s: float = 2.0,
                           disable_after_sync: bool = False) -> dict:
        """Torque ON → smooth mirror to the follower pose → convergence gate
        (≤1.0 deg equivalent error x 3 frames). On success the leader torque
        REMAINS ON by default (the arm holds; release only when formally
        entering CORRECTING). Returns the ActiveHandoff report dict
        ({"synced": bool, "reason": ..., "torque_on": bool})."""
        if self._handoff is None:
            from .active_handoff import ActiveHandoff, FollowerToLeaderMap
            self._handoff = ActiveHandoff(None, FollowerToLeaderMap(self._mapping_cache))
        self._handoff.bus = self.reader
        return self._handoff.sync_to_robot_pose(follower_joint_deg, gripper_m, duration_s,
                                                disable_after_sync=disable_after_sync)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.reader.disconnect()
        logger.info("%s disconnected", self)
