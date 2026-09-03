from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

try:
    from lerobot.types import RobotAction, RobotObservation
except ImportError:
    RobotAction = dict[str, Any]
    RobotObservation = dict[str, Any]

from .config_piper import PiperRobotConfig
from .features import (
    ACTION_POS_KEYS,
    ARM_JOINTS,
    GRIPPER_POS_KEY,
    MOTOR_POS_KEYS,
)
from .piper_interface import PiperHardware

logger = logging.getLogger(__name__)


class PiperRobot(Robot):
    """
    LeRobot 机器人插件：Piper 从臂。

    这一层负责把 LeRobot 的统一接口转换成 PiperHardware 的底层接口。
    """

    config_class = PiperRobotConfig
    name = "piper"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config

        self.hardware = PiperHardware(can_name=config.can_name, config=config)
        self.cameras = make_cameras_from_configs(config.cameras)

        # 内部指令位姿（用于限速插值）；与 get_observation 读到的真实反馈分离
        self._command_joint_pos_deg: list[float] | None = None
        self._last_send_action_time: float | None = None

    @property
    def observation_features(self) -> dict:
        features = {}

        for key in MOTOR_POS_KEYS:
            features[key] = float

        if self.config.use_gripper:
            features[GRIPPER_POS_KEY] = float

        for cam_key, cam in self.cameras.items():
            features[cam_key] = (cam.height, cam.width, 3)

        return features

    @property
    def action_features(self) -> dict:
        features = {}

        for key in ACTION_POS_KEYS:
            features[key] = float

        if self.config.use_gripper:
            features[GRIPPER_POS_KEY] = float

        return features

    @property
    def is_connected(self) -> bool:
        return self.hardware.is_connected

    def motion_command_counters(self) -> dict[str, int]:
        """Expose hardware-boundary motion counters for deployment reports."""
        return self.hardware.motion_command_counters()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        self.hardware.connect()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        logger.info("Connected to Piper robot on %s", self.config.can_name)

    def go_home(self) -> None:
        """回零位并保持使能（需已 connect 且已使能）。"""
        if not self.is_connected:
            raise ConnectionError("PiperRobot is not connected.")
        self.hardware.go_home()

    def disable(self) -> None:
        """掉使能（卸力矩），不断开 CAN。建议先 go_home() 或扶住机械臂。"""
        if not self.is_connected:
            raise ConnectionError("PiperRobot is not connected.")
        self.hardware.disable_arm()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()

        self.hardware.disconnect()
        logger.info("Disconnected from Piper robot.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if self.config.enable_on_connect:
            self.hardware.enable_arm(timeout_s=self.config.enable_timeout_s)
        self._command_joint_pos_deg = self.hardware.read_joint_positions_deg()

    def _max_joint_step_deg(self) -> float:
        step = self.config.max_relative_target_deg
        if self._last_send_action_time is not None:
            dt = time.time() - self._last_send_action_time
            step = min(step, self.config.max_joint_speed_deg_s * max(dt, 1e-3))
        return step

    def _joint_soft_limits_deg(self, joint_name: str) -> tuple[float, float]:
        lo, hi = self.config.joint_limits_deg[joint_name]
        margin = self.config.limit_margin_deg
        return lo + margin, hi - margin

    def _clip_gripper_target_m(self, opening_m: float) -> float:
        lo, hi = self.config.gripper_limits_m
        return max(lo, min(hi, opening_m))

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError("PiperRobot is not connected.")

        obs = {}

        joints_deg = self.hardware.read_joint_positions_deg()

        for key, value in zip(MOTOR_POS_KEYS, joints_deg, strict=True):
            obs[key] = float(value)

        if self.config.use_gripper:
            obs[GRIPPER_POS_KEY] = float(self.hardware.read_gripper_opening_m())

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError("PiperRobot is not connected.")

        if self._command_joint_pos_deg is None:
            self._command_joint_pos_deg = self.hardware.read_joint_positions_deg()

        max_step = self._max_joint_step_deg()
        command_deg: list[float] = []
        sent_action: RobotAction = {}

        for index, key in enumerate(ACTION_POS_KEYS):
            joint_name = ARM_JOINTS[index]
            if key not in action:
                logger.warning("动作缺少 %s，本关节保持当前位置", key)
                command_deg.append(float(self._command_joint_pos_deg[index]))
                sent_action[key] = float(self._command_joint_pos_deg[index])
                continue

            raw_target = float(action[key])
            current = float(self._command_joint_pos_deg[index])
            lo_m, hi_m = self._joint_soft_limits_deg(joint_name)

            delta = raw_target - current
            delta = max(-max_step, min(max_step, delta))
            safe_target = max(lo_m, min(hi_m, current + delta))
            command_deg.append(safe_target)
            sent_action[key] = safe_target

        gripper_target_m: float | None = None
        if self.config.use_gripper and GRIPPER_POS_KEY in action:
            gripper_target_m = self._clip_gripper_target_m(float(action[GRIPPER_POS_KEY]))
            sent_action[GRIPPER_POS_KEY] = gripper_target_m

        if self.config.enable_motion:
            self.hardware.send_joint_positions_deg(command_deg)
            if (
                self.config.send_gripper
                and gripper_target_m is not None
            ):
                self.hardware.send_gripper_opening_m(gripper_target_m)
        else:
            logger.warning(
                "Motion disabled. Not sending action to Piper. "
                "Set robot.enable_motion=true only after read-state test is correct."
            )

        self._command_joint_pos_deg = command_deg
        self._last_send_action_time = time.time()

        return sent_action
