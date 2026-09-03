from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .config_piper import PiperRobotConfig

logger = logging.getLogger(__name__)

# SDK 关节角 / 夹爪行程换算
_JOINT_SDK_SCALE = 1000  # 0.001° per integer
_GRIPPER_SDK_SCALE = 1_000_000  # 0.001 mm per integer → LeRobot 用米: m * 1e6
_GRIPPER_EFFORT_SDK_SCALE = 1000  # 0.001 N·m per integer


class PiperHardware:
    """
    Piper SDK 的轻量包装层。

    这一层只负责和 piper_sdk 通信。
    不要在这里写 LeRobot 逻辑。
    """

    def __init__(self, can_name: str = "<CAN_INTERFACE>", config: PiperRobotConfig | None = None):
        self.can_name = can_name
        self.config = config
        self.piper = None
        self._connected = False
        self._arm_enabled = False
        self._last_command_time = 0.0
        # Counters are deliberately kept at the hardware boundary so direct
        # recovery calls and PiperRobot.send_action() use the same accounting.
        # "sent" means the SDK method returned; piper_sdk currently consumes
        # per-CAN-frame status internally, so this is not a CAN ACK.
        self._motion_command_attempt_count = 0
        self._motion_command_sent_count = 0
        self._joint_position_command_attempt_count = 0
        self._joint_position_command_sent_count = 0
        self._gripper_command_attempt_count = 0
        self._gripper_command_sent_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def motion_command_counters(self) -> dict[str, int]:
        """Return hardware-boundary command counters for one robot object.

        The sent counters indicate that the corresponding piper_sdk call
        returned.  They must not be interpreted as a CAN acknowledgement,
        because the current SDK does not propagate SendCanMessage status.
        """
        return {
            "motion_command_attempt_count": self._motion_command_attempt_count,
            "motion_command_sent_count": self._motion_command_sent_count,
            "joint_position_command_attempt_count": self._joint_position_command_attempt_count,
            "joint_position_command_sent_count": self._joint_position_command_sent_count,
            "gripper_command_attempt_count": self._gripper_command_attempt_count,
            "gripper_command_sent_count": self._gripper_command_sent_count,
        }

    def connect(self) -> None:
        try:
            from piper_sdk import C_PiperInterface_V2 as PiperInterface
        except ImportError:
            from piper_sdk import C_PiperInterface as PiperInterface

        sdk_kwargs = {
            "can_name": self.can_name,
            "judge_flag": False,
            "can_auto_init": True,
            "start_sdk_joint_limit": False,
            "start_sdk_gripper_limit": False,
        }
        try:
            self.piper = PiperInterface(**sdk_kwargs)
        except TypeError:
            # 旧版 piper_sdk 仅接受 can_name 位置参数
            self.piper = PiperInterface(self.can_name)
        except ConnectionError as exc:
            bitrate = 1_000_000 if self.config is None else int(self.config.can_bitrate)
            raise ConnectionError(
                f"无法打开 CAN 接口 '{self.can_name}'（SDK: {exc}）。"
                "请先执行: ip link | grep -E 'can[0-9]' 确认网卡名；"
                f"再执行: sudo ip link set {self.can_name} up type can bitrate {bitrate}"
                "（若换过 USB 口，网卡可能是 can1，请在 PiperRobotConfig(can_name=...) 里改）。"
            ) from exc

        self.piper.ConnectPort()
        time.sleep(0.1)
        self._connected = True
        self._arm_enabled = False

    def go_home(self) -> None:
        """在保持使能的前提下，向零位目标连续下发关节/夹爪指令。"""
        if not self._connected or not self._arm_enabled:
            return
        if self.config is None:
            return

        joints_deg = list(self.config.home_joint_positions_deg)
        duration_s = max(0.0, float(self.config.home_motion_duration_s))
        interval_s = self._command_interval_s()
        frame_count = max(1, int(self.config.command_rate_hz * duration_s))

        joints_sdk = [int(round(float(x) * _JOINT_SDK_SCALE)) for x in joints_deg]
        gripper_m = float(self.config.home_gripper_opening_m)
        angle_sdk = int(round(max(0.0, gripper_m) * _GRIPPER_SDK_SCALE))
        effort = self._gripper_effort_sdk()

        logger.info(
            "Piper go home: joints_deg=%s duration_s=%.2f frames=%d",
            joints_deg,
            duration_s,
            frame_count,
        )

        for frame in range(frame_count):
            self._prepare_joint_motion()
            self.piper.JointCtrl(*joints_sdk)
            if self.config.use_gripper and hasattr(self.piper, "GripperCtrl"):
                self.piper.GripperCtrl(angle_sdk, effort, 0x01, 0)
            if frame + 1 < frame_count:
                time.sleep(interval_s)

        self._last_command_time = time.time()

    def shutdown_before_disconnect(self) -> None:
        """断开 CAN 前：可选回零，可选掉使能。"""
        if not self._connected or self.piper is None:
            return

        if self.config and self.config.go_home_on_disconnect and self._arm_enabled:
            self.go_home()
            settle_s = max(0.0, float(self.config.home_settle_s))
            if settle_s > 0:
                time.sleep(settle_s)

        if self._arm_enabled and (
            self.config is None or self.config.disable_torque_on_disconnect
        ):
            self.disable_arm()

    def disconnect(self) -> None:
        try:
            self.shutdown_before_disconnect()
        finally:
            if self.piper is not None and hasattr(self.piper, "DisconnectPort"):
                self.piper.DisconnectPort()

            self.piper = None
            self._connected = False
            self._arm_enabled = False

    def enable_arm(self, timeout_s: float = 6.0) -> None:
        if not self._connected:
            raise ConnectionError("Piper is not connected.")
        if not hasattr(self.piper, "EnableArm"):
            raise RuntimeError("piper_sdk has no EnableArm method.")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if hasattr(self.piper, "EnablePiper"):
                self.piper.EnablePiper()
            else:
                self.piper.EnableArm(7)
            status = self.piper.GetArmEnableStatus()
            if len(status) >= 6 and all(status[:6]):
                self._arm_enabled = True
                if self.config and self.config.use_gripper:
                    self._enable_gripper_motor()
                logger.info("Piper arm enabled (6 joints).")
                return
            time.sleep(0.01)

        raise TimeoutError(f"Piper arm enable timed out after {timeout_s}s.")

    def disable_arm(self) -> None:
        if not self._connected or self.piper is None:
            return
        if hasattr(self.piper, "DisableArm"):
            self.piper.DisableArm(7)
        self._arm_enabled = False
        logger.info("Piper arm disabled.")

    def _enable_gripper_motor(self) -> None:
        effort = self._gripper_effort_sdk()
        if hasattr(self.piper, "GripperCtrl"):
            self.piper.GripperCtrl(0, effort, 0x02, 0)
            self.piper.GripperCtrl(0, effort, 0x01, 0)

    def _prepare_joint_motion(self) -> None:
        speed = 50
        if self.config is not None:
            speed = int(self.config.speed_percent)
        speed = max(0, min(100, speed))
        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)

    def _command_burst_count(self) -> int:
        if self.config is None or self.config.command_rate_hz <= 0:
            return 1
        duration = max(0.0, float(self.config.motion_command_duration_s))
        return max(1, int(self.config.command_rate_hz * duration))

    def _command_interval_s(self) -> float:
        if self.config is None or self.config.command_rate_hz <= 0:
            return 0.005
        return 1.0 / self.config.command_rate_hz

    def _gripper_effort_sdk(self) -> int:
        effort_nm = 1.0 if self.config is None else self.config.gripper_effort_nm
        effort_nm = max(0.0, min(5.0, float(effort_nm)))
        return int(round(effort_nm * _GRIPPER_EFFORT_SDK_SCALE))

    @staticmethod
    def _joint_field_names() -> list[str]:
        return ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    def read_joint_positions_deg(self) -> list[float]:
        if not self._connected:
            raise ConnectionError("Piper is not connected.")

        raw = self.piper.GetArmJointMsgs()
        candidate = raw.joint_state if hasattr(raw, "joint_state") else raw

        values: list[float] = []
        for name in self._joint_field_names():
            if hasattr(candidate, name):
                values.append(float(getattr(candidate, name)) / _JOINT_SDK_SCALE)

        if len(values) == 6:
            return values

        values = []
        for name in ["j1", "j2", "j3", "j4", "j5", "j6"]:
            if hasattr(candidate, name):
                values.append(float(getattr(candidate, name)) / _JOINT_SDK_SCALE)

        if len(values) == 6:
            return values

        raise RuntimeError(
            "Cannot parse joint positions from Piper SDK message. "
            f"Raw message type={type(raw)}, raw={raw}"
        )

    def read_gripper_opening_m(self) -> float:
        if not self._connected:
            raise ConnectionError("Piper is not connected.")
        if not hasattr(self.piper, "GetArmGripperMsgs"):
            return 0.0

        raw = self.piper.GetArmGripperMsgs()
        state = raw.gripper_state if hasattr(raw, "gripper_state") else raw
        if hasattr(state, "grippers_angle"):
            return float(state.grippers_angle) / _GRIPPER_SDK_SCALE
        if hasattr(state, "gripper_angle"):
            return float(state.gripper_angle) / _GRIPPER_SDK_SCALE
        return 0.0

    def send_joint_positions_deg(self, joints_deg: Sequence[float]) -> None:
        if not self._connected:
            raise ConnectionError("Piper is not connected.")
        if not self._arm_enabled:
            raise RuntimeError("Piper arm is not enabled. Call enable_arm() first.")

        if len(joints_deg) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joints_deg)}")

        if not hasattr(self.piper, "JointCtrl"):
            raise RuntimeError("Current piper_sdk object has no JointCtrl method.")

        joints_sdk = [int(round(float(x) * _JOINT_SDK_SCALE)) for x in joints_deg]
        self._motion_command_attempt_count += 1
        self._joint_position_command_attempt_count += 1
        interval_s = self._command_interval_s()
        for frame in range(self._command_burst_count()):
            self._prepare_joint_motion()
            self.piper.JointCtrl(*joints_sdk)
            if frame + 1 < self._command_burst_count():
                time.sleep(interval_s)

        self._motion_command_sent_count += 1
        self._joint_position_command_sent_count += 1
        self._last_command_time = time.time()

    def send_gripper_opening_m(self, opening_m: float) -> None:
        if not self._connected:
            raise ConnectionError("Piper is not connected.")
        if not hasattr(self.piper, "GripperCtrl"):
            raise RuntimeError("Current piper_sdk object has no GripperCtrl method.")

        angle_sdk = int(round(max(0.0, float(opening_m)) * _GRIPPER_SDK_SCALE))
        self._motion_command_attempt_count += 1
        self._gripper_command_attempt_count += 1
        effort = self._gripper_effort_sdk()
        interval_s = self._command_interval_s()
        for frame in range(self._command_burst_count()):
            self.piper.GripperCtrl(angle_sdk, effort, 0x01, 0)
            if frame + 1 < self._command_burst_count():
                time.sleep(interval_s)

        self._motion_command_sent_count += 1
        self._gripper_command_sent_count += 1
        self._last_command_time = time.time()

    def get_firmware_version(self):
        if not self._connected:
            raise ConnectionError("Piper is not connected.")

        if hasattr(self.piper, "GetPiperFirmwareVersion"):
            return self.piper.GetPiperFirmwareVersion()

        return None
