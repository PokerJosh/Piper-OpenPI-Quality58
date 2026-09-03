import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

# student 限位 (rad)，来自 Piper 从臂文档；在配置里换算为度供软件限幅使用
_PIPER_JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "J1": (-2.618, 2.168),
    "J2": (0.0, 3.14),
    "J3": (-2.967, 0.0),
    "J4": (-1.745, 1.745),
    "J5": (-1.22, 1.22),
    "J6": (-2.0944, 2.0944),
}


def _rad_limits_to_deg(
    limits_rad: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {
        name: (math.degrees(lo), math.degrees(hi)) for name, (lo, hi) in limits_rad.items()
    }


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    # CAN 设备名：USB 换口后可能是 can1；先用 `ip link | grep can` 确认
    can_name: str = "<CAN_INTERFACE>"

    # CAN 波特率 (Hz)，Piper 默认 1 Mbps
    can_bitrate: int = 1_000_000

    # 是否使用夹爪
    use_gripper: bool = True

    # 安全开关：刚开始建议 False，只读不动
    enable_motion: bool = True

    # 单次动作最大允许变化角度，单位：度
    max_relative_target_deg: float = 12.0

    # 各关节软限位 (度)，键名与 features.ARM_JOINTS 一致 (J1～J6)
    joint_limits_deg: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _rad_limits_to_deg(_PIPER_JOINT_LIMITS_RAD)
    )

    # 单夹爪最大开口行程 (米)，7 cm
    gripper_max_opening_m: float = 0.07

    # 夹爪软限位 (米)，闭合为 0，最大开口为 gripper_max_opening_m；键名与 features.GRIPPER_JOINT 一致
    gripper_limits_m: tuple[float, float] = (0.0, 0.07)

    # 下发到 Piper 控制器的全局速度百分比 (0～100)
    speed_percent: int = 70

    # 关节运动速度上限 (度/秒)，用于限速/插值
    max_joint_speed_deg_s: float = 60.0

    # 向机械臂发送目标位姿的频率 (Hz)
    command_rate_hz: float = 50.0

    # 每次 send_action 连续下发 CAN 的时长 (秒)。遥操作主循环会周期性调用 send_action，
    # 默认只发 1 帧，避免阻塞；单独测动作用 test_piper_plugin_motion 可调大。
    motion_command_duration_s: float = 0.02

    # 在 joint_limits_deg 内侧再缩进的软限位余量 (度)，避免贴硬限位
    limit_margin_deg: float = 1.0

    # 连接后是否自动使能电机
    enable_on_connect: bool = True

    # 等待使能完成的超时 (秒)
    enable_timeout_s: float = 6.0

    # 断开连接时是否掉使能；False=保持力矩（建议配合 go_home_on_disconnect，避免松脱摔落）
    disable_torque_on_disconnect: bool = False

    # 断开前先回零位并持续下发指令（保持使能，直到回零指令发完）
    go_home_on_disconnect: bool = True

    # 回零目标关节角 (度)，顺序 J1～J6；请按你机械臂真实安全零位修改
    home_joint_positions_deg: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # 断开前回零时连续下发 CAN 的时长 (秒)
    home_motion_duration_s: float = 3.0

    # 回零时夹爪开口 (米)
    home_gripper_opening_m: float = 0.0

    # 回零指令发完后的等待 (秒)，再执行掉使能 / 关闭 CAN
    home_settle_s: float = 0.5

    # 是否在动作流里下发夹爪目标
    send_gripper: bool = True

    # 夹爪抓取时的力矩/力矩上限 (N·m)，配合位置目标做力矩限幅，不是纯力矩控制模式
    gripper_effort_nm: float = 1.0

    # 相机配置，暂时可以为空
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
