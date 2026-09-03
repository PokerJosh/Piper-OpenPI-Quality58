#!/usr/bin/env python3
"""最小闭环：主臂读数 → 从臂 send_action，打印反馈是否变化。"""

import time

from lerobot_robot_piper import PiperRobot, PiperRobotConfig
from lerobot_teleoperator_piper_leader import PiperLeaderTeleop, PiperLeaderTeleopConfig

LEADER_KIT = "<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit"
PORT = "/dev/ttyACM0"
CAN = "can0"


def main() -> None:
    teleop = PiperLeaderTeleop(
        PiperLeaderTeleopConfig(
            port=PORT,
            leader_kit_path=LEADER_KIT,
            require_all_servos=False,
            max_tick_jump=0,
        )
    )
    robot = PiperRobot(
        PiperRobotConfig(
            can_name=CAN,
            enable_motion=True,
            max_relative_target_deg=15.0,
            motion_command_duration_s=0.02,
        )
    )
    teleop.connect()
    robot.connect()
    print("扳动主臂 5 秒，观察 obs 与 sent 是否变化…")
    try:
        for i in range(150):
            action = teleop.get_action()
            obs = robot.get_observation()
            sent = robot.send_action(action)
            if i % 15 == 0:
                j1_obs = obs.get("J1.pos", 0.0)
                j1_sent = sent.get("J1.pos", 0.0)
                j1_tgt = action.get("J1.pos", 0.0)
                print(
                    f"[{i:03d}] leader J1={j1_tgt:7.2f}  sent J1={j1_sent:7.2f}  obs J1={j1_obs:7.2f}"
                )
            time.sleep(1.0 / 30.0)
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
