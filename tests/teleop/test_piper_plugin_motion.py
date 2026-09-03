"""Piper 插件：读状态 → 动 J1 → 可选夹爪 → 断开时回零（默认不掉使能）。"""

import time

from lerobot_robot_piper import PiperRobot, PiperRobotConfig

# USB 换口后若不是 can0，改这里
CAN_NAME = "can0"

config = PiperRobotConfig(
    can_name=CAN_NAME,
    enable_motion=True,
    enable_on_connect=True,
    max_relative_target_deg=10.0,
    motion_command_duration_s=1.0,
    disable_torque_on_disconnect=False,
    go_home_on_disconnect=True,
)

robot = PiperRobot(config)
robot.connect()

obs0 = robot.get_observation()
pos_keys = [k for k in obs0 if k.endswith(".pos")]

print("Before:")
for k in pos_keys:
    print(f"  {k}: {obs0[k]}")

action = {k: obs0[k] for k in pos_keys}
action["J1.pos"] = float(obs0["J1.pos"]) + 5.0
action["G.pos"] = 0.035  # 单边 3.5 cm

sent = robot.send_action(action)
print("Sent:")
for k in pos_keys:
    print(f"  {k}: {sent[k]}")

time.sleep(1.5)

obs1 = robot.get_observation()
print("After:")
for k in pos_keys:
    print(f"  {k}: {obs1[k]}")

robot.disconnect()
print("Done (disconnect: go home, keep torque until CAN closed).")
