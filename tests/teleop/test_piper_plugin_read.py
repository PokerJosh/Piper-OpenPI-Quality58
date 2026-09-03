from lerobot_robot_piper import PiperRobotConfig, PiperRobot

config = PiperRobotConfig(
    can_name="can0",
    enable_motion=False,
    use_gripper=True,
)

robot = PiperRobot(config)
robot.connect()

print("Connected:", robot.is_connected)
print("Observation features:")
print(robot.observation_features)

obs = robot.get_observation()
print("Observation:")
for k, v in obs.items():
    if "image" not in k and "camera" not in k:
        print(k, v)

robot.disconnect()
