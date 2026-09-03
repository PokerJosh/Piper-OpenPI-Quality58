# Piper 从臂：SDK/硬件侧 joint1～joint6 + 夹爪，对应 LeRobot 观测/动作键名 J1～J6、G。
# 顺序与 piper_interface 读取 joint_1…joint_6 一致。

ARM_JOINTS: list[str] = ["J1", "J2", "J3", "J4", "J5", "J6"]

GRIPPER_JOINT = "G"

# LeRobot observation / action 字典键（单位由 piper.py / 硬件层约定，关节一般为度）
MOTOR_POS_KEYS = [f"{joint}.pos" for joint in ARM_JOINTS]
MOTOR_VEL_KEYS = [f"{joint}.vel" for joint in ARM_JOINTS]

ACTION_POS_KEYS = [f"{joint}.pos" for joint in ARM_JOINTS]

GRIPPER_POS_KEY = f"{GRIPPER_JOINT}.pos"
