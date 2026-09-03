# 与 lerobot_robot_piper.features 对齐，便于 leader → follower 直连

ARM_JOINTS: list[str] = ["J1", "J2", "J3", "J4", "J5", "J6"]
GRIPPER_JOINT = "G"

ACTION_POS_KEYS = [f"{joint}.pos" for joint in ARM_JOINTS]
GRIPPER_POS_KEY = f"{GRIPPER_JOINT}.pos"

ALL_POS_KEYS = ACTION_POS_KEYS + [GRIPPER_POS_KEY]
