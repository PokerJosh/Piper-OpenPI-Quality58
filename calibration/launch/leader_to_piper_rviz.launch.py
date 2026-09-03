"""主臂 -> 松灵 PiPER RViz（使用 piper_leader_kit 配置）。"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

KIT_ROOT = Path("<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit")
PIPER_URDF = Path(
    "<PIPER_PROJECT_ROOT>/ros_workspace/workspace/controllab/external/piper_ros_foxy/"
    "src/piper_description/urdf/piper_description.urdf"
)


def _robot_description() -> str:
    mesh_dir = PIPER_URDF.parent.parent / "meshes"
    text = PIPER_URDF.read_text(encoding="utf-8")
    return text.replace("package://piper_description/meshes/", f"file://{mesh_dir}/")


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="<LEADER_SERIAL_PORT>"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "mapping_file",
                default_value=str(KIT_ROOT / "config/leader_to_piper_mapping.json"),
            ),
            DeclareLaunchArgument(
                "reader_config",
                default_value=str(KIT_ROOT / "config/leader_feetech_reader.example.yaml"),
            ),
            DeclareLaunchArgument(
                "mapper_config",
                default_value=str(KIT_ROOT / "config/leader_to_piper_mapper.example.yaml"),
            ),
            DeclareLaunchArgument("out_of_range_policy", default_value="clamp"),
            Node(
                package="uarm_piper_teleop",
                executable="uarm_feetech_reader_node",
                name="uarm_feetech_reader",
                output="screen",
                parameters=[
                    LaunchConfiguration("reader_config"),
                    {"port": LaunchConfiguration("port")},
                ],
            ),
            Node(
                package="uarm_piper_teleop",
                executable="piper_virtual_mapper_node",
                name="piper_virtual_mapper",
                output="screen",
                parameters=[
                    LaunchConfiguration("mapper_config"),
                    {
                        "mapping_file": LaunchConfiguration("mapping_file"),
                        "out_of_range_policy": LaunchConfiguration("out_of_range_policy"),
                    },
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": _robot_description()}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("uarm_piper_teleop"), "rviz", "piper_teleop.rviz"]
                    ),
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )
