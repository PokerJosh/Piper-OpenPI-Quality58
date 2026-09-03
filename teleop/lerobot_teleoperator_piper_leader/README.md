# Piper Leader → LeRobot Teleoperator

主臂：`piper_leader_kit`（Feetech 串口 + `leader_servo_mapping.py`）
从臂：`lerobot_robot_piper`（CAN）

## 安装（两个插件都要装，缺一不可）

```bash
conda activate lerobot
pip install -e <PIPER_PROJECT_ROOT>/external/lerobot_robot_piper
pip install -e <PIPER_PROJECT_ROOT>/external/lerobot_teleoperator_piper_leader
# 主臂串口依赖（scservo_sdk 需要）
pip install pyserial
```

若 `lerobot-teleoperate` 报 `invalid choice: 'piper_leader'`，说明**主臂 teleop 包未安装**（从臂 `piper` 已装也会报这个错）。

验证：

```bash
python -c "from lerobot.utils.import_utils import register_third_party_plugins; from lerobot.teleoperators.config import TeleoperatorConfig; register_third_party_plugins(); print('piper_leader' in TeleoperatorConfig.get_known_choices())"
# 应打印 True
```

## 限位逻辑（clamp）

- `out_of_range_policy=clamp`（默认）：主臂超出 `teacher_min/max` 或映射结果超出 `student_min/max` 时，**输出钳制到边界值**，从臂不会继续被推向限位外。
- `out_of_range_policy=reject`：超出主臂行程的关节**保持上一帧**，该自由度暂时不驱动从臂。

配置与标定仍编辑：`<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit/config/leader_to_piper_mapping.example.json`

## 遥操作（待 CAN + 串口就绪）

```bash
sudo ip link set <CAN_INTERFACE> up type can bitrate 1000000

lerobot-teleoperate \
  --teleop.type=piper_leader \
  --teleop.port=<LEADER_SERIAL_PORT> \
  --teleop.leader_kit_path=<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit \
  --robot.type=piper \
  --robot.can_name=<CAN_INTERFACE> \
  --robot.enable_motion=true \
  --fps=30
```

## 录制

```bash
lerobot-record \
  --teleop.type=piper_leader \
  --teleop.port=<LEADER_SERIAL_PORT> \
  --robot.type=piper \
  --robot.can_name=<CAN_INTERFACE> \
  --dataset.repo_id=local/piper_teleop_demo \
  --dataset.num_episodes=5 \
  --dataset.single_task="pick and place"
```
