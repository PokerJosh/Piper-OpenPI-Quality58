# 主臂编码器 → 松灵 PiPER：轴对应与角度转换

## 1. 编码器读数

- 量程：0~4095 ticks/圈（`ticks_per_revolution = 4096`）
- **零位**：EEPROM 标定后 `present = 2047`
- 原始话题：`/uarm/raw_servo_positions`（`Int32MultiArray`，下标 0 对应 ID1）

## 2. 主臂相对弧度（未做松灵限位）

对舵机 `id`：

```
θ_teacher = direction × (unwrap(ticks) - 2047) / 4096 × 2π
```

`direction` 见 `config/leader_to_piper_mapping.example.json` → `axis_table`。

## 3. 松灵关节输出（RViz / 从臂）

采用 **delta 模式**（`joint_maps[].mode = "delta"`）：

```
Δticks = direction × (unwrap(ticks) - teacher_zero)
q_joint = clamp( student_zero + Δticks/teacher_scale_ticks × student_span , student_min, student_max )
```

- `student_zero`：松灵零位姿态下关节角，均为 **0 rad**（夹爪 **0 m**）
- `student_min/max`：松灵 URDF 限位（弧度）
- `teacher_min/max/scale_ticks`：主臂在该关节上的行程；超出时见 `out_of_range_policy`

夹爪（ID7）：

```
opening = clamp( Δticks/scale × (opening_max - opening_min), 0, 0.07 )
joint7 = opening ,  joint8 = -opening
```

## 4. 方向与限位一览

| ID | PiPER | direction | student_min | student_max |
|----|-------|-----------|-------------|-------------|
| 1 | joint1 | -1 | -2.618 | 2.168 |
| 2 | joint2 | -1 | 0 | 3.14 |
| 3 | joint3 | -1 | -2.967 | 0 |
| 4 | joint4 | -1 | -1.745 | 1.745 |
| 5 | joint5 | +1 | -1.22 | 1.22 |
| 6 | joint6 | -1 | -2.0944 | 2.0944 |
| 7 | gripper | -1 | 0 m | 0.07 m |

RViz 中某关节**反向运动**：将该关节 `direction` 改为相反符号，重新 `colcon build` 后启动。

## 5. 标定结果字段

`calibration/leader_zero_result.example.json`：

- `homing_offset` / `eeprom_encoded`：写入舵机的偏移
- `present_after_calib`：应为 2047

## 6. 防抖（ROS2）

| 参数 | 文件 | 含义 |
|------|------|------|
| `hold_last_on_read_error` | leader_feetech_reader.example.yaml | 读失败保持上一帧 ticks |
| `max_tick_jump` | 同上 | 单帧野值丢弃 |
| `smoothing_alpha` | leader_to_piper_mapper.example.yaml | 输出 EMA 平滑 |
| `max_joint_step_rad` | 同上 | 单帧最大关节变化 |
| `require_all_servos` | 同上 | 7 路齐才更新 |

## 7. 修改映射后

1. 编辑 `config/leader_to_piper_mapping.example.json`
2. `python3 scripts/leader_servo_mapping.py` 验证
3. 重启 launch（mapper 支持热加载 json，但建议重启）
