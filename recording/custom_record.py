#!/usr/bin/env python3
"""
自定义 Piper 录制脚本 - 按键控制版本

操作流程：
- 程序启动后等待
- 按 'c': 开始录制当前 episode
- 按 's': 停止录制
- 按 'g': 保存这条数据
- 按 'b': 丢弃这条数据
- 按 'q' 或 Ctrl+C: 退出程序
"""

import sys
import time
import logging
from pathlib import Path
from collections import deque
import threading
import termios
import tty

import yaml
import draccus

# 注册第三方插件（Piper）
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.datasets.image_writer import AsyncImageWriter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class KeyboardReader:
    """非阻塞键盘读取"""
    def __init__(self):
        self.key = None
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """后台线程持续读取键盘"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                ch = sys.stdin.read(1)
                if ch:
                    self.key = ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_key(self):
        """获取最新按键并清除"""
        key = self.key
        self.key = None
        return key

    def stop(self):
        self.running = False


def print_instructions():
    """打印操作说明"""
    print("\n" + "="*60)
    print("  Piper 自定义录制 - 按键控制模式")
    print("="*60)
    print("  [c] 开始录制")
    print("  [s] 停止录制")
    print("  [g] 保存这条数据 (Good)")
    print("  [b] 丢弃这条数据 (Bad)")
    print("  [q] 退出程序")
    print("  [Ctrl+C] 强制退出")
    print("="*60 + "\n")


def main():
    # 加载配置
    config_path = Path("piper_camera_config.yaml")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # 解析配置
    from lerobot.scripts.lerobot_record import RecordConfig
    cfg = draccus.decode(RecordConfig, cfg_dict)

    print_instructions()
    logger.info("正在连接设备...")

    # 初始化机器人和遥操作
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()

    logger.info("✓ 设备已连接")
    logger.info(f"✓ 相机: {list(robot.cameras.keys())}")
    logger.info(f"✓ 数据集: {cfg.dataset.repo_id}")

    # 创建或加载数据集
    if cfg.resume:
        logger.info("加载已有数据集...")
        dataset = LeRobotDataset.load(cfg.dataset.repo_id, cfg.dataset.root)
        episode_index = len(dataset.episode_ids)
    else:
        logger.info("创建新数据集...")

        # 构建 features 字典
        features = {}
        # 从 robot 获取状态和动作的 features
        obs = robot.get_observation()
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                features[f"observation.{key}"] = {"dtype": str(value.dtype), "shape": list(value.shape)}

        action = teleop.get_action()
        for key, value in action.items():
            if isinstance(value, (int, float)):
                features[f"action.{key}"] = {"dtype": "float32", "shape": [1]}

        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset.repo_id,
            fps=cfg.dataset.fps,
            features=features,
            robot_type=cfg.robot.type,
            root=cfg.dataset.root,
            streaming_encoding=cfg.dataset.streaming_encoding,
        )
        episode_index = 0

    # 启动图像写入器
    image_writer = AsyncImageWriter(
        num_processes=cfg.dataset.num_image_writer_processes,
        num_threads=cfg.dataset.num_image_writer_threads_per_camera,
    )

    # 键盘读取器
    kb = KeyboardReader()

    # 状态变量
    STATE_IDLE = 'idle'
    STATE_RECORDING = 'recording'
    STATE_STOPPED = 'stopped'

    state = STATE_IDLE
    episode_data = []
    frame_count = 0
    start_time = None

    try:
        logger.info("\n等待指令... (按 'c' 开始录制第 %d 条)", episode_index)

        while True:
            key = kb.get_key()

            # === IDLE 状态 ===
            if state == STATE_IDLE:
                if key == 'c':
                    # 开始录制
                    state = STATE_RECORDING
                    episode_data = []
                    frame_count = 0
                    start_time = time.time()
                    logger.info("\n🔴 开始录制 Episode %d (按 's' 停止)", episode_index)

                elif key == 'q':
                    logger.info("\n退出程序...")
                    break

                elif key:
                    logger.warning("无效按键 '%s'，当前状态: 等待中 (按 'c' 开始)", key)

            # === RECORDING 状态 ===
            elif state == STATE_RECORDING:
                if key == 's':
                    # 停止录制
                    state = STATE_STOPPED
                    duration = time.time() - start_time
                    logger.info("\n⏸  录制停止 - 共 %d 帧, 时长 %.1f 秒", frame_count, duration)
                    logger.info("   [g] 保存  [b] 丢弃")
                    continue

                # 采集数据
                loop_start = time.time()

                # 读取遥操作动作
                action = teleop.get_action()

                # 读取机器人状态
                observation = robot.get_observation()

                # 发送动作到机器人
                robot.send_action(action)

                # 存储这一帧
                frame = {
                    'observation': observation,
                    'action': action,
                    'timestamp': time.time() - start_time,
                }
                episode_data.append(frame)
                frame_count += 1

                # 控制帧率
                dt = 1.0 / cfg.dataset.fps
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    precise_sleep(sleep_time)

                # 每1秒打印一次
                if frame_count % cfg.dataset.fps == 0:
                    logger.info("  录制中... %d 帧 (%.1f 秒)", frame_count, time.time() - start_time)

            # === STOPPED 状态 ===
            elif state == STATE_STOPPED:
                if key == 'g':
                    # 保存数据
                    logger.info("\n💾 正在保存 Episode %d (%d 帧)...", episode_index, frame_count)

                    # 将数据写入数据集
                    for frame_idx, frame in enumerate(episode_data):
                        # 保存图像到异步写入器
                        for cam_key, img in frame['observation'].items():
                            if cam_key in robot.cameras:
                                image_writer.save_image(
                                    img,
                                    dataset.root / f"videos/observation.images.{cam_key}/chunk-000/file-{episode_index:03d}-{frame_idx:06d}.png"
                                )
                        # 添加帧数据
                        dataset.add_frame(frame['observation'], frame['action'])

                    # 等待图像写入完成
                    image_writer.wait_until_done()

                    # 结束 episode
                    dataset.save_episode(episode_index, encode_videos=(not cfg.dataset.streaming_encoding))

                    logger.info("✓ Episode %d 已保存", episode_index)

                    # 重置状态
                    episode_index += 1
                    episode_data = []
                    state = STATE_IDLE
                    logger.info("\n等待指令... (按 'c' 开始录制第 %d 条)", episode_index)

                elif key == 'b':
                    # 丢弃数据
                    logger.info("\n🗑  已丢弃 Episode %d", episode_index)
                    episode_data = []
                    state = STATE_IDLE
                    logger.info("\n等待指令... (按 'c' 开始录制第 %d 条)", episode_index)

                elif key:
                    logger.warning("无效按键 '%s'，请按 'g' 保存或 'b' 丢弃", key)

            else:
                # 其他状态，短暂休眠避免CPU占用
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\n\n收到中断信号，正在退出...")

    finally:
        # 清理资源
        kb.stop()
        logger.info("正在断开连接...")

        try:
            image_writer.stop()
        except:
            pass

        try:
            robot.disconnect()
        except:
            pass

        try:
            teleop.disconnect()
        except:
            pass

        logger.info("✓ 已退出")
        logger.info(f"✓ 共录制 {episode_index} 条数据")


if __name__ == "__main__":
    main()
