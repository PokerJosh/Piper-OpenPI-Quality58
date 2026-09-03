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
import os
import time
import logging
from pathlib import Path
import threading
import termios
import tty
import select

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加 lerobot 到路径
sys.path.insert(0, str(Path(__file__).parent / "lerobot/src"))

# 注册第三方插件
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.configs.parser import parse_lerobot_config
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep


class KeyboardReader:
    """非阻塞键盘读取"""
    def __init__(self):
        self.key = None
        self.running = True
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """后台线程持续读取键盘"""
        try:
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.key = ch
        except Exception as e:
            logger.error(f"键盘读取错误: {e}")

    def get_key(self):
        """获取最新按键并清除"""
        key = self.key
        self.key = None
        return key

    def stop(self):
        self.running = False
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


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
    # 使用现有的配置文件加载方式
    config_path = Path("piper_camera_config.yaml")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    print_instructions()
    logger.info("正在加载配置...")

    # 使用 lerobot 的配置解析
    from lerobot.scripts.lerobot_record import RecordConfig
    import draccus
    import yaml

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = draccus.decode(RecordConfig, cfg_dict)

    logger.info("正在连接设备...")

    # 初始化机器人和遥操作
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()

    logger.info("✓ 设备已连接")
    logger.info(f"✓ 相机: {list(robot.cameras.keys())}")
    logger.info(f"✓ 数据集: {cfg.dataset.repo_id}")

    # 等待相机预热
    time.sleep(2)

    # 获取 features - 从一次实际读取中构建
    logger.info("正在初始化数据集...")
    obs = robot.get_observation()
    action = teleop.get_action()

    # 构建 features
    from lerobot.datasets.utils import create_lerobot_dataset
    dataset = create_lerobot_dataset(
        dataset_repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type="piper",
        robot=robot,
        use_videos=cfg.dataset.video,
    )

    episode_index = len(dataset.episode_ids) if dataset.episode_ids else 0

    # 键盘读取器
    kb = KeyboardReader()

    # 状态变量
    STATE_IDLE = 'idle'
    STATE_RECORDING = 'recording'
    STATE_STOPPED = 'stopped'

    state = STATE_IDLE
    episode_buffer = []
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
                    episode_buffer = []
                    frame_count = 0
                    start_time = time.time()
                    dataset.start_episode()
                    logger.info("\n🔴 开始录制 Episode %d (按 's' 停止)", episode_index)

                elif key == 'q':
                    logger.info("\n退出程序...")
                    break

                elif key:
                    logger.warning("无效按键 '%s'，当前状态: 等待中 (按 'c' 开始)", key)

                time.sleep(0.01)

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

                # 添加帧到数据集
                dataset.add_frame(observation, action)
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

                    # 结束并保存 episode
                    dataset.save_episode(episode_index, encode_videos=(not cfg.dataset.streaming_encoding))

                    logger.info("✓ Episode %d 已保存", episode_index)

                    # 重置状态
                    episode_index += 1
                    state = STATE_IDLE
                    logger.info("\n等待指令... (按 'c' 开始录制第 %d 条)", episode_index)

                elif key == 'b':
                    # 丢弃数据 - 清空当前 episode 的 buffer
                    logger.info("\n🗑  已丢弃 Episode %d", episode_index)
                    dataset.clear_episode_buffer()
                    state = STATE_IDLE
                    logger.info("\n等待指令... (按 'c' 开始录制第 %d 条)", episode_index)

                elif key:
                    logger.warning("无效按键 '%s'，请按 'g' 保存或 'b' 丢弃", key)

                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\n\n收到中断信号，正在退出...")

    finally:
        # 清理资源
        kb.stop()
        logger.info("正在断开连接...")

        try:
            robot.disconnect()
        except Exception as e:
            logger.error(f"断开机器人失败: {e}")

        try:
            teleop.disconnect()
        except Exception as e:
            logger.error(f"断开遥操作失败: {e}")

        logger.info("✓ 已退出")
        logger.info(f"✓ 共录制 {episode_index} 条数据")


if __name__ == "__main__":
    main()
