#!/usr/bin/env python3
"""
自定义 Piper 录制脚本 - 按键控制版本

操作流程：
- 按 'c': 开始录制
- 按 's': 停止录制
- 按 'g': 保存 / 按 'b': 丢弃
- 按 'q': 退出
"""

import argparse
import sys
import time
import logging
from pathlib import Path
import threading
import termios
import tty
import select
import re
import yaml
import draccus
import numpy as np
from episode_tools import commit_episode, discard_episode, load_config as load_review_config, load_raw_episode, suggest_trim_range, validate_episode
from record_review_guard import DEFAULT_REVIEW_CONFIG, recovery_guard

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 注册插件
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.robot_utils import precise_sleep


class KeyboardReader:
    def __init__(self):
        self.key = None
        self.running = True
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.key = ch
        except:
            pass

    def get_key(self):
        key = self.key
        self.key = None
        return key

    def stop(self):
        self.running = False
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        except:
            pass


def save_frame_to_disk(frame_data, episode_dir, frame_idx):
    """保存单帧数据到磁盘"""
    import pickle
    frame_file = episode_dir / f"frame_{frame_idx:06d}.pkl"
    with open(frame_file, 'wb') as f:
        pickle.dump(frame_data, f)


def load_episode_frames(episode_dir):
    """加载 episode 的所有帧"""
    import pickle
    frames = []
    frame_files = sorted(episode_dir.glob("frame_*.pkl"))
    for frame_file in frame_files:
        with open(frame_file, 'rb') as f:
            frames.append(pickle.load(f))
    return frames


_EPISODE_DIR_PATTERN = re.compile(r"episode_(\d+)(?:_temp)?$")


def get_next_episode_index(dataset_root: Path) -> int:
    """Return an unused episode index without reusing final or temp directories.

    Dataset folders can have gaps after manual cleanup, and an interrupted run can
    leave ``episode_XXXXXX_temp`` behind. Counting folders is therefore unsafe:
    it can select an index that already exists and cause ``rename`` to fail.
    """
    used_indices = []
    for path in dataset_root.iterdir():
        if not path.is_dir():
            continue
        match = _EPISODE_DIR_PATTERN.fullmatch(path.name)
        if match:
            used_indices.append(int(match.group(1)))
    return max(used_indices, default=-1) + 1


def reserve_episode_temp_dir(dataset_root: Path) -> tuple[int, Path]:
    """Atomically reserve an unused temporary directory for a new recording."""
    episode_index = get_next_episode_index(dataset_root)
    while True:
        temp_dir = dataset_root / f"episode_{episode_index:06d}_temp"
        final_dir = dataset_root / f"episode_{episode_index:06d}"
        if final_dir.exists():
            episode_index += 1
            continue
        try:
            temp_dir.mkdir(parents=False, exist_ok=False)
            return episode_index, temp_dir
        except FileExistsError:
            # Another recorder or an interrupted run occupies this number.
            episode_index += 1


def next_episode_index(*roots: Path) -> int:
    """Reserve numbers across raw temp and separately reviewed final roots."""
    used = []
    for root in roots:
        if root.exists():
            for path in root.glob("episode_*"):
                match = _EPISODE_DIR_PATTERN.fullmatch(path.name)
                if match:
                    used.append(int(match.group(1)))
    return max(used, default=-1) + 1


def trim_summary(frames, auto):
    active = auto.get("activity", {}).get("active")
    active_indices = np.flatnonzero(active) if active is not None else []
    first = int(active_indices[0]) if len(active_indices) else "NONE"
    last = int(active_indices[-1]) if len(active_indices) else "NONE"
    n, start, end = len(frames), auto["start"], auto["end"]
    logger.info("\n%s\nAUTO TRIM SUMMARY\n%s", "=" * 60, "=" * 60)
    logger.info("ORIGINAL_FRAMES = %d\nFIRST_ACTIVE_FRAME = %s\nLAST_ACTIVE_FRAME = %s", n, first, last)
    logger.info("AUTO_TRIM_START = %d\nAUTO_TRIM_END = %d\nREMOVE_HEAD = %d\nREMOVE_TAIL = %d\nKEEP_FRAMES = %d", start, end, start, n - 1 - end, end - start + 1)
    logger.info("PRE_ROLL_KEPT = %d\nPOST_ROLL_KEPT = %d\nAUTO_TRIM_VALIDATION = %s", auto["start"] if isinstance(first, int) else 0, (end - last) if isinstance(last, int) else 0, "PASS" if auto["confidence"] != "low" else "WARNING")
    logger.info("NEXT_ACTION = Press g to accept + auto trim/QA/dry-run; press b if this demonstration failed.\n")


def run_dry_run(final_dir: Path, review_config: dict):
    """Offline only: does not construct a robot or camera object."""
    from replay_episode import preflight
    frames = load_raw_episode(final_dir)
    validation = validate_episode(final_dir, review_config["fps"])
    replay = preflight(frames, review_config)
    replay["DATA_VALID"] = bool(replay["DATA_VALID"] and validation["valid"])
    logger.info("REPLAY_DRY_RUN\nPIPER_CONNECTED = NO\nMOTION = DISABLED")
    logger.info("DATA_VALID = %s\nREAL_REPLAY_SAFE_UNDER_CURRENT_GATE = %s", "YES" if replay["DATA_VALID"] else "NO", "YES" if replay["REAL_REPLAY_SAFE_UNDER_CURRENT_GATE"] else "NO")
    return validation, replay


def print_demonstration_qa(index: int, manifest: dict, validation: dict, replay: dict, approved_total: int, reviewed_root: Path):
    camera = "FAIL" if validation["CAMERA_FRAME_COMPLETENESS"] == "FAIL" else ("WARNING" if validation["warnings"] else "PASS")
    required = approved_total == 5
    logger.info("\n%s\nDEMONSTRATION QA\n%s", "=" * 60, "=" * 60)
    logger.info("EPISODE = %06d\nORIGINAL_FRAMES = %s\nTRIMMED_FRAMES = %s", index, manifest["original_frame_count"], manifest["saved_frame_count"])
    logger.info("TRIM_START = %s\nTRIM_END = %s\nHEAD_REMOVED = %s\nTAIL_REMOVED = %s", manifest["trim_start_original"], manifest["trim_end_original"], manifest["trim_start_original"], manifest["original_frame_count"] - 1 - manifest["trim_end_original"])
    logger.info("FRAME_ALIGNMENT = %s\nTIMESTAMP = %s\nCAMERA_QA = %s\nNaN_INF = %s", validation["FRAME_ALIGNMENT"], validation["TIMESTAMP_VALID"], camera, "PASS" if validation["valid"] else "FAIL")
    logger.info("DATA_VALID = %s\nDRY_RUN = %s\nREAL_REPLAY_SAFE = %s\nAPPROVED_TOTAL = %d\nCURRENT_STAGE = FIRST_5", "YES" if replay["DATA_VALID"] else "NO", "PASS" if replay["DATA_VALID"] else "FAIL", "YES" if replay["REAL_REPLAY_SAFE_UNDER_CURRENT_GATE"] else "NO", approved_total)
    logger.info("REAL_REPLAY_REQUIRED_NOW = %s", "YES" if required else "NO")
    if required:
        candidates = [p.name for p in sorted(reviewed_root.glob("episode_*"))[:2]]
        logger.warning("REAL_REPLAY_RECOMMENDED = YES\nCANDIDATES = %s\nNEXT_ACTION = STOP BULK COLLECTION. Wait for explicit user confirmation before one --real replay.", candidates)
    else:
        logger.info("NEXT_ACTION = Return robot to initial position OUTSIDE RECORDING, prepare scene, then press c for next demonstration.")
def finalize_episode(temp_dir: Path, dataset_root: Path, preferred_index: int) -> tuple[int, Path]:
    """Safely rename a temporary episode to an unused final directory.

    Never overwrite an existing episode. A collision should not discard the data
    that was just recorded, even if directories were manually added mid-recording.
    """
    episode_index = preferred_index
    while True:
        final_dir = dataset_root / f"episode_{episode_index:06d}"
        if not final_dir.exists():
            temp_dir.rename(final_dir)
            return episode_index, final_dir
        logger.warning("Episode %d 已存在，保留旧数据并改用下一个编号。", episode_index)
        episode_index += 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Piper 手动录制(按键控制)。传 --review-config <recovery 配置> 时启用 RECOVERY_PATH_GUARD。")
    parser.add_argument("--camera-config", default="piper_camera_config.yaml",
                        help="camera/robot/teleop 配置 YAML(默认 piper_camera_config.yaml)")
    parser.add_argument("--review-config", default=None,
                        help="review 配置 YAML(默认: 脚本目录 record_review_config.yaml)")
    parser.add_argument("--dry-check-config", action="store_true",
                        help="只解析并打印配置/路径/保护状态后退出；不录制、不连接设备")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    config_path = Path(args.camera_config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  Piper 自定义录制 - 按键控制模式")
    print("="*60)
    print("  [c] 开始录制")
    print("  [s] 停止录制")
    print("  [g] 接受：自动裁剪 → 验证 → 离线 Dry-run → 原子提交")
    print("  [b] 丢弃当前失败示教")
    print("  [q] 退出")
    print("="*60 + "\n")

    logger.info("正在加载配置...")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = draccus.decode(RecordConfig, cfg_dict)
    review_config = load_review_config(args.review_config)

    logger.info("新采集阶段: FIRST_5；AUTO_TRIM_ONLY=YES；每条 Dry-run=必做；Real Replay=仅提示、绝不自动执行。")
    logger.info("正确边界：初始静止→抓取/放置/松夹→最终静止→s；s 后回初始位绝不录入。")

    # 数据集目录 + 恢复路径保护：必须在任何 mkdir / 设备连接 / 录制 / 机器人动作之前
    configured_raw = review_config.get("recording", {}).get("raw_root")
    configured_reviewed = review_config.get("recording", {}).get("reviewed_root")
    if configured_raw:
        dataset_root = Path(configured_raw)
    elif cfg.dataset.root:
        dataset_root = Path(cfg.dataset.root) / cfg.dataset.repo_id.replace('/', '_')
    else:
        dataset_root = Path.home() / ".cache/huggingface/lerobot" / cfg.dataset.repo_id.replace('/', '_')

    reviewed_root = Path(configured_reviewed) if configured_reviewed else dataset_root

    active_review_config = args.review_config if args.review_config else DEFAULT_REVIEW_CONFIG
    guard_on, guard_ok, guard_reason = recovery_guard(args.review_config, dataset_root, reviewed_root)
    print(f"ACTIVE_REVIEW_CONFIG = {active_review_config}")
    print(f"ACTIVE_RAW_ROOT = {dataset_root}")
    print(f"ACTIVE_REVIEWED_ROOT = {reviewed_root}")
    print(f"RECOVERY_PATH_GUARD = {'ON' if guard_on else 'OFF'}")
    if guard_on and not guard_ok:
        logger.error("RECOVERY_PATH_GUARD %s", guard_reason)
        logger.error("REFUSE_RECORDING: 未开始录制、未连接机器人、未下发任何动作。")
        sys.exit(1)
    if args.dry_check_config:
        print("DRY_CHECK_CONFIG = OK (配置与路径已解析；未录制、未连接设备、未下发动作)")
        sys.exit(0)

    logger.info("正在连接设备...")
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()

    logger.info("✓ 机器人已连接")
    logger.info("✓ 遥操作已连接")
    logger.info("✓ 相机: %s", list(robot.cameras.keys()))

    # 硬件预热 - 固定等待5秒
    logger.info("等待硬件初始化完成... (5秒)")
    time.sleep(5)

    logger.info("✓ 硬件就绪\n")

    dataset_root.mkdir(parents=True, exist_ok=True)
    reviewed_root.mkdir(parents=True, exist_ok=True)

    # 目录可能有编号空洞或遗留的 _temp 目录；按最大已用编号分配，绝不能用目录数量。
    episode_index = next_episode_index(dataset_root, reviewed_root)

    # 键盘读取器
    kb = KeyboardReader()

    # 状态
    # REVIEW_READY is intentionally separate: no more frames are written after s.
    STATE_IDLE, STATE_RECORDING, STATE_REVIEW_READY, STATE_REVIEWING = 'idle', 'recording', 'review_ready', 'reviewing'
    state = STATE_IDLE
    episode_buffer = []
    frame_count = 0
    start_time = None
    episode_dir = None

    try:
        logger.info("等待指令... (按 'c' 开始录制第 %d 条)\n", episode_index)
        logger.info("遥操已启动，可以移动主臂测试")

        # 主循环 - 遥操持续运行
        while True:
            loop_start = time.time()
            key = kb.get_key()

            # === 读取并发送遥操数据（一直运行）===
            try:
                action = teleop.get_action()
                observation = robot.get_observation()
                robot.send_action(action)
            except Exception as e:
                logger.error(f"遥操错误: {e}")
                time.sleep(0.1)
                continue

            # === 状态机处理 ===
            if state == STATE_IDLE:
                if key == 'c':
                    state = STATE_RECORDING
                    episode_buffer = []
                    frame_count = 0
                    start_time = time.time()

                    # 创建唯一临时目录。exist_ok=False 防止把遗留录制追加到新 episode。
                    episode_index = next_episode_index(dataset_root, reviewed_root)
                    episode_dir = dataset_root / f"episode_{episode_index:06d}_temp"
                    episode_dir.mkdir(exist_ok=False)

                    logger.info("\n[STEP 1/4] RECORDING Episode %d. Keep 0.3–0.5s initial stillness; end after release + 0.5–1.0s final stillness; press s.", episode_index)

                elif key == 'q':
                    logger.info("\n退出程序...")
                    break

            elif state == STATE_RECORDING:
                # 录制状态 - 保存当前帧
                frame_data = {
                    'observation': observation,
                    'action': action,
                    'timestamp': time.time() - start_time,
                }
                save_frame_to_disk(frame_data, episode_dir, frame_count)
                frame_count += 1

                # 每秒打印
                if frame_count % cfg.dataset.fps == 0:
                    logger.info("  录制中... %d 帧 (%.1f 秒)", frame_count, time.time() - start_time)

                if key == 's':
                    state = STATE_REVIEW_READY
                    duration = time.time() - start_time
                    logger.info("\n[STEP 2/4] RECORDING FROZEN - %d frames, %.1fs. You may now return to initial position; it is outside this episode.", frame_count, duration)
                    trim_summary(load_raw_episode(episode_dir), suggest_trim_range(load_raw_episode(episode_dir), review_config))

            elif state == STATE_REVIEW_READY:
                if key == 'g':
                    if review_config.get("recording", {}).get("auto_trim_on_accept", True):
                        # g is the operator's acceptance. Only then apply the
                        # conservative suggestion and atomically publish it.
                        try:
                            frames = load_raw_episode(episode_dir)
                            auto = suggest_trim_range(frames, review_config)
                            final_dir = commit_episode(
                                episode_dir, reviewed_root, auto["start"], auto["end"],
                                review_config, delete_source=True,
                                selected_index=episode_index, auto=auto,
                            )
                            logger.info("[STEP 3/4] ATOMIC COMMIT PASS: Episode %d → %s", episode_index, final_dir)
                            validation, replay = run_dry_run(final_dir, review_config)
                            manifest = __import__('json').loads((final_dir / 'review_manifest.json').read_text())
                            approved_total = len([p for p in reviewed_root.glob('episode_*') if p.is_dir() and (p / 'review_manifest.json').exists()])
                            print_demonstration_qa(episode_index, manifest, validation, replay, approved_total, reviewed_root)
                            episode_index = next_episode_index(dataset_root, reviewed_root)
                            state = STATE_IDLE
                            continue
                        except Exception as exc:
                            logger.error("自动裁剪提交失败；temp 已保留: %s", exc)
                            state = STATE_REVIEW_READY
                            continue
                    # Review owns g/b/q keys. Stop the terminal reader so its key
                    # cannot be consumed by both state machines.
                    state = STATE_REVIEWING
                    kb.stop()
                    logger.info("进入离线 Review；录制器已冻结，Review 不连接 CAN/Piper。")
                    from review_episode import review
                    result = review(episode_dir, output_root=None, allow_discard=True)
                    kb = KeyboardReader()
                    episode_index = get_next_episode_index(dataset_root)
                    state = STATE_IDLE
                    logger.info("Review 结果: %s\n等待指令... (按 'c' 开始录制第 %d 条)\n", result, episode_index)

                elif key == 'b':
                    logger.info("🗑  丢弃 Episode %d", episode_index)
                    discard_episode(episode_dir)

                    state = STATE_IDLE
                    episode_index = next_episode_index(dataset_root, reviewed_root)
                    logger.info("[DISCARDED] NEXT_ACTION = Return to initial position OUTSIDE RECORDING, prepare scene, then press c for Episode %d.\n", episode_index)

            # 控制帧率
            dt = 1.0 / cfg.dataset.fps
            elapsed = time.time() - loop_start
            if dt - elapsed > 0:
                precise_sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("\n\n收到中断信号...")

    finally:
        kb.stop()
        logger.info("正在断开连接...")

        try:
            robot.disconnect()
        except:
            pass

        try:
            teleop.disconnect()
        except:
            pass

        logger.info("✓ 已退出")
        logger.info("✓ 共录制 %d 条数据", episode_index)
        logger.info("✓ 数据位置: %s", dataset_root)


if __name__ == "__main__":
    main()
