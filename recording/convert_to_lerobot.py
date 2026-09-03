#!/usr/bin/env python3
"""
将自定义录制脚本保存的原始 pickle 数据转换为 LeRobot 标准格式

用法:
    python convert_to_lerobot.py

转换后的数据可以直接用于 lerobot-train 训练
"""

import sys
import argparse
import logging
from pathlib import Path
import pickle
import shutil
import yaml
import draccus
from tqdm import tqdm
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 注册插件
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import RecordConfig
from episode_tools import list_episode_frames, load_config as load_review_config, validate_episode


# 关节/夹爪状态顺序 —— 显式写死，绝不依赖 dict 迭代顺序或 sorted()
JOINT_KEYS = [
    "J1.pos",
    "J2.pos",
    "J3.pos",
    "J4.pos",
    "J5.pos",
    "J6.pos",
    "G.pos",
]

# 原始 pickle 里的相机 key -> LeRobot 标准 observation.images.* key
CAMERA_KEY_MAP = {
    "wrist": "observation.images.wrist",
    "top": "observation.images.top",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", help="Reviewed/raw input root. Defaults to the historic local_piper_demo root.")
    parser.add_argument("--repo-id", help="New target LeRobot repo id, e.g. local/piper_demo_trimmed_v2.")
    args = parser.parse_args()
    # 加载配置
    config_path = Path("piper_camera_config.yaml")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = draccus.decode(RecordConfig, cfg_dict)

    # 原始数据目录
    raw_data_root = Path(args.input_root) if args.input_root else Path.home() / ".cache/huggingface/lerobot/local_piper_demo"

    if not raw_data_root.exists():
        logger.error(f"原始数据目录不存在: {raw_data_root}")
        logger.error(f"请先使用 custom_record_final.py 录制数据")
        sys.exit(1)

    # 查找所有 episode 目录
    episode_dirs = sorted([d for d in raw_data_root.glob("episode_*") if d.is_dir() and not d.name.endswith(('_temp', '_commit_tmp'))])

    if not episode_dirs:
        logger.error(f"没有找到 episode 数据: {raw_data_root}")
        sys.exit(1)

    logger.info(f"找到 {len(episode_dirs)} 个 episodes 需要转换")
    logger.info(f"原始数据: {raw_data_root}")

    # 目标数据集名称 —— 每次运行都用全新 timestamp，绝不 append 到旧的半成品数据集
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_repo_id = args.repo_id or f"local/piper_demo_{timestamp}"
    if "/" not in dataset_repo_id:
        logger.error("--repo-id must include namespace/name, for example local/piper_demo_trimmed_v2")
        sys.exit(1)
    dataset_name = dataset_repo_id.split("/", 1)[1]

    logger.info(f"目标数据集: {dataset_repo_id}")
    logger.info("")

    # 加载第一帧来获取 features 结构
    logger.info("正在分析数据结构...")
    first_episode = episode_dirs[0]
    first_frame_file = list_episode_frames(first_episode)[0]

    with open(first_frame_file, 'rb') as f:
        first_frame = pickle.load(f)

    observation = first_frame['observation']
    action = first_frame['action']

    # 校验：JOINT_KEYS 必须完整存在于 observation 和 action 中
    missing_obs_keys = [k for k in JOINT_KEYS if k not in observation]
    missing_act_keys = [k for k in JOINT_KEYS if k not in action]
    if missing_obs_keys:
        logger.error(f"observation 中缺少关节 key: {missing_obs_keys}")
        sys.exit(1)
    if missing_act_keys:
        logger.error(f"action 中缺少关节 key: {missing_act_keys}")
        sys.exit(1)

    # 校验：相机 key 必须存在
    missing_cam_keys = [k for k in CAMERA_KEY_MAP if k not in observation]
    if missing_cam_keys:
        logger.error(f"observation 中缺少相机 key: {missing_cam_keys}")
        sys.exit(1)

    # 构建 features 字典
    features = {}

    # observation.state: 固定 7 维，顺序 = JOINT_KEYS
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(JOINT_KEYS)],
        "names": list(JOINT_KEYS),
    }

    # action: 固定 7 维，顺序 = JOINT_KEYS
    features["action"] = {
        "dtype": "float32",
        "shape": [len(JOINT_KEYS)],
        "names": list(JOINT_KEYS),
    }

    # 相机 features: observation.images.wrist / observation.images.top
    for raw_key, feat_key in CAMERA_KEY_MAP.items():
        value = observation[raw_key]
        if not isinstance(value, np.ndarray):
            logger.error(f"相机 key '{raw_key}' 不是 ndarray，实际类型: {type(value)}")
            sys.exit(1)
        features[feat_key] = {
            "dtype": "video",
            "shape": list(value.shape),
            "names": ["height", "width", "channels"],
        }

    logger.info(f"✓ 数据结构分析完成")
    logger.info(f"  Observation keys (raw): {list(observation.keys())}")
    logger.info(f"  Action keys (raw): {list(action.keys())}")
    logger.info("")
    logger.info("最终 features schema:")
    print(features)
    logger.info("")

    # 确认关键字段存在
    assert features["observation.state"]["shape"] == [7], "observation.state shape 必须是 [7]"
    assert features["action"]["shape"] == [7], "action shape 必须是 [7]"
    assert "observation.images.wrist" in features, "缺少 observation.images.wrist"
    assert "observation.images.top" in features, "缺少 observation.images.top"

    # 创建 LeRobot 数据集
    logger.info("正在创建数据集...")

    # 使用完整路径，包含数据集名称（全新目录，不会与旧的半成品冲突）
    dataset_path = Path.home() / ".cache/huggingface/lerobot/local" / dataset_name

    if dataset_path.exists():
        logger.error(f"目标目录已存在，为避免 append 到半成品数据: {dataset_path}")
        sys.exit(1)

    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=cfg.dataset.fps,
        features=features,
        robot_type="piper",
        root=dataset_path,  # 直接指定完整路径
        use_videos=True,
        streaming_encoding=False,
    )

    logger.info("✓ 数据集已创建\n")

    # 转换所有 episodes
    success_count = 0

    for episode_dir in episode_dirs:
        episode_num = int(episode_dir.name.split('_')[1])
        logger.info(f"转换 Episode {episode_num}...")

        # 加载所有帧
        validation = validate_episode(episode_dir, cfg.dataset.fps)
        if not validation["valid"]:
            logger.warning("  跳过: 数据质量检查失败: %s", validation["errors"])
            continue
        manifest = episode_dir / "review_manifest.json"
        if manifest.exists():
            review = __import__('json').loads(manifest.read_text())
            logger.info("  Review: original=%s trimmed=%s range=%s..%s status=%s", review.get('original_frame_count'), review.get('saved_frame_count'), review.get('trim_start_original'), review.get('trim_end_original'), review.get('review_status'))
            if review.get("review_status") != "approved":
                logger.warning("  跳过: review_status 不是 approved")
                continue
        frame_files = list_episode_frames(episode_dir)
        if not frame_files:
            logger.warning(f"  跳过: 没有数据帧")
            continue

        logger.info(f"  加载 {len(frame_files)} 帧...")

        for frame_file in tqdm(frame_files, desc="  处理帧", leave=False):
            with open(frame_file, 'rb') as f:
                frame_data = pickle.load(f)

            obs = frame_data['observation']
            act = frame_data['action']

            # 严格按 JOINT_KEYS 顺序构建 7 维 state / action，绝不用 dict 迭代或 sorted()
            state_arr = np.asarray([obs[k] for k in JOINT_KEYS], dtype=np.float32)
            action_arr = np.asarray([act[k] for k in JOINT_KEYS], dtype=np.float32)

            frame = {
                "observation.state": state_arr,
                "action": action_arr,
                "observation.images.wrist": obs["wrist"],
                "observation.images.top": obs["top"],
                "task": cfg.dataset.single_task if hasattr(cfg.dataset, 'single_task') else "pick and place",
            }

            # 添加帧到数据集
            dataset.add_frame(frame)

        logger.info(f"  保存并编码视频... (这可能需要1-2分钟)")

        # 保存 episode（不需要传 episode_index，内部会自动管理）
        dataset.save_episode(episode_data=None, parallel_encoding=True)

        logger.info(f"  ✓ Episode {episode_num} → Episode {success_count} 转换完成\n")
        success_count += 1

    logger.info("="*60)
    logger.info(f"转换完成！")
    logger.info(f"  成功转换: {success_count}/{len(episode_dirs)} episodes")
    logger.info(f"  数据集位置: {dataset.root}")
    logger.info(f"  数据集 ID: {dataset_repo_id}")
    logger.info("="*60)
    logger.info("")
    logger.info("现在可以使用以下命令训练:")
    logger.info(f"  lerobot-train --policy.type=act --dataset.repo_id={dataset_repo_id} --robot.type=piper")
    logger.info("")

    # 输出机器可解析的结果摘要，供后续验证脚本使用
    print("DATASET_ROOT=" + str(dataset.root))
    print("DATASET_REPO_ID=" + dataset_repo_id)
    print("EPISODES_CONVERTED=" + str(success_count))
    print("EPISODES_TOTAL=" + str(len(episode_dirs)))


if __name__ == "__main__":
    main()
