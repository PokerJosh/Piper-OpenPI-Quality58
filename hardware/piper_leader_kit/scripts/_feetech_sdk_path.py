"""本配套包内 scservo_sdk 路径（所有脚本统一引用）。"""
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2] / "piper_leader_kit"
SDK_DIR = KIT_ROOT / "sdk"


def add_sdk_to_path() -> Path:
    import sys

    if not (SDK_DIR / "scservo_sdk" / "__init__.py").is_file():
        raise RuntimeError(f"未找到 scservo_sdk，请检查目录: {SDK_DIR}")
    sys.path.insert(0, str(SDK_DIR))
    return SDK_DIR
