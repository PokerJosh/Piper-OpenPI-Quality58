#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Install LeRobot and the Piper plugins from this <PIPER_PROJECT_ROOT> workspace.

Usage:
  bash piper_upper_pc/install_piper_upper_pc.sh [--root PATH]

Environment:
  LEROBOT_V2_ROOT   Workspace root. Defaults to the parent of piper_upper_pc.

Run this inside an activated Python 3.12 environment.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEROBOT_V2_ROOT:-/path/to/piper-project}"
OPENPI_ROOT="${OPENPI_ROOT:-/path/to/openpi}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$OPENPI_ROOT" || ! -d "$ROOT/hardware/lerobot_robot_piper" || ! -d "$ROOT/teleop/lerobot_teleoperator_piper_leader" ]]; then
  echo "Cannot find <PIPER_PROJECT_ROOT> workspace at: $ROOT" >&2
  echo "Expected: OPENPI_ROOT plus hardware/lerobot_robot_piper and teleop/lerobot_teleoperator_piper_leader" >&2
  exit 1
fi

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python >= 3.12 is required, current: {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip
python -m pip install -e "${OPENPI_ROOT}[core_scripts]"
python -m pip install -e "$ROOT/hardware/lerobot_robot_piper"
python -m pip install -e "$ROOT/teleop/lerobot_teleoperator_piper_leader"

python - <<'PY'
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
robots = RobotConfig.get_known_choices()
teleops = TeleoperatorConfig.get_known_choices()
missing = []
if "piper" not in robots:
    missing.append("robot:piper")
if "piper_leader" not in teleops:
    missing.append("teleop:piper_leader")
if missing:
    raise SystemExit("Plugin registration failed: " + ", ".join(missing))
print("Piper plugins registered: robot=piper, teleop=piper_leader")
PY

echo "Install complete."
