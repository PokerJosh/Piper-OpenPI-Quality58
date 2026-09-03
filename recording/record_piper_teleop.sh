#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Record Piper leader teleoperation data with LeRobot.

Usage:
  bash piper_upper_pc/record_piper_teleop.sh [options] [-- extra lerobot-record args...]

Common options:
  --leader-kit PATH       piper_leader_kit path
  --teleop-port PATH      Feetech leader serial port, e.g. <LEADER_SERIAL_PORT>
  --can NAME              Piper CAN interface, e.g. can0
  --repo-id ID            Dataset repo id, e.g. your_hf_name/piper_teleop_demo
  --task TEXT             Single task description
  --episodes N            Number of episodes
  --fps N                 Dataset/control FPS
  --push-to-hub BOOL      true or false
  --display BOOL          true or false
  --setup-can             Run sudo ip link setup for the CAN interface
  --dry-run               Print the command but do not execute it

Environment overrides:
  LEADER_KIT_PATH, TELEOP_PORT, CAN_NAME, DATASET_REPO_ID, SINGLE_TASK,
  NUM_EPISODES, FPS, PUSH_TO_HUB, DISPLAY_DATA, ENABLE_MOTION,
  REQUIRE_ALL_SERVOS, MAX_TICK_JUMP, MAX_RELATIVE_TARGET_DEG,
  MAX_JOINT_SPEED_DEG_S, MOTION_COMMAND_DURATION_S, DATASET_ROOT,
  PRIVATE_DATASET, CAN_BITRATE
EOF
}

bool_value() {
  case "$1" in
    true|false) printf '%s' "$1" ;;
    1|yes|y|on) printf 'true' ;;
    0|no|n|off) printf 'false' ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 2
      ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEROBOT_V2_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

LEADER_KIT_PATH="${LEADER_KIT_PATH:-$HOME/csy/piper_leader_kit}"
TELEOP_PORT="${TELEOP_PORT:-<LEADER_SERIAL_PORT>}"
CAN_NAME="${CAN_NAME:-<CAN_INTERFACE>}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER:-local}/piper_teleop_demo}"
SINGLE_TASK="${SINGLE_TASK:-Describe this task in one sentence}"
NUM_EPISODES="${NUM_EPISODES:-5}"
FPS="${FPS:-30}"
DISPLAY_DATA="$(bool_value "${DISPLAY_DATA:-false}")"
ENABLE_MOTION="$(bool_value "${ENABLE_MOTION:-true}")"
REQUIRE_ALL_SERVOS="$(bool_value "${REQUIRE_ALL_SERVOS:-false}")"
MAX_TICK_JUMP="${MAX_TICK_JUMP:-0}"
MAX_RELATIVE_TARGET_DEG="${MAX_RELATIVE_TARGET_DEG:-30.0}"
MAX_JOINT_SPEED_DEG_S="${MAX_JOINT_SPEED_DEG_S:-50.0}"
MOTION_COMMAND_DURATION_S="${MOTION_COMMAND_DURATION_S:-0.02}"
DATASET_ROOT="${DATASET_ROOT:-}"
PRIVATE_DATASET="$(bool_value "${PRIVATE_DATASET:-false}")"

PUSH_TO_HUB_EXPLICIT=false
if [[ -n "${PUSH_TO_HUB:-}" ]]; then
  PUSH_TO_HUB="$(bool_value "$PUSH_TO_HUB")"
  PUSH_TO_HUB_EXPLICIT=true
elif [[ "$DATASET_REPO_ID" == local/* ]]; then
  PUSH_TO_HUB="false"
else
  PUSH_TO_HUB="true"
fi

SETUP_CAN=false
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --leader-kit)
      LEADER_KIT_PATH="$2"
      shift 2
      ;;
    --teleop-port)
      TELEOP_PORT="$2"
      shift 2
      ;;
    --can)
      CAN_NAME="$2"
      shift 2
      ;;
    --repo-id)
      DATASET_REPO_ID="$2"
      if ! $PUSH_TO_HUB_EXPLICIT; then
        if [[ "$DATASET_REPO_ID" == local/* ]]; then
          PUSH_TO_HUB=false
        else
          PUSH_TO_HUB=true
        fi
      fi
      shift 2
      ;;
    --task)
      SINGLE_TASK="$2"
      shift 2
      ;;
    --episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --push-to-hub)
      PUSH_TO_HUB="$(bool_value "$2")"
      PUSH_TO_HUB_EXPLICIT=true
      shift 2
      ;;
    --display)
      DISPLAY_DATA="$(bool_value "$2")"
      shift 2
      ;;
    --setup-can)
      SETUP_CAN=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
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

if [[ ! -d "$ROOT/lerobot" || ! -d "$ROOT/external" ]]; then
  echo "Cannot find lerobot_v2 workspace at: $ROOT" >&2
  exit 1
fi

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "lerobot-record is not on PATH. Activate the environment and run install_piper_upper_pc.sh first." >&2
  exit 1
fi

if [[ ! -d "$LEADER_KIT_PATH" ]]; then
  echo "leader kit path does not exist: $LEADER_KIT_PATH" >&2
  exit 1
fi

if [[ ! -f "$LEADER_KIT_PATH/config/leader_to_piper_mapping.example.json" ]]; then
  echo "mapping json is missing: $LEADER_KIT_PATH/config/leader_to_piper_mapping.example.json" >&2
  exit 1
fi

if [[ ! -e "$TELEOP_PORT" ]]; then
  echo "teleop serial port does not exist: $TELEOP_PORT" >&2
  echo "Check with: ls <LEADER_SERIAL_PORT> /dev/ttyUSB*" >&2
  exit 1
fi

if $SETUP_CAN; then
  sudo ip link set "$CAN_NAME" down >/dev/null 2>&1 || true
  sudo ip link set "$CAN_NAME" up type can bitrate "$CAN_BITRATE"
fi

if ! ip link show "$CAN_NAME" >/dev/null 2>&1; then
  echo "CAN interface does not exist: $CAN_NAME" >&2
  echo "Check with: ip link | grep can" >&2
  exit 1
fi

python - <<'PY'
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
if "piper" not in RobotConfig.get_known_choices():
    raise SystemExit("robot type 'piper' is not registered")
if "piper_leader" not in TeleoperatorConfig.get_known_choices():
    raise SystemExit("teleop type 'piper_leader' is not registered")
PY

cmd=(
  lerobot-record
  "--teleop.type=piper_leader"
  "--teleop.port=$TELEOP_PORT"
  "--teleop.leader_kit_path=$LEADER_KIT_PATH"
  "--teleop.require_all_servos=$REQUIRE_ALL_SERVOS"
  "--teleop.max_tick_jump=$MAX_TICK_JUMP"
  "--robot.type=piper"
  "--robot.can_name=$CAN_NAME"
  "--robot.enable_motion=$ENABLE_MOTION"
  "--robot.max_relative_target_deg=$MAX_RELATIVE_TARGET_DEG"
  "--robot.max_joint_speed_deg_s=$MAX_JOINT_SPEED_DEG_S"
  "--robot.motion_command_duration_s=$MOTION_COMMAND_DURATION_S"
  "--robot.cameras.wrist.type=intelrealsense"
  "--robot.cameras.wrist.serial_number=<WRIST_CAMERA_SERIAL>"
  "--robot.cameras.wrist.fps=30"
  "--robot.cameras.wrist.width=640"
  "--robot.cameras.wrist.height=480"
  "--robot.cameras.top.type=opencv"
  "--robot.cameras.top.index=0"
  "--robot.cameras.top.fps=30"
  "--robot.cameras.top.width=640"
  "--robot.cameras.top.height=480"
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.num_episodes=$NUM_EPISODES"
  "--dataset.single_task=$SINGLE_TASK"
  "--dataset.fps=$FPS"
  "--dataset.push_to_hub=$PUSH_TO_HUB"
  "--dataset.private=$PRIVATE_DATASET"
  "--display_data=$DISPLAY_DATA"
)

if [[ -n "$DATASET_ROOT" ]]; then
  cmd+=("--dataset.root=$DATASET_ROOT")
fi

cmd+=("${EXTRA_ARGS[@]}")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if $DRY_RUN; then
  exit 0
fi

exec "${cmd[@]}"
