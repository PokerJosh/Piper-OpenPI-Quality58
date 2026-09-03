#!/usr/bin/env python3
"""
教师端 7 路飞特 STS 舵机交互式零位校准。

用法:
  1. 手动将各关节摆到期望零位姿态（可逐个微调）
  2. 运行本脚本，观察实时位置
  3. 按 Enter：将当前位置写入 EEPROM 零位（校准后读数应接近 2047）
  4. 断电再上电后，用 --verify 检查零位是否保留

单关节: 按 1-7 只校准对应 ID；a 校准全部；v 刷新验证读数；q 退出
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _feetech_sdk_path import KIT_ROOT, add_sdk_to_path

add_sdk_to_path()
from scservo_sdk import *  # noqa: E402

CENTER_POSITION = 2047
DEFAULT_IDS = list(range(1, 8))
CALIBRATION_LOG = KIT_ROOT / "calibration" / "leader_zero_log.json"
CALIBRATION_RESULT = KIT_ROOT / "calibration" / "leader_zero_result.json"


def encode_offset(homing_offset: int) -> int:
    if homing_offset < 0:
        return (1 << 11) | abs(homing_offset)
    return homing_offset


def decode_offset(encoded: int) -> int:
    if encoded & (1 << 11):
        return -(encoded & 0x7FF)
    return encoded & 0x7FF


def read_present_position(packet_handler: sms_sts, scs_id: int) -> tuple[int | None, str | None]:
    raw_pos, result, error = packet_handler.read2ByteTxRx(scs_id, SMS_STS_PRESENT_POSITION_L)
    if result != COMM_SUCCESS:
        return None, packet_handler.getTxRxResult(result)
    if error != 0:
        return None, packet_handler.getRxPacketError(error)
    return int(raw_pos), None


def read_eeprom_offset(packet_handler: sms_sts, scs_id: int) -> tuple[int | None, str | None]:
    encoded, result, error = packet_handler.read2ByteTxRx(scs_id, SMS_STS_OFS_L)
    if result != COMM_SUCCESS:
        return None, packet_handler.getTxRxResult(result)
    if error != 0:
        return None, packet_handler.getRxPacketError(error)
    return int(encoded), None


def write_zero_at_present(
    packet_handler: sms_sts,
    scs_id: int,
    *,
    dry_run: bool = False,
    tolerance: int = 5,
    method: str = "offset",
) -> dict:
    """将当前物理位置设为零位（校准后位置读数 -> 2047）。"""
    record: dict = {"id": scs_id, "ok": False, "method": method}

    raw_before, err = read_present_position(packet_handler, scs_id)
    if err:
        record["error"] = err
        return record
    record["raw_before"] = raw_before

    if dry_run:
        record["homing_offset"] = int(raw_before) - CENTER_POSITION
        record["encoded_offset"] = encode_offset(record["homing_offset"])
        record["ok"] = True
        record["dry_run"] = True
        return record

    if method == "inst":
        return _calibrate_with_inst(packet_handler, scs_id, record, tolerance=tolerance)

    packet_handler.unLockEprom(scs_id)
    time.sleep(0.1)

    comm, error = packet_handler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, 0)
    if comm != COMM_SUCCESS:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getTxRxResult(comm)
        return record
    if error != 0:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getRxPacketError(error)
        return record
    time.sleep(0.15)

    raw_after_clear, err = read_present_position(packet_handler, scs_id)
    if err:
        packet_handler.LockEprom(scs_id)
        record["error"] = err
        return record
    record["raw_after_clear_ofs"] = raw_after_clear

    # 必须在偏移清零后读取位置，再计算新偏移（与官方示例一致）
    homing_offset = int(raw_after_clear) - CENTER_POSITION
    encoded = encode_offset(homing_offset)
    record["homing_offset"] = homing_offset
    record["encoded_offset"] = encoded

    comm, error = packet_handler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, encoded)
    if comm != COMM_SUCCESS:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getTxRxResult(comm)
        return record
    if error != 0:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getRxPacketError(error)
        return record
    time.sleep(0.15)

    packet_handler.LockEprom(scs_id)
    time.sleep(0.1)

    raw_after, err = read_present_position(packet_handler, scs_id)
    if err:
        record["error"] = err
        return record
    record["raw_after"] = raw_after
    record["delta_from_center"] = int(raw_after) - CENTER_POSITION

    stored, err = read_eeprom_offset(packet_handler, scs_id)
    if err:
        record["error"] = err
        return record
    record["eeprom_encoded"] = stored
    record["eeprom_decoded"] = decode_offset(int(stored))
    record["ok"] = abs(record["delta_from_center"]) <= tolerance
    if not record["ok"] and "error" not in record:
        record["error"] = (
            f"校准后读数={raw_after}，偏离中心 Δ={record['delta_from_center']:+d} "
            f"(允许±{tolerance})"
        )
    return record


def _calibrate_with_inst(
    packet_handler: sms_sts, scs_id: int, record: dict, *, tolerance: int
) -> dict:
    """使用舵机内置 OFSCAL 指令，将当前位置设为目标读数。"""
    packet_handler.unLockEprom(scs_id)
    time.sleep(0.1)
    comm, error = packet_handler.reOfsCal(scs_id, CENTER_POSITION)
    if comm != COMM_SUCCESS:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getTxRxResult(comm)
        return record
    if error != 0:
        packet_handler.LockEprom(scs_id)
        record["error"] = packet_handler.getRxPacketError(error)
        return record
    time.sleep(0.15)
    packet_handler.LockEprom(scs_id)
    time.sleep(0.1)

    raw_after, err = read_present_position(packet_handler, scs_id)
    if err:
        record["error"] = err
        return record
    record["raw_after"] = raw_after
    record["delta_from_center"] = int(raw_after) - CENTER_POSITION
    stored, err = read_eeprom_offset(packet_handler, scs_id)
    if not err:
        record["eeprom_encoded"] = stored
        record["eeprom_decoded"] = decode_offset(int(stored))
    record["ok"] = abs(record["delta_from_center"]) <= tolerance
    if not record["ok"] and "error" not in record:
        record["error"] = (
            f"OFSCAL 后读数={raw_after}，偏离中心 Δ={record['delta_from_center']:+d} "
            f"(允许±{tolerance})"
        )
    return record


def read_all_status(packet_handler: sms_sts, servo_ids: list[int]) -> list[dict]:
    rows = []
    for scs_id in servo_ids:
        row: dict = {"id": scs_id}
        pos, err = read_present_position(packet_handler, scs_id)
        if err:
            row["error"] = err
        else:
            row["present"] = pos
            row["delta_from_center"] = int(pos) - CENTER_POSITION
        ofs, err2 = read_eeprom_offset(packet_handler, scs_id)
        if err2:
            row["offset_error"] = err2
        else:
            row["eeprom_encoded"] = ofs
            row["eeprom_decoded"] = decode_offset(int(ofs))
        rows.append(row)
    return rows


def format_status_line(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        sid = r["id"]
        if "error" in r:
            parts.append(f"J{sid}:ERR")
            continue
        pos = r["present"]
        d = r["delta_from_center"]
        parts.append(f"J{sid}:{pos:4d}(Δ{d:+4d})")
    return "  ".join(parts)


def save_log(entries: list[dict], *, mode: str) -> None:
    payload = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "center_position": CENTER_POSITION,
        "servos": entries,
    }
    history: list = []
    if CALIBRATION_LOG.exists():
        try:
            history = json.loads(CALIBRATION_LOG.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            history = []
    history.append(payload)
    CALIBRATION_LOG.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n已写入日志: {CALIBRATION_LOG}")


def open_bus(port: str, baud: int) -> tuple[PortHandler, sms_sts]:
    port_handler = PortHandler(port)
    packet_handler = sms_sts(port_handler)
    if not port_handler.openPort():
        raise RuntimeError(f"无法打开串口 {port}")
    if not port_handler.setBaudRate(baud):
        port_handler.closePort()
        raise RuntimeError(f"无法设置波特率 {baud}")
    return port_handler, packet_handler


def run_verify(packet_handler: sms_sts, servo_ids: list[int], *, tolerance: int = 5) -> int:
    print("=== 零位验证（只读 EEPROM 偏移 + 当前位置）===")
    print(f"期望 present ≈ {CENTER_POSITION}（±{tolerance} 步视为正常）\n")
    rows = read_all_status(packet_handler, servo_ids)
    ok_count = 0
    for r in rows:
        sid = r["id"]
        if "error" in r:
            print(f"  ID {sid}: 读位置失败 — {r['error']}")
            continue
        pos = r["present"]
        d = r["delta_from_center"]
        ofs_dec = r.get("eeprom_decoded", "?")
        status = "OK" if abs(d) <= tolerance else "偏离零位"
        if abs(d) <= tolerance:
            ok_count += 1
        print(
            f"  ID {sid}: present={pos:4d}  Δ={d:+4d}  "
            f"EEPROM偏移={ofs_dec}  [{status}]"
        )
    save_log(rows, mode="verify")
    print(f"\n{ok_count}/{len(servo_ids)} 个舵机在读数上接近零位。")
    return 0 if ok_count == len(servo_ids) else 1


def run_interactive(
    packet_handler: sms_sts,
    servo_ids: list[int],
    *,
    dry_run: bool,
    tolerance: int,
    method: str,
) -> int:
    print("=== 交互式零位校准（教师端 7 舵机）===")
    print(f"串口已连接。零位目标读数 = {CENTER_POSITION}")
    print("操作: [Enter]=全部校准  [1-7]=单关节  [v]=打印验证  [q]=退出")
    if dry_run:
        print("*** dry-run 模式：不会写入 EEPROM ***")
    print("请手动将机械臂摆到期望零位后操作。\n")

    stdin_fd = sys.stdin.fileno()
    old_settings = None
    try:
        import termios
        import tty

        old_settings = termios.tcgetattr(stdin_fd)
    except ImportError:
        termios = None
        tty = None

    def read_key(timeout: float) -> str | None:
        if termios is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            return "enter"
        return ch.lower()

    try:
        if termios is not None:
            tty.setcbreak(stdin_fd)

        while True:
            rows = read_all_status(packet_handler, servo_ids)
            line = format_status_line(rows)
            print(f"\r{line}    ", end="", flush=True)

            key = read_key(0.15)
            if key is None:
                continue

            if key == "q":
                print("\n退出。")
                break
            if key == "v":
                print()
                run_verify(packet_handler, servo_ids, tolerance=tolerance)
                continue
            if key == "enter":
                target_ids = servo_ids
            elif key in "1234567":
                target_ids = [int(key)]
            elif key == "a":
                target_ids = servo_ids
            else:
                continue

            print(f"\n>>> 校准 ID {target_ids} ...")
            results = []
            for scs_id in target_ids:
                rec = write_zero_at_present(
                    packet_handler,
                    scs_id,
                    dry_run=dry_run,
                    tolerance=tolerance,
                    method=method,
                )
                results.append(rec)
                if rec.get("ok"):
                    print(
                        f"  ID {scs_id}: OK  "
                        f"清零后={rec.get('raw_after_clear_ofs', '?')} -> "
                        f"校准后={rec.get('raw_after')} "
                        f"(Δ{rec.get('delta_from_center', 0):+d})"
                    )
                else:
                    print(f"  ID {scs_id}: 失败 — {rec.get('error', '未知错误')}")

            save_log(results, mode="calibrate_dry_run" if dry_run else "calibrate")
            print("校准完成。可断电上电后执行: python3 feetech_servo_zero.py --verify")
            time.sleep(0.5)
    finally:
        if old_settings is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="教师端飞特舵机零位校准")
    parser.add_argument("--port", default="<LEADER_SERIAL_PORT>")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument(
        "--ids",
        default="1-7",
        help="舵机 ID，如 1-7 或 1,2,3",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="仅验证零位（断电重启后使用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算偏移，不写入 EEPROM",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="校准后读数与 2047 的允许偏差（默认 5）",
    )
    parser.add_argument(
        "--method",
        choices=("offset", "inst"),
        default="offset",
        help="offset=写 EEPROM 偏移（默认）；inst=舵机 OFSCAL 指令",
    )
    args = parser.parse_args()

    if "-" in args.ids:
        lo, hi = args.ids.split("-", 1)
        servo_ids = list(range(int(lo), int(hi) + 1))
    else:
        servo_ids = [int(x) for x in args.ids.split(",")]

    port_handler = None
    try:
        port_handler, packet_handler = open_bus(args.port, args.baud)
        if args.verify:
            return run_verify(packet_handler, servo_ids, tolerance=args.tolerance)
        return run_interactive(
            packet_handler,
            servo_ids,
            dry_run=args.dry_run,
            tolerance=args.tolerance,
            method=args.method,
        )
    except RuntimeError as exc:
        print(f"错误: {exc}")
        return 1
    finally:
        if port_handler is not None:
            port_handler.closePort()


if __name__ == "__main__":
    sys.exit(main())
