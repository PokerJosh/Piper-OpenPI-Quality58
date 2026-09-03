#!/usr/bin/env python3
"""单舵机改 ID（一次只接一颗舵机到总线）。"""
from __future__ import annotations

import argparse
import sys
import time

from _feetech_sdk_path import add_sdk_to_path

add_sdk_to_path()
from scservo_sdk import *  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="飞特 STS 舵机改 ID")
    parser.add_argument("--port", default="<LEADER_SERIAL_PORT>")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--new-id", type=int, required=True, help="新 ID，教师臂一般为 1~7")
    args = parser.parse_args()

    if not 1 <= args.new_id <= 253:
        print("new-id 应在 1~253")
        return 1

    port_handler = PortHandler(args.port)
    packet_handler = sms_sts(port_handler)
    if not port_handler.openPort():
        print(f"无法打开串口 {args.port}")
        return 1
    if not port_handler.setBaudRate(args.baud):
        print(f"无法设置波特率 {args.baud}")
        port_handler.closePort()
        return 1

    print(f"将把当前总线上唯一舵机的 ID 改为 {args.new_id}（请确认只连接一颗）")
    input("按 Enter 继续...")

    packet_handler.unLockEprom(BROADCAST_ID)
    time.sleep(0.1)
    result, error = packet_handler.write1ByteTxRx(BROADCAST_ID, SMS_STS_ID, args.new_id)
    if result != COMM_SUCCESS:
        print(f"失败: {packet_handler.getTxRxResult(result)}")
        packet_handler.LockEprom(BROADCAST_ID)
        port_handler.closePort()
        return 1
    if error != 0:
        print(f"失败: error={error}")
        packet_handler.LockEprom(BROADCAST_ID)
        port_handler.closePort()
        return 1

    time.sleep(0.1)
    packet_handler.LockEprom(args.new_id)
    time.sleep(0.1)
    port_handler.closePort()
    print(f"成功，新 ID = {args.new_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
