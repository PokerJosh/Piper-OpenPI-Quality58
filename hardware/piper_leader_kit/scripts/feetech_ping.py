#!/usr/bin/env python3
"""只 ping 舵机，不改 ID。用于标号后日常检查。"""
import argparse
import sys

from _feetech_sdk_path import add_sdk_to_path

add_sdk_to_path()
from scservo_sdk import *  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ping 飞特 STS/SMS 舵机")
    parser.add_argument("--port", default="<LEADER_SERIAL_PORT>")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--id", type=int, default=None, help="只 ping 指定 ID")
    parser.add_argument("--scan", default="1-20", help="未指定 --id 时扫描范围")
    args = parser.parse_args()

    port_handler = PortHandler(args.port)
    packet_handler = sms_sts(port_handler)

    if not port_handler.openPort():
        print("失败: 无法打开串口")
        sys.exit(1)
    if not port_handler.setBaudRate(args.baud):
        print("失败: 无法设置波特率")
        port_handler.closePort()
        sys.exit(1)

    ids = [args.id] if args.id is not None else []
    if not ids:
        if "-" in args.scan:
            lo, hi = args.scan.split("-", 1)
            ids = list(range(int(lo), int(hi) + 1))
        else:
            ids = [int(x) for x in args.scan.split(",")]

    print(f"串口 {args.port}, 波特率 {args.baud}\n")
    found = 0
    try:
        for sid in ids:
            model, result, error = packet_handler.ping(sid)
            if result == COMM_SUCCESS:
                print(f"[ID:{sid:03d}] OK  model={model}")
                found += 1
            else:
                print(f"[ID:{sid:03d}] --  {packet_handler.getTxRxResult(result)}")
    finally:
        port_handler.closePort()

    print(f"\n共发现 {found} 个舵机")
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
