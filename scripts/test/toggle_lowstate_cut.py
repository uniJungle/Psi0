#!/usr/bin/env python3
"""Simulate LowState loss for SONIC deploy (run as root).

Unitree G1 has two hosts on the robot LAN:
  PC1 192.168.123.161 — motion / publishes rt/lowstate + takes rt/lowcmd (DDS)
  PC2 192.168.123.164 — Orin (SSH, cameras, brainco_hand, …)

Dropping only .164 does NOTHING to LowState. Default cuts UDP from PC1 .161
(SSH to .164 stays up).

  sudo python3 scripts/test/toggle_lowstate_cut.py
  sudo python3 scripts/test/toggle_lowstate_cut.py --diagnose
  sudo python3 scripts/test/toggle_lowstate_cut.py --src 192.168.123.0/24

Keys: s=toggle  q=quit. Hold the robot while CUT.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import termios
import tty


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"cmd failed ({r.returncode}): {' '.join(cmd)}\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )
    return r


def restore(iface: str) -> None:
    sh(["tc", "qdisc", "del", "dev", iface, "ingress"], check=False)
    # leftover raw rules from --mode raw
    sh(
        ["iptables", "-t", "raw", "-D", "PREROUTING", "-i", iface, "-j", "DROP"],
        check=False,
    )
    while True:
        r = sh(
            ["iptables", "-t", "raw", "-C", "PREROUTING", "-i", iface, "-j", "DROP"],
            check=False,
        )
        if r.returncode != 0:
            break
        sh(
            ["iptables", "-t", "raw", "-D", "PREROUTING", "-i", iface, "-j", "DROP"],
            check=False,
        )


def cut_tc_udp_subnet(iface: str, cidr: str) -> None:
    restore(iface)
    sh(["tc", "qdisc", "add", "dev", iface, "handle", "ffff:", "ingress"])
    sh(
        [
            "tc", "filter", "add", "dev", iface, "parent", "ffff:",
            "protocol", "ip", "prio", "1", "u32",
            "match", "ip", "src", cidr,
            "match", "ip", "protocol", "17", "0xff",
            "action", "drop",
        ]
    )


def cut_tc_all_subnet(iface: str, cidr: str) -> None:
    restore(iface)
    sh(["tc", "qdisc", "add", "dev", iface, "handle", "ffff:", "ingress"])
    sh(
        [
            "tc", "filter", "add", "dev", iface, "parent", "ffff:",
            "protocol", "ip", "prio", "1", "u32",
            "match", "ip", "src", cidr,
            "action", "drop",
        ]
    )


def cut_tc_iface(iface: str) -> None:
    """Drop every ingress IPv4 packet on this NIC (SSH via this NIC dies)."""
    restore(iface)
    sh(["tc", "qdisc", "add", "dev", iface, "handle", "ffff:", "ingress"])
    sh(
        [
            "tc", "filter", "add", "dev", iface, "parent", "ffff:",
            "protocol", "ip", "prio", "1", "u32",
            "match", "u32", "0", "0",
            "action", "drop",
        ]
    )


def cut_raw_iface(iface: str) -> None:
    """iptables raw PREROUTING — earliest netfilter hook."""
    restore(iface)
    sh(["iptables", "-t", "raw", "-I", "PREROUTING", "1", "-i", iface, "-j", "DROP"])


def show_status(iface: str) -> None:
    r = sh(["tc", "qdisc", "show", "dev", iface], check=False)
    print(f"  tc qdisc:\n{(r.stdout or '').rstrip()}")
    r2 = sh(["tc", "filter", "show", "dev", iface, "parent", "ffff:"], check=False)
    if (r2.stdout or "").strip():
        print(f"  tc filter:\n{(r2.stdout or '').rstrip()}")
    r3 = sh(["iptables", "-t", "raw", "-L", "PREROUTING", "-n", "-v"], check=False)
    if "DROP" in (r3.stdout or ""):
        print(f"  iptables raw PREROUTING:\n{(r3.stdout or '').rstrip()}")


def diagnose(iface: str, seconds: float) -> int:
    print(f"[diagnose] sniffing UDP on {iface} for {seconds}s ...")
    r = sh(
        [
            "timeout", str(seconds), "tcpdump", "-i", iface, "-nn", "-c", "300", "udp",
        ],
        check=False,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-4000:] if len(out) > 4000 else out)
    # Parse "IP a.b.c.d.port > ..."
    import re
    from collections import Counter

    srcs = Counter()
    for m in re.finditer(
        r"IP (\d+\.\d+\.\d+\.\d+)\.\d+ > (\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\d+)",
        out,
    ):
        srcs[m.group(1)] += 1
    print("[diagnose] UDP source IPs (use these with --src):")
    if not srcs:
        print("  (no UDP seen — is deploy running? wrong --iface?)")
        return 1
    for ip, n in srcs.most_common():
        print(f"  {ip}: {n} pkts")
    return 0


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iface", default="enp4s0")
    p.add_argument(
        "--src",
        default="192.168.123.161/32",
        help="IPv4 CIDR to drop on ingress (default: G1 PC1 / motion board)",
    )
    p.add_argument(
        "--mode",
        choices=["udp", "all", "iface", "raw"],
        default="udp",
        help="udp=UDP from --src; all=all IP from --src; iface=all ingress on NIC; raw=iptables raw DROP",
    )
    p.add_argument("--diagnose", action="store_true", help="Sniff UDP sources and exit")
    p.add_argument("--diagnose-sec", type=float, default=3.0)
    args = p.parse_args()

    if os.geteuid() != 0:
        print("[ERR] Need root:")
        print(f"  sudo python3 {sys.argv[0]} ...")
        return 1

    if args.diagnose:
        return diagnose(args.iface, args.diagnose_sec)

    restore(args.iface)
    cut_on = False
    print(f"[ready] iface={args.iface} src={args.src} mode={args.mode}")
    print("[ready] s=toggle  q=quit | tip: if udp fails, try --mode iface (SSH on this NIC will die)")
    print("[ready] first run --diagnose once to see real LowState source IPs")

    try:
        while True:
            k = read_key().lower()
            if k == "q":
                break
            if k != "s":
                continue
            if not cut_on:
                if args.mode == "udp":
                    cut_tc_udp_subnet(args.iface, args.src)
                elif args.mode == "all":
                    cut_tc_all_subnet(args.iface, args.src)
                elif args.mode == "iface":
                    cut_tc_iface(args.iface)
                else:
                    cut_raw_iface(args.iface)
                cut_on = True
                print("[CUT ] drop armed")
                show_status(args.iface)
            else:
                restore(args.iface)
                cut_on = False
                print("[ON  ] drop removed")
                show_status(args.iface)
    except KeyboardInterrupt:
        print()
    finally:
        restore(args.iface)
        print("[exit] cleaned")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
