#!/usr/bin/env python3
"""Auto-enable controller for SONIC deploy (zmq_manager mode).

Sends ZMQ commands so the robot:
  1. Enters PLANNER mode with start=True
  2. Receives idle planner commands to stay upright

Default is **handoff** mode: stand for a few seconds, then release the ZMQ
PUB bind on :5556 **without** sending stop, so ``replay_real.py`` can bind
the same port and switch to STREAMED_MOTION.

Usage:
    # After deploy is up (recommended before token replay):
    python scripts/replay/enable_control.py
    # → stands ~3s, exits, port 5556 freed; robot stays in CONTROL/PLANNER

    # Keep holding idle forever (must Ctrl+C before starting replay):
    python scripts/replay/enable_control.py --hold

Requirements:
    - C++ deploy running with --input-type zmq_manager
    - Nothing else binding localhost:5556 (no pico / no parallel replay)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party" / "GR00T-WholeBodyControl"
sys.path.insert(0, str(_THIRD_PARTY))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)

LOCOMOTION_MODE_IDLE = 0


class ZMQController:
    """Enable PLANNER control, then hand off the ZMQ port for replay."""

    def __init__(
        self,
        host: str = "*",
        port: int = 5556,
        rate_hz: float = 30.0,
        verbose: bool = True,
    ):
        self.host = host
        self.port = port
        self.interval = 1.0 / rate_hz
        self.verbose = verbose
        self.running = True
        self.ctx = None
        self.sock = None
        self._frame_index = 0
        self._send_stop_on_exit = False

    def connect(self):
        """Bind ZMQ PUB socket."""
        import zmq

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)

        bind_host = self.host if self.host not in ("localhost", "127.0.0.1") else "*"
        endpoint = f"tcp://{bind_host}:{self.port}"

        try:
            self.sock.bind(endpoint)
        except zmq.ZMQError as e:
            print(f"[ZMQController] Failed to bind to {endpoint}: {e}")
            print(
                "[ZMQController] Port already in use — stop pico / another "
                "enable_control / replay first (deploy SUB expects one PUB on 5556)."
            )
            raise

        if self.verbose:
            print(f"[ZMQController] Bound to {endpoint}")

        time.sleep(0.5)

    def send_start_planner(self):
        """Send start command with planner=True to enter PLANNER mode."""
        msg = build_command_message(start=True, stop=False, planner=True)
        self.sock.send(msg)
        if self.verbose:
            print("[ZMQController] Sent: start=True, planner=True (enter CONTROL)")

    def send_idle_planner(self):
        """Send idle planner command to keep robot upright."""
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
        )
        self.sock.send(msg)
        self._frame_index += 1
        if self.verbose and self._frame_index % 30 == 0:
            print(
                f"[ZMQController] Sending idle planner commands "
                f"({self._frame_index} frames sent)"
            )

    def _idle_for(self, seconds: float):
        """Send idle planner frames for ``seconds``."""
        deadline = time.perf_counter() + max(0.0, seconds)
        prev_time = time.perf_counter()
        while self.running and time.perf_counter() < deadline:
            self.send_idle_planner()
            elapsed = time.perf_counter() - prev_time
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_time = time.perf_counter()

    def release(self, send_stop: bool = False):
        """Close PUB without (by default) stopping control — hand off to replay."""
        if self.sock:
            if send_stop:
                msg = build_command_message(start=False, stop=True, planner=True)
                self.sock.send(msg)
                if self.verbose:
                    print("[ZMQController] Sent: stop=True")
                time.sleep(0.1)
            else:
                if self.verbose:
                    print(
                        "[ZMQController] Releasing port without stop "
                        "(robot stays in PLANNER/CONTROL for replay handoff)"
                    )
            self.sock.close(linger=0)
            self.sock = None
        if self.ctx:
            self.ctx.term()
            self.ctx = None
        print("[ZMQController] Stopped — port freed for replay_real.py")

    def run_handoff(self, warmup_seconds: float = 2.0, hold_seconds: float = 3.0):
        """Stand up, idle briefly, exit and free :5556 for replay."""
        print("[ZMQController] Step 1/3: Sending start command (planner mode)...")
        self.send_start_planner()

        print(f"[ZMQController] Step 2/3: Waiting {warmup_seconds}s for planner init...")
        time.sleep(warmup_seconds)

        print(
            f"[ZMQController] Step 3/3: Holding idle for {hold_seconds}s, "
            "then releasing port for replay..."
        )
        self._idle_for(hold_seconds)
        # Do not send stop — leave CONTROL so replay can switch to STREAMED_MOTION.
        self._send_stop_on_exit = False

    def run_hold(self, warmup_seconds: float = 2.0):
        """Keep sending idle until Ctrl+C (must stop before replay)."""
        print("[ZMQController] Step 1/3: Sending start command (planner mode)...")
        self.send_start_planner()

        print(f"[ZMQController] Step 2/3: Waiting {warmup_seconds}s for planner init...")
        time.sleep(warmup_seconds)

        print(
            "[ZMQController] Step 3/3: Holding idle forever. "
            "Ctrl+C to release port (no stop) before starting replay."
        )
        prev_time = time.perf_counter()
        while self.running:
            self.send_idle_planner()
            elapsed = time.perf_counter() - prev_time
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_time = time.perf_counter()

    def _signal_handler(self, sig, frame):
        print(f"\n[ZMQController] Signal {sig}, shutting down...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Enable SONIC PLANNER control, then hand off ZMQ port to replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended before token replay (stands ~3s then frees :5556):
  python scripts/replay/enable_control.py

  # Keep holding until Ctrl+C (then run replay in another terminal):
  python scripts/replay/enable_control.py --hold

  # Actually stop control on exit (robot may go soft):
  python scripts/replay/enable_control.py --hold --stop-on-exit
""",
    )
    parser.add_argument("--host", type=str, default="*", help="ZMQ bind host (default: *)")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port (default: 5556)")
    parser.add_argument("--rate", type=float, default=30.0, help="Idle command rate Hz")
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Seconds after start before sending idle (default: 2.0)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="In handoff mode: how long to send idle before releasing port (default: 3.0)",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep sending idle until Ctrl+C (must stop before replay binds 5556)",
    )
    parser.add_argument(
        "--stop-on-exit",
        action="store_true",
        help="Send stop=True on exit (default: release port only, keep CONTROL)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    controller = ZMQController(
        host=args.host,
        port=args.port,
        rate_hz=args.rate,
        verbose=not args.quiet,
    )
    controller._send_stop_on_exit = bool(args.stop_on_exit)

    signal.signal(signal.SIGINT, controller._signal_handler)
    signal.signal(signal.SIGTERM, controller._signal_handler)

    try:
        controller.connect()
        if args.hold:
            controller.run_hold(warmup_seconds=args.warmup)
        else:
            controller.run_handoff(
                warmup_seconds=args.warmup,
                hold_seconds=args.hold_seconds,
            )
    except KeyboardInterrupt:
        print("\n[ZMQController] Interrupted")
    except Exception as e:
        print(f"[ZMQController] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        controller.release(send_stop=controller._send_stop_on_exit)


if __name__ == "__main__":
    main()
