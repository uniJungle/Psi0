#!/usr/bin/env python3
"""Auto-enable controller for SONIC deploy (zmq_manager mode).

This script automatically enables robot control after C++ deploy starts,
sending the required ZMQ commands to:
  1. Enter PLANNER mode with start=True
  2. Send idle planner commands to keep the robot upright

Usage:
    # Option A: Launch deploy and auto-enable in sequence
    bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy &
    sleep 5  # Wait for deploy to start
    python scripts/replay/enable_control.py

    # Option B: Run in parallel with deploy (from another terminal)
    python scripts/replay/enable_control.py

This is particularly useful for token replay workflows where you want
the robot to be in a stable state before starting replay.

Requirements:
    - C++ deploy must be running with --input-type zmq_manager
    - ZMQ bind at localhost:5556
"""

from __future__ import annotations

import sys
import time
import signal
import argparse
from pathlib import Path

# Add third_party/GR00T-WholeBodyControl to path
_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party" / "GR00T-WholeBodyControl"
sys.path.insert(0, str(_THIRD_PARTY))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


# LocomotionMode enum values (from localmotion_kplanner.hpp)
LOCOMOTION_MODE_IDLE = 0
LOCOMOTION_MODE_STANDING = 1


class ZMQController:
    """Simple ZMQ controller that keeps the robot in a stable state."""

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
            print("[ZMQController] Is deploy running? Check if port 5556 is already in use.")
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
            print(f"[ZMQController] Sending idle planner commands ({self._frame_index} frames sent)")

    def stop(self):
        """Send stop command and cleanup."""
        if self.sock:
            msg = build_command_message(start=False, stop=True, planner=True)
            self.sock.send(msg)
            if self.verbose:
                print("[ZMQController] Sent: stop=True")
            self.sock.close(linger=0)
        if self.ctx:
            self.ctx.term()
        print("[ZMQController] Stopped")

    def run(self, warmup_seconds: float = 2.0):
        """Run the controller loop."""
        print("[ZMQController] Step 1/3: Sending start command (planner mode)...")
        self.send_start_planner()

        print(f"[ZMQController] Step 2/3: Waiting {warmup_seconds}s for planner to initialize...")
        time.sleep(warmup_seconds)

        print("[ZMQController] Step 3/3: Sending idle planner commands to keep robot upright...")
        print("[ZMQController] Press Ctrl+C to stop and release control")

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
        description="Auto-enable SONIC robot control after deploy starts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After starting deploy, run this in another terminal:
  python scripts/replay/enable_control.py

  # With custom port and warmup time:
  python scripts/replay/enable_control.py --port 5556 --warmup 3.0

  # Quick launch both deploy and controller:
  (bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy &) && sleep 5 && python scripts/replay/enable_control.py
""",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="ZMQ bind host (default: * for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="ZMQ port (default: 5556)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Planner command rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Warmup time after start command (default: 2.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    controller = ZMQController(
        host=args.host,
        port=args.port,
        rate_hz=args.rate,
        verbose=not args.quiet,
    )

    signal.signal(signal.SIGINT, controller._signal_handler)
    signal.signal(signal.SIGTERM, controller._signal_handler)

    try:
        controller.connect()
        controller.run(warmup_seconds=args.warmup)
    except KeyboardInterrupt:
        print("\n[ZMQController] Interrupted")
    except Exception as e:
        print(f"[ZMQController] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
