"""Psi0 real-robot RTC client for SONIC + Brainco (33D state / 68D action).

WebSocket client for ``psi_serve_rtc_token-sonic.py`` (RTC mode @ 30 Hz).

Layout (matches finetune run_config):
  - image key: observation.images.egocentric_right
  - state: qpos(29) + hand(4) = 33
  - action: motion_token(64) + hand(4) = 68
    hand(4) = left[thumb_aux, others] + right[thumb_aux, others]

Body tokens → SONIC WBC via ZMQ Protocol v4.
Brainco hands → DDS (eef.brainco.set_2d_targets), same as act_inference / BRAINCO_HAND.md.

Optional ``--save-pred-action`` writes the same closed-loop artifacts as ACT:
  ``<DATA_ROOT>/closeloop_psi0_{ckpt}/pred_actions.npy`` (+ plots / lerobot / logs).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from websocket import WebSocketApp
except ModuleNotFoundError as exc:
    if exc.name == "websocket":
        raise ModuleNotFoundError(
            "Missing dependency: websocket-client\n"
            "Install in current env with:\n"
            "  python -m pip install websocket-client"
        ) from exc
    raise

_DEPLOY_DIR = Path(__file__).resolve().parent
_PSI0_ROOT = Path(__file__).resolve().parents[2]
_GROOT_ROOT = _PSI0_ROOT / "third_party" / "GR00T-WholeBodyControl"
_ACT_BASELINE = _PSI0_ROOT / "baselines" / "act"
for p in (_GROOT_ROOT, _ACT_BASELINE, _PSI0_ROOT):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Unconditional: keep real/deploy ahead of baselines/act (same module name).
sys.path.insert(0, str(_DEPLOY_DIR))

from act_inference import (  # noqa: E402
    ACTION_DIM,
    TOKEN_DIM,
    ComposedCamera,
    IMAGE_KEY,
    RobotStateSubscriber,
    STATE_DIM,
    TokenPublisher,
    build_state33,
    convert_numpy_in_dict,
    create_brainco_hand,
    execute_action68,
    extract_body_q_action,
    extract_body_q_measured,
    numpy_deserialize,
    numpy_serialize,
    save_control_logs,
    save_pred_action_trajectory,
    shutdown_brainco_hand,
)
from pred_action_io import resolve_closeloop_dir, write_pred_lerobot  # noqa: E402

TASK_INSTRUCTION = "walk to table and place apple on pink plate"

running = threading.Event()
running.set()


class Psi0RTCWebSocketClient:
    """Bidirectional WebSocket client: stream obs → receive 68D actions @ ~30 Hz."""

    def __init__(
        self,
        server_url: str,
        state_subscriber: RobotStateSubscriber,
        camera: ComposedCamera,
        token_publisher: TokenPublisher,
        instruction: str,
        brainco_hand=None,
        pred_action_buffer: list[np.ndarray] | None = None,
        sent_quantized_token_buffer: list[np.ndarray] | None = None,
        sent_timestamp_buffer: list[float] | None = None,
        body_q_action_buffer: list[np.ndarray] | None = None,
        body_q_measured_buffer: list[np.ndarray] | None = None,
        save_enabled: bool = False,
    ):
        self.server_url = server_url
        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._instruction = instruction
        self._brainco_hand = brainco_hand
        self._save_enabled = save_enabled
        self._pred_action_buffer = pred_action_buffer
        self._sent_quantized_token_buffer = sent_quantized_token_buffer
        self._sent_timestamp_buffer = sent_timestamp_buffer
        self._body_q_action_buffer = body_q_action_buffer
        self._body_q_measured_buffer = body_q_measured_buffer

        self._running = True
        self._connected = threading.Event()
        self._ws: WebSocketApp | None = None
        self._send_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._action_count = 0
        self._last_robot_state: dict | None = None
        self._send_start = time.time()
        self._recv_start = time.time()

    def execute_action(self, action: np.ndarray) -> None:
        if action.ndim > 1:
            action = action[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < ACTION_DIM:
            raise ValueError(f"action dim {action.size} < {ACTION_DIM}")

        raw_action = action[:ACTION_DIM].copy()
        send_t = time.perf_counter()
        token_qtz, left_2d, right_2d = execute_action68(
            raw_action,
            self._token_publisher,
            brainco_hand=self._brainco_hand,
        )

        if self._save_enabled and self._pred_action_buffer is not None:
            post_state = self._state_sub.get_state()
            if post_state is not None:
                self._last_robot_state = post_state
            else:
                post_state = self._last_robot_state

            q_action = extract_body_q_action(post_state)
            q_measured = extract_body_q_measured(post_state)

            self._pred_action_buffer.append(raw_action.copy())
            if self._sent_quantized_token_buffer is not None:
                self._sent_quantized_token_buffer.append(
                    np.asarray(token_qtz, dtype=np.float32).reshape(-1)[:TOKEN_DIM].copy()
                )
            if self._sent_timestamp_buffer is not None:
                self._sent_timestamp_buffer.append(float(send_t))
            if self._body_q_action_buffer is not None and q_action is not None:
                self._body_q_action_buffer.append(q_action.astype(np.float32, copy=True))
            if self._body_q_measured_buffer is not None and q_measured is not None:
                self._body_q_measured_buffer.append(q_measured.astype(np.float32, copy=True))

        _ = (left_2d, right_2d)

    def _on_open(self, ws) -> None:
        print("[Psi0] WebSocket connected")
        self._connected.set()

    def _on_message(self, ws, message: str) -> None:
        interval = time.time() - self._recv_start
        self._recv_start = time.time()
        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)
            if action_data is None:
                return
            action = convert_numpy_in_dict(action_data, numpy_deserialize)
            if not isinstance(action, np.ndarray):
                return
            with self._action_lock:
                self.execute_action(action)
                self._action_count += 1
            if self._action_count <= 3 or self._action_count % 30 == 0:
                print(
                    f"[Psi0] action #{self._action_count} version={version} "
                    f"shape={action.shape} recv_interval={interval:.3f}s"
                )
        except Exception as exc:
            print(f"[Psi0] action handling error: {exc}")

    def _on_error(self, ws, error) -> None:
        print(f"[Psi0] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        print(f"[Psi0] WebSocket closed: {close_status_code} {close_msg}")
        self._running = False
        running.clear()

    def _send_thread(self) -> None:
        print("[Psi0] Send thread waiting for connection...")
        self._connected.wait()
        print("[Psi0] Streaming observations")

        while self._running and running.is_set():
            try:
                robot_state = self._state_sub.get_state()
                if robot_state is not None:
                    self._last_robot_state = robot_state
                elif self._last_robot_state is not None:
                    robot_state = self._last_robot_state
                else:
                    time.sleep(0.05)
                    continue

                states = build_state33(robot_state, brainco_hand=self._brainco_hand)
                if states.shape[0] != STATE_DIM:
                    print(f"[Psi0] bad state dim {states.shape}, expected ({STATE_DIM},)")
                    time.sleep(0.05)
                    continue

                frame = self._camera.get_frame()
                payload = {
                    "image": {IMAGE_KEY: np.asarray(frame, dtype=np.uint8)},
                    "state": {"states": states.astype(np.float32)},
                    "gt_action": None,
                    "dataset_name": "real",
                    "instruction": self._instruction,
                    "history": None,
                    "condition": None,
                    "timestamp": None,
                }
                message = json.dumps(convert_numpy_in_dict(payload, numpy_serialize))

                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[Psi0] WebSocket disconnected, stopping send thread")
                        break

                interval = time.time() - self._send_start
                self._send_start = time.time()
                if self._action_count <= 3 or self._action_count % 30 == 0:
                    print(f"[Psi0] sent obs state={states.shape} send_interval={interval:.3f}s")

            except Exception as exc:
                print(f"[Psi0] send error: {exc}")
                break

            time.sleep(1.0 / 30.0)

        print("[Psi0] Send thread stopped")

    def run(self) -> None:
        print(f"[Psi0] Connecting to {self.server_url}")
        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()
        self._ws.run_forever()

        self._running = False
        send_thread.join(timeout=1.0)
        print("[Psi0] Client stopped")

    def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            self._ws.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Psi0 SONIC Brainco RTC inference client (33/68)")
    parser.add_argument("--host", type=str, default="localhost", help="Psi0 policy server host")
    parser.add_argument("--port", type=int, default=8014, help="Psi0 policy server port")
    parser.add_argument("--zmq-host", type=str, default="localhost", help="Robot state ZMQ host")
    parser.add_argument("--zmq-pub-port", type=int, default=5556, help="Token PUB port to WBC")
    parser.add_argument("--zmq-sub-port", type=int, default=5557, help="Robot state SUB port")
    parser.add_argument("--zmq-topic", type=str, default="pose")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug")
    parser.add_argument(
        "--camera-address",
        type=str,
        default="tcp://192.168.123.164:5555",
        help="SONIC composed_camera PUB endpoint (SUB client)",
    )
    parser.add_argument(
        "--camera-stream-key",
        type=str,
        default=None,
        help="Live stream key (default: ego_view_right → egocentric_right)",
    )
    parser.add_argument("--instruction", type=str, default=TASK_INSTRUCTION)
    parser.add_argument(
        "--eef",
        type=str,
        default="brainco",
        choices=["none", "brainco"],
        help="End-effector: brainco (DDS 2D→6D) or none (body token only)",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default="enp5s0",
        help="DDS network interface for Brainco (see BRAINCO_HAND.md)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Seconds in PLANNER after start before STREAMED_MOTION",
    )
    parser.add_argument(
        "--save-pred-action",
        type=str,
        default=None,
        metavar="DATA_ROOT",
        help=(
            "Task dataset root, e.g. .../SONIC/walk_to_table_and_place_apple_on_pink_plate. "
            "Writes to <DATA_ROOT>/closeloop_psi0_{ckpt_step}/ "
            "(pred_actions.npy + plots + lerobot_v2.1)."
        ),
    )
    parser.add_argument(
        "--ckpt-step",
        type=int,
        default=None,
        help="Checkpoint step used by the Psi0 server (required with --save-pred-action)",
    )
    args = parser.parse_args()

    save_pred_dir: Path | None = None
    template_dir: Path | None = None
    if args.save_pred_action is not None:
        if args.ckpt_step is None:
            print("[MAIN] ERROR: --ckpt-step is required when using --save-pred-action")
            return
        data_root = Path(args.save_pred_action).expanduser().resolve()
        save_pred_dir = resolve_closeloop_dir(data_root, args.ckpt_step, prefix="psi0")
        save_pred_dir.mkdir(parents=True, exist_ok=True)
        template_candidate = data_root / "lerobot_v2.1"
        if template_candidate.is_dir():
            template_dir = template_candidate
        print(f"[MAIN] Will save closeloop outputs to {save_pred_dir}")
        if template_dir is not None:
            print(f"[MAIN] lerobot template={template_dir}")

    pred_action_buffer: list[np.ndarray] = []
    sent_quantized_token_buffer: list[np.ndarray] = []
    sent_timestamp_buffer: list[float] = []
    body_q_action_buffer: list[np.ndarray] = []
    body_q_measured_buffer: list[np.ndarray] = []

    brainco_hand = None
    if args.eef == "brainco":
        brainco_hand = create_brainco_hand(args.dds_interface)
    else:
        print("[MAIN] --eef=none: body token only, hands not commanded")

    token_publisher = TokenPublisher(host="*", port=args.zmq_pub_port, topic=args.zmq_topic)
    state_sub = RobotStateSubscriber(
        host=args.zmq_host, port=args.zmq_sub_port, topic=args.zmq_sub_topic, queue_size=1
    )
    camera = ComposedCamera(address=args.camera_address, stream_key=args.camera_stream_key)
    print(f"[MAIN] Camera={args.camera_address}, image_key={IMAGE_KEY}, eef={args.eef}")

    print("[MAIN] Waiting for first camera frame...")
    camera.wait_ready(timeout_s=20.0)

    print("[MAIN] Waiting for robot state (g1_debug)...")
    for _ in range(30):
        if state_sub.get_state() is not None:
            print(f"[MAIN] Got robot state keys={list(state_sub.get_state().keys())}")
            break
        time.sleep(0.5)
    else:
        print(
            "[MAIN] WARNING: no robot state after 15s. "
            "Ensure deploy is streaming g1_debug on :5557."
        )

    def _signal_handler(sig, frame) -> None:
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    token_publisher.enter_streamed_motion(warmup_seconds=args.warmup)
    print("[MAIN] STREAMED_MOTION active — starting Psi0 RTC client. Ctrl+C to stop.")

    client = Psi0RTCWebSocketClient(
        server_url=f"ws://{args.host}:{args.port}/ws",
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        instruction=args.instruction,
        brainco_hand=brainco_hand,
        pred_action_buffer=pred_action_buffer,
        sent_quantized_token_buffer=sent_quantized_token_buffer,
        sent_timestamp_buffer=sent_timestamp_buffer,
        body_q_action_buffer=body_q_action_buffer,
        body_q_measured_buffer=body_q_measured_buffer,
        save_enabled=save_pred_dir is not None,
    )

    ws_thread = threading.Thread(target=client.run, daemon=True)
    ws_thread.start()

    try:
        while running.is_set() and ws_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        running.clear()

    print("[MAIN] Shutting down...")
    client.stop()
    ws_thread.join(timeout=2.0)
    try:
        token_publisher.handoff_to_idle_planner(hold_seconds=2.0)
    except Exception as exc:
        print(f"[MAIN] handoff failed: {exc}")
    try:
        token_publisher.stop(send_stop=False)
    except Exception as exc:
        print(f"[MAIN] publisher stop failed: {exc}")
    camera.close()
    shutdown_brainco_hand(brainco_hand)
    state_sub.stop()

    if save_pred_dir is not None and pred_action_buffer:
        try:
            pred = np.stack(pred_action_buffer, axis=0)
            save_pred_action_trajectory(pred, save_pred_dir, title="Psi0 closed-loop pred")
            states_for_lerobot = None
            if body_q_measured_buffer and len(body_q_measured_buffer) == pred.shape[0]:
                states_for_lerobot = np.stack(body_q_measured_buffer, axis=0)
            write_pred_lerobot(
                out_dir=save_pred_dir,
                actions=pred,
                states=states_for_lerobot,
                fps=30,
                instruction=args.instruction,
                template_dir=template_dir,
                keep_videos=False,
            )
            save_control_logs(
                save_pred_dir,
                sent_quantized_token_buffer,
                sent_timestamp_buffer,
                body_q_action_buffer,
                body_q_measured_buffer,
            )
        except Exception as exc:
            print(f"[SAVE] failed to save closeloop outputs: {exc}")
    elif save_pred_dir is not None:
        print("[SAVE] no pred actions recorded")
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
