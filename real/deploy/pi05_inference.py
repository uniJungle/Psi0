"""π0.5 real-robot client for SONIC + Brainco (33D state / 68D action).

Talks to ``baselines/pi05/serve_pi05_sonic.sh`` (OpenPI WebSocket PolicyServer).

Layout (matches walk_to_table_* TrainConfig / modality.json):
  - video: mono ``observation.images.egocentric_right`` (composed_camera right)
  - state: qpos(29) + hand(4) = 33
  - action: motion_token(64) + hand(4) = 68

Body tokens → SONIC WBC via ZMQ Protocol v4.
Brainco hands → DDS (eef.brainco.set_2d_targets).

Optional ``--save-pred-action`` writes closed-loop artifacts like ACT/GR00T:
  ``<DATA_ROOT>/closeloop_pi05_{ckpt}/pred_actions.npy`` (+ plots / lerobot / logs).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_DEPLOY_DIR = Path(__file__).resolve().parent
_PSI0_ROOT = Path(__file__).resolve().parents[2]
_GROOT_WBC = _PSI0_ROOT / "third_party" / "GR00T-WholeBodyControl"
_ACT_BASELINE = _PSI0_ROOT / "baselines" / "act"
_PI05_BASELINE = _PSI0_ROOT / "baselines" / "pi05"
for p in (_GROOT_WBC, _ACT_BASELINE, _PI05_BASELINE, _PSI0_ROOT):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(_DEPLOY_DIR))

from act_inference import (  # noqa: E402
    ACTION_DIM,
    TOKEN_DIM,
    ComposedCamera,
    RobotStateSubscriber,
    STATE_DIM,
    StreamKeepalive,
    TokenPublisher,
    build_state33,
    create_brainco_hand,
    execute_action68,
    extract_body_q_action,
    extract_body_q_measured,
    save_control_logs,
    save_pred_action_trajectory,
    shutdown_brainco_hand,
)
from pi05_ws_client import Pi05WebsocketClient  # noqa: E402
from pred_action_io import resolve_closeloop_dir, write_pred_lerobot  # noqa: E402

TASK_INSTRUCTION = "Go to the table, pick up the apple, place the apple on the pink plate."
FREQ_VLA = 30
MAX_STEPS = 100000

running = threading.Event()
running.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="π0.5 SONIC Brainco closeloop (33/68)")
    parser.add_argument("--host", type=str, default="localhost", help="OpenPI PolicyServer host")
    parser.add_argument("--port", type=int, default=9000, help="OpenPI WebSocket port")
    parser.add_argument("--zmq-host", type=str, default="localhost")
    parser.add_argument("--zmq-pub-port", type=int, default=5556)
    parser.add_argument("--zmq-sub-port", type=int, default=5557)
    parser.add_argument("--zmq-topic", type=str, default="pose")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug")
    parser.add_argument(
        "--camera-address",
        type=str,
        default="tcp://192.168.123.164:5555",
        help="SONIC composed_camera PUB (uses right/ego_view_right)",
    )
    parser.add_argument(
        "--camera-stream-key",
        type=str,
        default=None,
        help="Force composed_camera stream key (default: prefer ego_view_right)",
    )
    parser.add_argument("--instruction", type=str, default=TASK_INSTRUCTION)
    parser.add_argument("--freq", type=float, default=FREQ_VLA)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=1,
        help="How many steps from each predicted action chunk to execute before the next policy query (default: 1)",
    )
    parser.add_argument("--eef", type=str, default="brainco", choices=["none", "brainco"])
    parser.add_argument("--dds-interface", type=str, default="enp5s0")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--save-pred-action",
        type=str,
        default=None,
        metavar="DATA_ROOT",
        help=(
            "Task dataset root, e.g. .../SONIC/walk_to_table_and_place_apple_on_pink_plate. "
            "Writes to <DATA_ROOT>/closeloop_pi05_{ckpt_step}/ "
            "(pred_actions.npy + plots + lerobot_v2.1)."
        ),
    )
    parser.add_argument(
        "--ckpt-step",
        type=int,
        default=None,
        help="Checkpoint step used by the π0.5 server (required with --save-pred-action)",
    )
    args = parser.parse_args()
    execute_horizon = max(1, int(args.execute_horizon))

    save_pred_dir: Path | None = None
    template_dir: Path | None = None
    if args.save_pred_action is not None:
        if args.ckpt_step is None:
            print("[MAIN] ERROR: --ckpt-step is required when using --save-pred-action")
            return
        data_root = Path(args.save_pred_action).expanduser().resolve()
        save_pred_dir = resolve_closeloop_dir(data_root, args.ckpt_step, prefix="pi05")
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

    policy = Pi05WebsocketClient(host=args.host, port=args.port)
    print(f"[MAIN] Connected π0.5 server {args.host}:{args.port}")
    print(f"[MAIN] metadata={policy.get_server_metadata()}")
    try:
        policy.reset()
    except Exception as exc:
        print(f"[MAIN] reset warning: {exc}")

    brainco_hand = None
    if args.eef == "brainco":
        brainco_hand = create_brainco_hand(args.dds_interface)
    else:
        print("[MAIN] --eef=none: body token only")

    token_publisher = TokenPublisher(host="*", port=args.zmq_pub_port, topic=args.zmq_topic)
    state_sub = RobotStateSubscriber(
        host=args.zmq_host, port=args.zmq_sub_port, topic=args.zmq_sub_topic, queue_size=1
    )
    camera = ComposedCamera(address=args.camera_address, stream_key=args.camera_stream_key)
    keepalive = StreamKeepalive(token_publisher, stream_hz=max(args.freq, 30.0))
    print(f"[MAIN] Camera={args.camera_address} (mono right), eef={args.eef}")
    print(f"[MAIN] execute_horizon={execute_horizon} → action_chunk = actions[:{execute_horizon}]")

    print("[MAIN] Waiting for camera...")
    camera.wait_ready(timeout_s=20.0)

    print("[MAIN] Waiting for robot state (g1_debug)...")
    last_robot_state = None
    for _ in range(30):
        st = state_sub.get_state()
        if st is not None:
            last_robot_state = st
            print(f"[MAIN] Got robot state keys={list(st.keys())}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: no robot state after 15s")

    def _signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    dt = 1.0 / max(args.freq, 1e-3)

    first_actions = None
    if last_robot_state is not None:
        try:
            states0 = build_state33(last_robot_state, brainco_hand=brainco_hand)
            frame0 = camera.get_frame()
            print(f"[MAIN] Warmup query frame={frame0.shape} state={states0.shape}")
            first_actions = policy.query_action68(
                frame0, states0, args.instruction, reset=True
            )
            print(f"[MAIN] Warmup action chunk={first_actions.shape}")
        except Exception as exc:
            print(f"[MAIN] Warmup failed (will retry in loop): {exc}")

    token_publisher.enter_streamed_motion(warmup_seconds=args.warmup)
    if first_actions is not None and first_actions.size:
        keepalive.set_action(first_actions[0])
    keepalive.start()
    print(f"[MAIN] Running @ {args.freq} Hz in STREAMED_MOTION. Ctrl+C to stop.")

    next_send_time = time.perf_counter()
    try:
        for step in range(args.max_steps):
            if not running.is_set():
                break
            try:
                keepalive.resume()
                robot_state = state_sub.get_state()
                if robot_state is not None:
                    last_robot_state = robot_state
                elif last_robot_state is not None:
                    robot_state = last_robot_state
                else:
                    print("[VLA] waiting for robot state...")
                    time.sleep(0.05)
                    continue

                states = build_state33(robot_state, brainco_hand=brainco_hand)
                if states.shape[0] != STATE_DIM:
                    print(f"[VLA] bad state dim {states.shape}")
                    continue

                if step == 0 and first_actions is not None:
                    actions = first_actions
                    first_actions = None
                else:
                    frame = camera.get_frame()
                    actions = policy.query_action68(frame, states, args.instruction)

                if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                    print(f"[VLA] bad action shape {actions.shape}")
                    continue

                action_chunk = actions[: min(execute_horizon, actions.shape[0])]
                keepalive.pause()
                try:
                    for i in range(action_chunk.shape[0]):
                        if not running.is_set():
                            break
                        now = time.perf_counter()
                        if now < next_send_time:
                            time.sleep(next_send_time - now)

                        raw_action = np.asarray(action_chunk[i], dtype=np.float32).reshape(-1)[
                            :ACTION_DIM
                        ]
                        send_t = time.perf_counter()
                        token_qtz, left_2d, right_2d = execute_action68(
                            raw_action, token_publisher, brainco_hand=brainco_hand
                        )
                        keepalive.set_last_sent(token_qtz, left_2d, right_2d)

                        post_state = state_sub.get_state()
                        if post_state is not None:
                            last_robot_state = post_state
                        else:
                            post_state = last_robot_state

                        q_action = extract_body_q_action(post_state)
                        q_measured = extract_body_q_measured(post_state)

                        if save_pred_dir is not None:
                            pred_action_buffer.append(raw_action.copy())
                            sent_quantized_token_buffer.append(
                                np.asarray(token_qtz, dtype=np.float32).reshape(-1)[:TOKEN_DIM].copy()
                            )
                            sent_timestamp_buffer.append(float(send_t))
                            if q_action is not None:
                                body_q_action_buffer.append(q_action.astype(np.float32, copy=True))
                            if q_measured is not None:
                                body_q_measured_buffer.append(q_measured.astype(np.float32, copy=True))

                        if step % 10 == 0 and i == 0:
                            print(
                                f"[VLA] step={step} chunk={actions.shape} exec={action_chunk.shape} "
                                f"horizon={execute_horizon} "
                                f"token[:3]={token_qtz[:3]} L={left_2d} R={right_2d}"
                            )

                        next_send_time += dt
                        if next_send_time < time.perf_counter() - dt:
                            next_send_time = time.perf_counter() + dt
                finally:
                    keepalive.resume()
            except Exception as exc:
                print(f"[VLA] step {step} failed: {exc}")
                try:
                    keepalive.resume()
                except Exception:
                    pass
                next_send_time = time.perf_counter() + dt
                time.sleep(min(dt, 0.05))
    finally:
        print("[MAIN] Shutting down...")
        running.clear()
        try:
            keepalive.resume()
        except Exception:
            pass
        keepalive.stop()
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
        policy.close()

        if save_pred_dir is not None and pred_action_buffer:
            try:
                pred = np.stack(pred_action_buffer, axis=0)
                save_pred_action_trajectory(
                    pred, save_pred_dir, title="π0.5 closed-loop pred"
                )
                states_for_lerobot = None
                if body_q_measured_buffer and len(body_q_measured_buffer) == pred.shape[0]:
                    states_for_lerobot = np.stack(body_q_measured_buffer, axis=0)
                write_pred_lerobot(
                    out_dir=save_pred_dir,
                    actions=pred,
                    states=states_for_lerobot,
                    fps=int(round(args.freq)),
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
