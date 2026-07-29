"""ACT real-robot client for SONIC + Brainco (33D state / 68D action).

Matches training layout from run_config:
  - image key: observation.images.egocentric_right
  - state: qpos(29) + hand(4) = 33
  - action: motion_token(64) + hand(4) = 68
    hand(4) = left[thumb_aux, others] + right[thumb_aux, others]

Body tokens go to SONIC WBC via ZMQ Protocol v4.
Brainco hands go via DDS (eef.brainco), same path as replay_real / BRAINCO_HAND.md:
  2D [thumb_aux, others] → broadcast to 6 motors → rt/brainco/*/cmd
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from base64 import b64decode, b64encode
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import requests
import zmq
from numpy.lib.format import descr_to_dtype, dtype_to_descr

# gear_sonic / eef live in the GR00T-WholeBodyControl submodule.
_GROOT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "GR00T-WholeBodyControl"
if _GROOT_ROOT.is_dir():
    sys.path.insert(0, str(_GROOT_ROOT))

_DEPLOY_DIR = Path(__file__).resolve().parent
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# ---- layout matching ACT_200k_g1_33d_* training ----
IMAGE_KEY = "observation.images.egocentric_right"
STATE_DIM = 33
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2  # brainco per hand: [thumb_aux, others]
QPOS_DIM = 29

FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625

LOCOMOTION_MODE_IDLE = 0

TASK_INSTRUCTION = "walk to table and place apple on pink plate"
FREQ_VLA = 30
MAX_STEPS = 100000

running = threading.Event()
running.set()


def _observation_for_rerun(frame_rgb: np.ndarray) -> dict:
    """Build LeRobot-style image observation for Rerun (CHW torch tensor)."""
    import torch

    chw = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1)
    return {IMAGE_KEY: chw}


def _state_for_rerun(states: np.ndarray) -> np.ndarray:
    return np.asarray(states, dtype=np.float64).ravel()[:STATE_DIM]


def _action_for_rerun(action: np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float64).ravel()[:ACTION_DIM]


HAND_DIM_NAMES = ("L_thumb_aux", "L_others", "R_thumb_aux", "R_others")


def _dim_label(dim: int) -> str:
    if dim < TOKEN_DIM:
        return f"token[{dim}]"
    hand_i = dim - TOKEN_DIM
    name = HAND_DIM_NAMES[hand_i] if hand_i < len(HAND_DIM_NAMES) else f"hand[{hand_i}]"
    return f"action[{dim}] {name}"


def _state_dim_label(dim: int) -> str:
    return f"qpos[{dim}]"


def save_pred_action_trajectory(pred: np.ndarray, out_dir: Path) -> None:
    """Save executed 68D pred trajectory: .npy + 68 rows × 1 col plot (same layout as openloop)."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected pred shape (T, {ACTION_DIM}), got {pred.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "pred_actions.npy"
    np.save(npy_path, pred)
    print(f"[SAVE] pred actions {pred.shape} -> {npy_path}")

    t = np.arange(pred.shape[0])
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        ACTION_DIM,
        1,
        figsize=(14, 1.1 * ACTION_DIM),
        sharex=True,
        constrained_layout=True,
    )
    if ACTION_DIM == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(t, pred[:, dim], linestyle="-", alpha=0.95, color="C0", label="pred", linewidth=1.0)
        ax.set_ylabel(_dim_label(dim), fontsize=7, rotation=0, labelpad=55, va="center")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(f"ACT closed-loop pred  |  all {ACTION_DIM} dims", fontsize=11)

    axes[-1].set_xlabel("frame index", fontsize=9)
    plot_path = plots_dir / "closedloop_pred_all_dims.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] plot -> {plot_path}")


def save_pred_state_trajectory(pred: np.ndarray, out_dir: Path) -> None:
    """Save postprocessed 29D qpos trajectory: .npy + 29 rows × 1 col plot."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] != QPOS_DIM:
        raise ValueError(f"Expected pred state shape (T, {QPOS_DIM}), got {pred.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "pred_states.npy"
    np.save(npy_path, pred)
    print(f"[SAVE] pred states {pred.shape} -> {npy_path}")

    t = np.arange(pred.shape[0])
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        QPOS_DIM,
        1,
        figsize=(14, 1.1 * QPOS_DIM),
        sharex=True,
        constrained_layout=True,
    )
    if QPOS_DIM == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(t, pred[:, dim], linestyle="-", alpha=0.95, color="C0", label="pred_state", linewidth=1.0)
        ax.set_ylabel(_state_dim_label(dim), fontsize=7, rotation=0, labelpad=45, va="center")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(f"ACT closed-loop pred qpos  |  all {QPOS_DIM} dims", fontsize=11)

    axes[-1].set_xlabel("frame index", fontsize=9)
    plot_path = plots_dir / "closedloop_pred_state_all_dims.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] plot -> {plot_path}")


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    return np.clip(quantized, fsq_min, fsq_max)


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def pad_hand_to_dex3(hand: np.ndarray) -> np.ndarray:
    """Pad Brainco 2D hand to Dex3 7D wire format expected by Protocol v4."""
    hand = np.asarray(hand, dtype=np.float32).reshape(-1)
    if hand.size >= 7:
        return hand[:7].astype(np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: hand.size] = hand
    return out


def take_hand(vec, hand_dim: int = HAND_DIM) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size >= hand_dim:
        return arr[:hand_dim]
    out = np.zeros(hand_dim, dtype=np.float32)
    out[: arr.size] = arr
    return out


def split_action68(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 68D action → token64, left_2d[thumb_aux, others], right_2d[thumb_aux, others]."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < ACTION_DIM:
        raise ValueError(f"action dim {action.size} < {ACTION_DIM}")
    token = action[:TOKEN_DIM]
    left_2d = action[TOKEN_DIM : TOKEN_DIM + HAND_DIM]
    right_2d = action[TOKEN_DIM + HAND_DIM : TOKEN_DIM + 2 * HAND_DIM]
    return token, left_2d, right_2d


def create_brainco_hand(dds_interface: str):
    try:
        from eef.brainco.brainco import Brainco
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n"
            "Brainco DDS needs SONIC teleop deps (unitree_sdk2py + eef).\n"
            "Run client in GR00T .venv_teleop, or install into current env:\n"
            "  cd third_party/GR00T-WholeBodyControl\n"
            "  uv pip install -e external_dependencies/unitree_sdk2_python --python <your-python>\n"
            "See real/SONIC/BRAINCO_HAND.md"
        ) from e

    print(f"[Brainco] Initializing DDS (interface={dds_interface or 'default'})...")
    hand = Brainco(passive=False, network_interface=dds_interface or None)
    hand.change_open_pose(801)
    hand.set_gripper_targets(0.0, 0.0)
    print("[Brainco] Ready — will drive hands via set_2d_targets (2D→6D broadcast)")
    return hand


def shutdown_brainco_hand(hand) -> None:
    if hand is None:
        return
    try:
        hand.set_gripper_targets(0.0, 0.0)
        time.sleep(0.1)
    except Exception as exc:
        print(f"[Brainco] open-on-exit failed: {exc}")
    try:
        hand.close()
    except Exception as exc:
        print(f"[Brainco] close failed: {exc}")
    print("[Brainco] Shutdown complete")


def _parse_tcp_endpoint(address: str) -> tuple[str, int]:
    """Parse 'tcp://host:port' or 'host:port' → (host, port)."""
    s = address.strip()
    if s.startswith("tcp://"):
        s = s[len("tcp://") :]
    if ":" not in s:
        raise ValueError(f"Bad camera address (need host:port): {address}")
    host, port_s = s.rsplit(":", 1)
    return host, int(port_s)


class ComposedCamera:
    """SUB client for SONIC composed_camera SensorServer (PUB on :5555).

    Do NOT use the legacy RealSense REQ ``get_frame`` client here — the robot
    camera is a ZMQ PUB stream (see ``[Sensor server] Message sent: ...``).
    Training stereo right eye maps: ego_view_right → observation.images.egocentric_right.
    """

    # Preference order for live stream keys → IMAGE_KEY
    STREAM_KEY_CANDIDATES = (
        "ego_view_right",
        "egocentric_right",
        "ego_view",
        "egocentric",
    )

    def __init__(self, address: str = "tcp://192.168.123.164:5555", stream_key: str | None = None):
        host, port = _parse_tcp_endpoint(address)
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        self._client = ComposedCameraClientSensor(server_ip=host, port=port)
        self._stream_key = stream_key
        self._resolved_key: str | None = None
        print(f"[Camera] SUB composed_camera at tcp://{host}:{port} (PUB server)")

    def wait_ready(self, timeout_s: float = 15.0) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            sample = self._client.read(blocking=False)
            if sample and sample.get("images"):
                keys = list(sample["images"].keys())
                self._resolved_key = self._pick_key(keys)
                print(f"[Camera] Streams={keys}, using '{self._resolved_key}' → {IMAGE_KEY}")
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"No frames from composed_camera within {timeout_s}s. "
            "On robot: python -m gear_sonic.camera.composed_camera --port 5555 ..."
        )

    def _pick_key(self, keys: list[str]) -> str:
        if self._stream_key:
            if self._stream_key not in keys:
                raise KeyError(f"stream_key={self._stream_key!r} not in {keys}")
            return self._stream_key
        for cand in self.STREAM_KEY_CANDIDATES:
            if cand in keys:
                return cand
        if not keys:
            raise KeyError("camera message has empty images dict")
        return keys[0]

    def get_frame(self) -> np.ndarray:
        """Return RGB uint8 HxWx3 matching training image."""
        sample = self._client.read(blocking=False)
        if sample is None or not sample.get("images"):
            # brief blocking wait for first/new frame
            sample = self._client.read(blocking=True)
        if sample is None or not sample.get("images"):
            raise RuntimeError("composed_camera returned empty message")
        if self._resolved_key is None:
            self._resolved_key = self._pick_key(list(sample["images"].keys()))
        img = sample["images"].get(self._resolved_key)
        if img is None:
            raise KeyError(f"{self._resolved_key} missing; have {list(sample['images'])}")
        return np.asarray(img, dtype=np.uint8)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class RobotStateSubscriber:
    def __init__(self, host="localhost", port=5557, topic="g1_debug", queue_size=1):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._topic = topic
        self._lock = threading.Lock()
        self._state_queue: deque = deque(maxlen=queue_size)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            topic_bytes = self._topic.encode("utf-8")
            payload = msg[len(topic_bytes) :] if msg.startswith(topic_bytes) else msg
            try:
                state = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._state_queue.append(state)
            except Exception as e:
                print(f"[StateSubscriber] Unpack error: {e}")

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return self._state_queue[-1] if self._state_queue else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._context.term()


class TokenPublisher:
    """ZMQ PUB for SONIC zmq_manager — same handshake as replay_real token mode."""

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        bind_host = "*" if host in ("localhost", "127.0.0.1", "") else host
        endpoint = f"tcp://{bind_host}:{port}"
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError as e:
            raise RuntimeError(
                f"Failed to bind {endpoint}: {e}\n"
                "Port 5556 must be free. Run `enable_control.py` (handoff, not --hold) "
                "first and wait until it exits; stop pico / another publisher."
            ) from e
        self._topic = topic
        self._port = port
        self._send_lock = threading.Lock()
        time.sleep(0.5)
        print(f"[TokenPublisher] Bound PUB at {endpoint}")

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        with self._send_lock:
            self._socket.send(msg)
        print(f"[TokenPublisher] Command: start={start} stop={stop} planner={planner}")

    def send_idle_planner(self):
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
        )
        with self._send_lock:
            self._socket.send(msg)

    def enter_streamed_motion(self, warmup_seconds: float = 2.0):
        """Match replay_real token mode handshake exactly.

        ZMQManager only accepts start while in PLANNER:
          1) start + planner=True  → CONTROL
          2) start + planner=False → STREAMED_MOTION (consumes pose/token)

        Caller MUST start publishing pose/token immediately after this returns.
        Deploy auto-falls back to PLANNER+IDLE if no pose for ~1s.
        """
        print("[TokenPublisher] Step 1/2: start control in PLANNER mode...")
        self.send_command(start=True, stop=False, planner=True)
        time.sleep(max(warmup_seconds, 2.0))

        print("[TokenPublisher] Step 2/2: switch to STREAMED_MOTION (pose/token)...")
        self.send_command(start=True, stop=False, planner=False)
        # Keep this short — replay_real sleeps 1.0 then immediately streams tokens.
        # Any further gap before first pose risks STREAM_TIMEOUT → PLANNER+IDLE.
        time.sleep(0.2)

    def handoff_to_idle_planner(self, hold_seconds: float = 2.0):
        """Leave deploy running in PLANNER+idle and free the port."""
        print(
            f"[TokenPublisher] Handoff: PLANNER + idle ({hold_seconds}s), "
            "then release port..."
        )
        self.send_command(start=True, stop=False, planner=True)
        time.sleep(0.5)
        deadline = time.perf_counter() + max(0.0, hold_seconds)
        while time.perf_counter() < deadline:
            self.send_idle_planner()
            time.sleep(1.0 / 30.0)

    def publish_token(self, token64: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray):
        """Publish body token (+ padded hand fields for Protocol v4 wire compat)."""
        pose_data = {
            "token_state": token64.astype(np.float32).reshape(1, -1),
            "left_hand_joints": pad_hand_to_dex3(left_hand).reshape(1, -1),
            "right_hand_joints": pad_hand_to_dex3(right_hand).reshape(1, -1),
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        with self._send_lock:
            self._socket.send(msg)

    def send_stream_heartbeat(self):
        """Quiet command heartbeat (planner=False) to keep STREAMED_MOTION alive."""
        msg = build_command_message(start=True, stop=False, planner=False)
        with self._send_lock:
            self._socket.send(msg)

    def stop(self, send_stop: bool = False):
        if self._socket is not None:
            if send_stop:
                self.send_command(start=False, stop=True, planner=True)
                time.sleep(0.1)
            with self._send_lock:
                self._socket.close(linger=0)
                self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        print(f"[TokenPublisher] Stopped — port {self._port} freed")


class StreamKeepalive:
    """Republish last token at stream_hz while policy inference may take >1s.

    Deploy zmq_manager falls back to PLANNER+IDLE if no pose for ~1s. replay_real
    never has that gap; ACT HTTP inference often does.
    """

    def __init__(self, publisher: TokenPublisher, stream_hz: float = 30.0):
        self._publisher = publisher
        self._interval = 1.0 / max(stream_hz, 1.0)
        self._lock = threading.Lock()
        self._last_action: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cmd_every = 20  # reaffirm STREAMED_MOTION periodically
        self._ticks = 0

    def set_action(self, action68: np.ndarray) -> None:
        with self._lock:
            self._last_action = np.asarray(action68, dtype=np.float32).reshape(-1).copy()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[StreamKeepalive] Publishing last token @ {1.0 / self._interval:.0f} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                action = None if self._last_action is None else self._last_action.copy()
            if action is not None:
                try:
                    token, left_2d, right_2d = split_action68(action)
                    self._publisher.publish_token(fsq_quantize(token), left_2d, right_2d)
                except Exception as e:
                    print(f"[StreamKeepalive] publish failed: {e}")
                self._ticks += 1
                if self._ticks % self._cmd_every == 0:
                    try:
                        self._publisher.send_stream_heartbeat()
                    except Exception:
                        pass
            elapsed = time.perf_counter() - t0
            sleep_t = self._interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


class ACTHTTPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 22085, timeout: float = 30.0):
        self.url = f"http://{host}:{port}/act"
        self.health_url = f"http://{host}:{port}/health"
        self.reset_url = f"http://{host}:{port}/reset"
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        try:
            resp = self.session.get(self.health_url, timeout=self.timeout)
            return resp.ok and resp.json().get("status") == "ok"
        except Exception as e:
            print(f"[ACT] health check failed: {e}")
            return False

    def reset(self) -> None:
        try:
            self.session.get(self.reset_url, timeout=self.timeout)
        except Exception as e:
            print(f"[ACT] reset failed: {e}")

    def query_action(
        self,
        image: np.ndarray,
        states: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        payload = {
            "image": {IMAGE_KEY: np.asarray(image, dtype=np.uint8)},
            "state": {"states": np.asarray(states, dtype=np.float32)},
            "instruction": instruction,
            "history": {},
            "condition": {},
            "gt_action": [],
            "dataset_name": "real",
            "timestamp": str(datetime.now()).replace(" ", "_").replace(":", "-"),
        }
        resp = self.session.post(
            self.url,
            json=convert_numpy_in_dict(payload, numpy_serialize),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = convert_numpy_in_dict(resp.json(), numpy_deserialize)
        if "error" in data:
            raise RuntimeError(data["error"])
        action = np.asarray(data["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action


def build_state33(robot_state: dict, brainco_hand=None) -> np.ndarray:
    """Compose training state: body_q(29) + left_hand(2) + right_hand(2)."""
    body_q = np.asarray(
        robot_state.get("body_q_measured", robot_state.get("body_q")),
        dtype=np.float32,
    ).reshape(-1)
    if body_q.size < QPOS_DIM:
        raise ValueError(f"body_q_measured dim {body_q.size} < {QPOS_DIM}")
    body_q = body_q[:QPOS_DIM]

    if brainco_hand is not None:
        left_2d, right_2d = brainco_hand.get_2d_states()
        left = take_hand(left_2d)
        right = take_hand(right_2d)
    else:
        left = take_hand(
            robot_state.get(
                "left_hand_q_measured",
                robot_state.get("left_hand_q", np.zeros(HAND_DIM)),
            )
        )
        right = take_hand(
            robot_state.get(
                "right_hand_q_measured",
                robot_state.get("right_hand_q", np.zeros(HAND_DIM)),
            )
        )
    return np.concatenate([body_q, left, right], axis=0).astype(np.float32)


def extract_body_q_target(robot_state: dict | None) -> np.ndarray | None:
    if robot_state is None:
        return None
    for key in ("body_q_target", "body_q_measured", "body_q"):
        if key in robot_state and robot_state[key] is not None:
            arr = np.asarray(robot_state[key], dtype=np.float32).reshape(-1)
            if arr.size >= QPOS_DIM:
                return arr[:QPOS_DIM].copy()
    return None


def wait_for_body_q_target(
    state_sub: RobotStateSubscriber,
    prev_qpos: np.ndarray | None,
    timeout_s: float = 0.08,
    poll_s: float = 0.002,
) -> tuple[np.ndarray | None, dict | None]:
    """Poll g1_debug for the latest body_q_target after token decode."""
    deadline = time.perf_counter() + max(timeout_s, 0.0)
    latest_state = state_sub.get_state()
    latest_qpos = extract_body_q_target(latest_state)

    while time.perf_counter() < deadline:
        state = state_sub.get_state()
        if state is not None:
            latest_state = state
            qpos = extract_body_q_target(state)
            if qpos is not None:
                latest_qpos = qpos
                if prev_qpos is None or not np.allclose(qpos, prev_qpos, atol=1e-6, rtol=0.0):
                    return qpos, latest_state
        time.sleep(poll_s)
    return latest_qpos, latest_state


def execute_action68(
    action: np.ndarray,
    token_publisher: TokenPublisher,
    state_sub: RobotStateSubscriber | None = None,
    prev_body_q_target: np.ndarray | None = None,
    brainco_hand=None,
) -> tuple[np.ndarray | None, dict | None]:
    """Send token, fetch decoded 29D qpos from g1_debug, and drive Brainco hands."""
    if action.ndim > 1:
        action = action[0]
    token, left_2d, right_2d = split_action68(action)
    token_qtz = fsq_quantize(token)

    # Body / WBC path
    token_publisher.publish_token(token_qtz, left_2d, right_2d)

    # Brainco DDS path: 2D [thumb_aux, others] → 6D broadcast inside set_2d_targets
    #   cmd[1] = thumb_aux
    #   cmd[[0,2,3,4,5]] = others  (Thumb, Index, Middle, Ring, Pinky)
    if brainco_hand is not None:
        brainco_hand.set_2d_targets(left_2d, right_2d)

    if state_sub is None:
        return None, None

    decoded_qpos, latest_state = wait_for_body_q_target(state_sub, prev_body_q_target)
    if decoded_qpos is None:
        return None, latest_state
    return decoded_qpos, latest_state


def main():
    parser = argparse.ArgumentParser(description="ACT SONIC Brainco inference client (33/68)")
    parser.add_argument("--host", type=str, default="localhost", help="ACT policy server host")
    parser.add_argument("--port", type=int, default=22085, help="ACT policy server port")
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
    parser.add_argument("--freq", type=float, default=FREQ_VLA, help="Policy query / execute rate (Hz)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
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
        default="enp4s0",
        help="DDS network interface for Brainco (see BRAINCO_HAND.md)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Seconds in PLANNER after start before STREAMED_MOTION (match replay_real)",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Open Rerun viewer for live image / state(33) / action(68) (same as eval_g1_36_brainco_act)",
    )
    parser.add_argument(
        "--save-pred-action",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Save executed 68D pred trajectory (.npy + 68×1 plot). "
            "Optional DIR (default: real/deploy/logs/pred_actions/<timestamp>)"
        ),
    )
    parser.add_argument(
        "--save-pred-state",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Save decoded/postprocessed 29D qpos trajectory (.npy + 29×1 plot). "
            "Optional DIR (default: real/deploy/logs/pred_states/<timestamp>)"
        ),
    )
    args = parser.parse_args()

    rerun_logger = None
    rerun_idx = 0
    visualization_data = None
    if args.visualization:
        from utils.rerun_visualizer import RerunLogger, visualization_data
        rerun_logger = RerunLogger()
        print("[MAIN] Rerun visualization enabled")

    save_pred_dir: Path | None = None
    if args.save_pred_action is not None:
        if args.save_pred_action:
            save_pred_dir = Path(args.save_pred_action)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_pred_dir = _DEPLOY_DIR / "logs" / "pred_actions" / ts
        save_pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MAIN] Will save pred actions to {save_pred_dir}")

    save_pred_state_dir: Path | None = None
    if args.save_pred_state is not None:
        if args.save_pred_state:
            save_pred_state_dir = Path(args.save_pred_state)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_pred_state_dir = _DEPLOY_DIR / "logs" / "pred_states" / ts
        save_pred_state_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MAIN] Will save pred states to {save_pred_state_dir}")

    pred_action_buffer: list[np.ndarray] = []
    pred_state_buffer: list[np.ndarray] = []

    policy = ACTHTTPClient(host=args.host, port=args.port)
    print(f"[MAIN] Checking ACT server at {args.host}:{args.port} ...")
    if not policy.health_check():
        print("[MAIN] ACT server not healthy. Start serve_act_g1_real.sh first.")
        return
    print("[MAIN] ACT server OK")
    policy.reset()

    brainco_hand = None
    if args.eef == "brainco":
        brainco_hand = create_brainco_hand(args.dds_interface)
    else:
        print("[MAIN] --eef=none: body token only, hands not commanded")

    # Same ZMQ ownership as replay_real: only one PUB on :5556.
    # Run enable_control.py (default handoff) first so deploy is in CONTROL and port is free.
    token_publisher = TokenPublisher(host="*", port=args.zmq_pub_port, topic=args.zmq_topic)
    state_sub = RobotStateSubscriber(
        host=args.zmq_host, port=args.zmq_sub_port, topic=args.zmq_sub_topic, queue_size=1
    )
    camera = ComposedCamera(address=args.camera_address, stream_key=args.camera_stream_key)
    keepalive = StreamKeepalive(token_publisher, stream_hz=max(args.freq, 30.0))
    print(f"[MAIN] Camera={args.camera_address}, image_key={IMAGE_KEY}, eef={args.eef}")

    # Prep obs BEFORE STREAMED_MOTION — same as replay_real which has frames ready
    # and starts token streaming immediately after the mode switch.
    print("[MAIN] Waiting for first camera frame...")
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
        print(
            "[MAIN] WARNING: no robot state after 15s. "
            "Will retry each step; ensure deploy is streaming g1_debug on :5557."
        )

    def _signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    dt = 1.0 / max(args.freq, 1e-3)

    # Warm first policy query while still in PLANNER (enable_control left deploy there).
    # Avoids multi-second CUDA cold-start AFTER STREAMED_MOTION (would trigger 1s timeout).
    first_actions = None
    last_vis_frame: np.ndarray | None = None
    if last_robot_state is not None:
        try:
            states0 = build_state33(last_robot_state, brainco_hand=brainco_hand)
            frame0 = camera.get_frame()
            last_vis_frame = frame0
            print(f"[MAIN] Warmup policy query, frame={frame0.shape} state={states0.shape}")
            first_actions = policy.query_action(frame0, states0, args.instruction)
            print(f"[MAIN] Warmup action chunk={first_actions.shape}")
        except Exception as e:
            print(f"[MAIN] Warmup policy query failed (will retry in loop): {e}")

    # Match replay_real: switch to STREAMED_MOTION then stream tokens with no gap.
    token_publisher.enter_streamed_motion(warmup_seconds=args.warmup)
    if first_actions is not None and first_actions.size:
        keepalive.set_action(first_actions[0])
    keepalive.start()
    print(f"[MAIN] Running @ {args.freq} Hz in STREAMED_MOTION. Ctrl+C to stop.")
    last_body_q_target: np.ndarray | None = extract_body_q_target(last_robot_state)

    try:
        for step in range(args.max_steps):
            if not running.is_set():
                break
            try:
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
                    print(f"[VLA] bad state dim {states.shape}, expected {STATE_DIM}")
                    continue

                if step == 0 and first_actions is not None:
                    actions = first_actions
                    first_actions = None
                    print(f"[VLA] first frame using warmup chunk={actions.shape}")
                else:
                    frame = camera.get_frame()
                    last_vis_frame = frame
                    if step == 0:
                        print(f"[VLA] first frame shape={frame.shape} dtype={frame.dtype}")
                    # Keepalive continues last token during this (possibly slow) HTTP call.
                    actions = policy.query_action(frame, states, args.instruction)

                if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                    print(f"[VLA] bad action shape {actions.shape}, expected (T, {ACTION_DIM})")
                    continue

                # Execute the full returned chunk at --freq (keepalive mirrors last action).
                for i in range(actions.shape[0]):
                    if not running.is_set():
                        break
                    t0 = time.perf_counter()
                    keepalive.set_action(actions[i])
                    pred_state29, post_state = execute_action68(
                        actions[i],
                        token_publisher,
                        state_sub=state_sub,
                        prev_body_q_target=last_body_q_target,
                        brainco_hand=brainco_hand,
                    )
                    if post_state is not None:
                        last_robot_state = post_state
                    if pred_state29 is not None:
                        last_body_q_target = pred_state29.copy()
                        if save_pred_state_dir is not None:
                            pred_state_buffer.append(pred_state29.astype(np.float32, copy=True))
                    elif save_pred_state_dir is not None and step % 10 == 0 and i == 0:
                        print("[VLA] warning: missing body_q_target for pred-state logging")
                    if save_pred_dir is not None:
                        pred_action_buffer.append(np.asarray(actions[i], dtype=np.float32).reshape(-1)[:ACTION_DIM])
                    if rerun_logger is not None and last_vis_frame is not None:
                        visualization_data(
                            rerun_idx,
                            _observation_for_rerun(last_vis_frame),
                            _state_for_rerun(states),
                            _action_for_rerun(actions[i]),
                            rerun_logger,
                        )
                        rerun_idx += 1
                    if step % 10 == 0 and i == 0:
                        _, l2, r2 = split_action68(actions[0])
                        print(
                            f"[VLA] step={step} chunk={actions.shape} "
                            f"token[:3]={actions[0, :3]} left_2d={l2} right_2d={r2}"
                        )
                    elapsed = time.perf_counter() - t0
                    sleep_t = dt - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
            except Exception as e:
                print(f"[VLA] step {step} failed: {e}")
                time.sleep(dt)
    finally:
        print("[MAIN] Shutting down...")
        running.clear()
        keepalive.stop()
        try:
            token_publisher.handoff_to_idle_planner(hold_seconds=2.0)
        except Exception as e:
            print(f"[MAIN] handoff failed: {e}")
        try:
            token_publisher.stop(send_stop=False)
        except Exception as e:
            print(f"[MAIN] publisher stop failed: {e}")
        camera.close()
        shutdown_brainco_hand(brainco_hand)
        state_sub.stop()
        if save_pred_dir is not None and pred_action_buffer:
            try:
                save_pred_action_trajectory(np.stack(pred_action_buffer, axis=0), save_pred_dir)
            except Exception as e:
                print(f"[SAVE] failed to save pred actions: {e}")
        elif save_pred_dir is not None:
            print("[SAVE] no pred actions recorded")
        if save_pred_state_dir is not None and pred_state_buffer:
            try:
                save_pred_state_trajectory(np.stack(pred_state_buffer, axis=0), save_pred_state_dir)
            except Exception as e:
                print(f"[SAVE] failed to save pred states: {e}")
        elif save_pred_state_dir is not None:
            print("[SAVE] no pred states recorded")
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    main()
