"""Minimal ZMQ client for GR00T-N1.7 PolicyServer (msgpack_numpy).

Compatible with ``gr00t/policy/server_client.py`` wire format for numpy payloads.
Does not require importing the full ``gr00t`` package — usable from ACT / teleop envs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zmq

try:
    import msgpack
    import msgpack_numpy as mnp
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Need msgpack + msgpack_numpy for GR00T ZMQ client.\n"
        "  python -m pip install msgpack msgpack-numpy"
    ) from exc


STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 31),
    "right_hand": (31, 33),
}

ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
STATE_DIM = 33
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2


def split_state33(states: np.ndarray) -> dict[str, np.ndarray]:
    """Split flat 33D state → modality dict with shape (1, 1, D) for GR00T."""
    s = np.asarray(states, dtype=np.float32).reshape(-1)
    if s.size < STATE_DIM:
        raise ValueError(f"state dim {s.size} < {STATE_DIM}")
    out = {}
    for key, (a, b) in STATE_SLICES.items():
        out[key] = s[a:b][None, None, :].astype(np.float32)
    return out


def concat_action68(action_dict: dict[str, Any]) -> np.ndarray:
    """Concat GR00T action dict → (T, 68) array."""
    parts = []
    for key in ACTION_KEYS:
        if key not in action_dict:
            raise KeyError(f"missing action key {key!r}; have {list(action_dict)}")
        arr = np.asarray(action_dict[key], dtype=np.float32)
        # Accept (B,T,D) or (T,D) or (D,)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr[None, :]
        parts.append(arr)
    t = parts[0].shape[0]
    for p in parts[1:]:
        if p.shape[0] != t:
            raise ValueError(f"action horizon mismatch: {[x.shape for x in parts]}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def build_gr00t_obs(
    image_left: np.ndarray,
    image_right: np.ndarray,
    states33: np.ndarray,
    instruction: str,
) -> dict[str, Any]:
    """Build nested observation for UNITREE_G1_SONIC (stereo + 33D state)."""
    left = np.asarray(image_left, dtype=np.uint8)
    right = np.asarray(image_right, dtype=np.uint8)
    if left.ndim == 3:
        left = left[None, None, ...]  # (B=1, T=1, H, W, C)
    if right.ndim == 3:
        right = right[None, None, ...]
    return {
        "video": {
            "ego_view_left": left,
            "ego_view_right": right,
        },
        "state": split_state33(states33),
        "language": {
            "annotation.human.task_description": [[str(instruction)]],
        },
    }


class Gr00tZMQClient:
    """ZMQ REQ client talking to ``run_gr00t_server.py``."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._sock = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception as exc:
            print(f"[Gr00tZMQ] ping failed: {exc}")
            self._init_socket()
            return False

    def reset(self) -> Any:
        return self.call_endpoint("reset", {"options": None})

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self._sock.send(msgpack.packb(request, default=mnp.encode))
            message = self._sock.recv()
        except zmq.Again:
            self._init_socket()
            raise TimeoutError(f"GR00T server timeout on endpoint={endpoint}")
        if message == b"ERROR":
            raise RuntimeError("GR00T server returned ERROR")
        response = msgpack.unpackb(message, object_hook=mnp.decode, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def query_action68(
        self,
        image_left: np.ndarray,
        image_right: np.ndarray,
        states33: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Query policy and return (T, 68) action chunk."""
        obs = build_gr00t_obs(image_left, image_right, states33, instruction)
        response = self.call_endpoint(
            "get_action", {"observation": obs, "options": None}
        )
        # Server returns [action_dict, info] (tuple → list over msgpack)
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            action_dict = response[0]
        elif isinstance(response, dict) and any(k in response for k in ACTION_KEYS):
            action_dict = response
        else:
            raise RuntimeError(f"Unexpected get_action response type: {type(response)}")
        return concat_action68(action_dict)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()
