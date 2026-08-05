"""Thin OpenPI WebSocket client for π0.5 SONIC (33D state / 68D action).

Uses in-repo ``openpi_client`` (msgpack_numpy + websockets.sync). Suitable for
open-loop (``.venv-openpi``) and closed-loop (teleop env after ``pip install websockets``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PSI0_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _PSI0_ROOT / "src" / "openpi" / "openpi-client" / "src"
if _OPENPI_CLIENT_SRC.is_dir() and str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

try:
    from openpi_client import websocket_client_policy as _ws
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Need openpi_client + websockets.\n"
        "  Open-loop: source .venv-openpi && pip install -e src/openpi/openpi-client\n"
        "  Closed-loop teleop env: pip install websockets\n"
        f"(openpi-client src expected at {_OPENPI_CLIENT_SRC})"
    ) from exc

STATE_DIM = 33
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2


class Pi05WebsocketClient:
    """Query π0.5 PolicyServer with mono RGB + flat 33D state."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        api_key: str | None = None,
        connect_timeout_s: float = 120.0,
    ):
        self.host = host
        self.port = port
        self._api_key = api_key
        self._policy: _ws.WebsocketClientPolicy | None = None
        self._connect(timeout_s=connect_timeout_s)

    def _connect(self, timeout_s: float) -> None:
        t0 = time.time()
        last_err: Exception | None = None
        while time.time() - t0 < timeout_s:
            try:
                self._policy = _ws.WebsocketClientPolicy(
                    host=self.host, port=self.port, api_key=self._api_key
                )
                return
            except Exception as exc:  # noqa: BLE001 — retry until timeout
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(
            f"Failed to connect to π0.5 server {self.host}:{self.port}: {last_err}"
        )

    def get_server_metadata(self) -> dict[str, Any]:
        assert self._policy is not None
        return self._policy.get_server_metadata()

    def ping(self) -> bool:
        try:
            _ = self.get_server_metadata()
            return True
        except Exception:
            return False

    def reset(self) -> None:
        assert self._policy is not None
        self._policy.reset()

    def query_action68(
        self,
        image_rgb: np.ndarray,
        states33: np.ndarray,
        instruction: str,
        *,
        reset: bool = False,
    ) -> np.ndarray:
        """Return action chunk ``(T, 68)``."""
        assert self._policy is not None
        img = np.asarray(image_rgb, dtype=np.uint8)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"image must be HWC uint8 RGB, got {img.shape} {img.dtype}")
        states = np.asarray(states33, dtype=np.float32).reshape(-1)
        if states.size < STATE_DIM:
            raise ValueError(f"state dim {states.size} < {STATE_DIM}")
        states = states[:STATE_DIM]

        obs: dict[str, Any] = {
            "observation/image": img,
            "states": states,
            "prompt": str(instruction),
        }
        if reset:
            obs["reset"] = True

        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM:
            raise RuntimeError(
                f"bad action shape {actions.shape}, expected (T, >={ACTION_DIM})"
            )
        return actions[:, :ACTION_DIM].astype(np.float32, copy=False)

    def close(self) -> None:
        self._policy = None
