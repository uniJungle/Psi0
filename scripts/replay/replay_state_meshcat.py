#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from psi.utils import resolve_data_path

# Reuse the existing visualization helpers under scripts/viz.
import sys

_THIS_DIR = Path(__file__).resolve().parent
_VIZ_DIR = _THIS_DIR.parent / "viz"
sys.path.insert(0, str(_VIZ_DIR))

from fk import G1FK
from g1 import ARM_JOINT_NAMES, HAND_JOINT_NAMES, LEG_JOINT_NAMES


JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES + HAND_JOINT_NAMES

is_playing = False
current_frame = 0
play_speed = 1
suppress_joint_slider_events = False


@dataclasses.dataclass
class Args:
    data_dir: str = "/home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate/lerobot_v2.1"
    episode_idx: int = 0
    host: str = "0.0.0.0"
    port: int = 9000
    fps: float = 30.0
    urdf: str = "assets/robots/g1/g1_body29_hand14.urdf"
    mode: str = "state"

    def __post_init__(self) -> None:
        if not Path(self.data_dir).is_absolute():
            self.data_dir = str(resolve_data_path(self.data_dir))
        if not Path(self.urdf).is_absolute():
            self.urdf = str(resolve_data_path(self.urdf, auto_download=True))
        if self.mode not in ("state", "action"):
            raise ValueError("--mode must be 'state' or 'action'")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _matrix_to_quat(rot_matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf, joint_cfg: dict[str, float] | None = None
) -> tuple[dict[str, viser.GuiInputHandle[float]], dict[str, float]]:
    slider_handles: dict[str, viser.GuiInputHandle[float]] = {}
    initial_config: dict[str, float] = {}
    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = joint_cfg[joint_name] if joint_cfg and joint_name in joint_cfg else 0.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=float(np.clip(initial_pos, lower, upper)),
        )

        def _on_slider_update(_):
            global suppress_joint_slider_events
            if suppress_joint_slider_events:
                return
            viser_urdf.update_cfg({name: s.value for name, s in slider_handles.items()})  # type: ignore[arg-type]

        slider.on_update(_on_slider_update)
        slider_handles[joint_name] = slider
        initial_config[joint_name] = float(initial_pos)
    return slider_handles, initial_config


def _get_state_vector(frame: dict[str, Any]) -> np.ndarray:
    for key in ("states", "observation.state"):
        if key in frame:
            return _to_numpy(frame[key])
    raise KeyError("Expected 'states' or 'observation.state' in frame")


def _state_to_joint_cfg(state_vec: np.ndarray) -> dict[str, float]:
    joint_cfg = {name: 0.0 for name in JOINT_NAMES}

    if state_vec.size == 33:
        # Psi0 lerobot_v2.1 layout:
        # [0:15] lower | [15:22] larm | [22:29] rarm | [29:31] lhand(2) | [31:33] rhand(2)
        joint_cfg.update(dict(zip(LEG_JOINT_NAMES, state_vec[0:15].tolist())))
        joint_cfg.update(dict(zip(ARM_JOINT_NAMES[:7], state_vec[15:22].tolist())))
        joint_cfg.update(dict(zip(ARM_JOINT_NAMES[7:14], state_vec[22:29].tolist())))
        # Brainco 2D hand -> map to aux finger pair only.
        joint_cfg["left_hand_thumb_2_joint"] = float(state_vec[29])
        joint_cfg["left_hand_middle_1_joint"] = float(state_vec[30])
        joint_cfg["right_hand_thumb_2_joint"] = float(state_vec[31])
        joint_cfg["right_hand_middle_1_joint"] = float(state_vec[32])
        return joint_cfg

    if state_vec.size == 43:
        # Raw SONIC layout:
        # [0:15] lower | [15:22] larm | [22:29] lhand | [29:36] rarm | [36:43] rhand
        joint_cfg.update(dict(zip(LEG_JOINT_NAMES, state_vec[0:15].tolist())))
        joint_cfg.update(dict(zip(ARM_JOINT_NAMES[:7], state_vec[15:22].tolist())))
        joint_cfg.update(dict(zip(HAND_JOINT_NAMES[:7], state_vec[22:29].tolist())))
        joint_cfg.update(dict(zip(ARM_JOINT_NAMES[7:14], state_vec[29:36].tolist())))
        joint_cfg.update(dict(zip(HAND_JOINT_NAMES[7:14], state_vec[36:43].tolist())))
        return joint_cfg

    raise ValueError(f"Unsupported state dimension {state_vec.size}; expected 33 or 43")


def _get_action_joint_cfg(frame: dict[str, Any]) -> dict[str, float] | None:
    if "action" in frame:
        action_vec = _to_numpy(frame["action"])
        # Psi0 lerobot_v2.1 action = token64 + hand4; not directly visualizable as qpos.
        if action_vec.size in (68, 78):
            return None
        if action_vec.size in (33, 43):
            return _state_to_joint_cfg(action_vec)
    if "action.wbc" in frame:
        return _state_to_joint_cfg(_to_numpy(frame["action.wbc"]))
    return None


def main(args: Args) -> None:
    global current_frame, is_playing, play_speed, suppress_joint_slider_events

    from psi.data.lerobot.compat import LeRobotDataset

    server = viser.ViserServer(args.host, args.port)
    dataset = LeRobotDataset(args.data_dir, episodes=[args.episode_idx])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.add_notification(
            f"Loaded {len(dataset)} frames.",
            f"from {Path(args.data_dir).name}, episode {args.episode_idx}",
            with_close_button=True,
        )

    urdf = yourdfpy.URDF.load(args.urdf)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=True,
        load_collision_meshes=False,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    frame0 = dataset[current_frame]
    state_cfg0 = _state_to_joint_cfg(_get_state_vector(frame0))
    action_cfg0 = _get_action_joint_cfg(frame0)
    g1 = G1FK(args.urdf, mode="default")

    with server.gui.add_folder("Dataset Playback"):
        play_button = server.gui.add_button("Play/Pause")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(dataset) - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=1,
            max=50,
            step=1,
            initial_value=1,
        )
        mode_text = server.gui.add_markdown(
            "State replay only. `--mode action` will use action if it is joint-like; otherwise fallback to state."
        )

    with server.gui.add_folder("Joint position control"):
        slider_handles, initial_config = create_robot_control_sliders(server, viser_urdf, state_cfg0)

    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
        show_collision_meshes_cb = server.gui.add_checkbox("Show collision meshes", viser_urdf.show_collision)

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = True
    show_collision_meshes_cb.visible = False

    action_result0 = g1.fk(action_cfg0 if action_cfg0 is not None else state_cfg0)
    state_result0 = g1.fk(state_cfg0)

    l_action_handle = server.scene.add_icosphere(
        "/action/l_ee", radius=0.05, color=(1.0, 0.0, 0.0), position=tuple(action_result0["l_ee"]["position"])
    )
    r_action_handle = server.scene.add_icosphere(
        "/action/r_ee", radius=0.05, color=(1.0, 0.0, 0.0), position=tuple(action_result0["r_ee"]["position"])
    )
    l_state_handle = server.scene.add_frame(
        "/state/l_ee",
        wxyz=_matrix_to_quat(np.asarray(state_result0["l_ee"]["matrix"])[:3, :3]),
        position=tuple(state_result0["l_ee"]["position"]),
        axes_length=0.1,
        axes_radius=0.005,
    )
    r_state_handle = server.scene.add_frame(
        "/state/r_ee",
        wxyz=_matrix_to_quat(np.asarray(state_result0["r_ee"]["matrix"])[:3, :3]),
        position=tuple(state_result0["r_ee"]["position"]),
        axes_length=0.1,
        axes_radius=0.005,
    )

    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(0.0, 0.0, trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0),
    )

    reset_button = server.gui.add_button("Reset")

    def render_frame(frame_idx: int) -> None:
        global suppress_joint_slider_events

        frame = dataset[frame_idx]
        state_cfg = _state_to_joint_cfg(_get_state_vector(frame))
        action_cfg = _get_action_joint_cfg(frame)
        if args.mode == "action" and action_cfg is not None:
            display_cfg = action_cfg
        else:
            display_cfg = state_cfg

        suppress_joint_slider_events = True
        try:
            for jname, q in display_cfg.items():
                if jname in slider_handles:
                    slider_handles[jname].value = q
        finally:
            suppress_joint_slider_events = False

        viser_urdf.update_cfg(display_cfg)  # type: ignore[arg-type]

        state_fk = g1.fk(state_cfg)
        l_state_handle.position = tuple(state_fk["l_ee"]["position"])
        l_state_handle.wxyz = _matrix_to_quat(np.asarray(state_fk["l_ee"]["matrix"])[:3, :3])
        r_state_handle.position = tuple(state_fk["r_ee"]["position"])
        r_state_handle.wxyz = _matrix_to_quat(np.asarray(state_fk["r_ee"]["matrix"])[:3, :3])

        action_fk = g1.fk(action_cfg if action_cfg is not None else state_cfg)
        l_action_handle.position = tuple(action_fk["l_ee"]["position"])
        r_action_handle.position = tuple(action_fk["r_ee"]["position"])

    @speed_slider.on_update
    def _(_):
        global play_speed
        play_speed = int(speed_slider.value)

    @frame_slider.on_update
    def _(_):
        global current_frame
        current_frame = int(frame_slider.value)
        render_frame(current_frame)

    @play_button.on_click
    def _(event: viser.GuiEvent):
        global is_playing
        is_playing = not is_playing
        client = event.client
        assert client is not None
        client.add_notification(
            "Playing" if is_playing else "Paused",
            f"total frames: {len(dataset)}",
            with_close_button=True,
        )

    @reset_button.on_click
    def _(_):
        global suppress_joint_slider_events
        suppress_joint_slider_events = True
        try:
            for jname, init_q in initial_config.items():
                slider_handles[jname].value = init_q
        finally:
            suppress_joint_slider_events = False
        viser_urdf.update_cfg(initial_config)  # type: ignore[arg-type]

    mode_text.content = (
        "Showing `action` as robot pose." if args.mode == "action" else "Showing `state` as robot pose."
    )

    viser_urdf.update_cfg(initial_config)  # type: ignore[arg-type]
    render_frame(current_frame)

    while True:
        if is_playing:
            current_frame = (current_frame + play_speed) % len(dataset)
            frame_slider.value = current_frame
        time.sleep(1.0 / max(args.fps, 1.0))


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Replay lerobot states in meshcat/viser.")
    parser.add_argument("--data_dir", type=str, default=Args.data_dir)
    parser.add_argument("--episode_idx", type=int, default=Args.episode_idx)
    parser.add_argument("--host", type=str, default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--fps", type=float, default=Args.fps)
    parser.add_argument("--urdf", type=str, default=Args.urdf)
    parser.add_argument("--mode", type=str, default=Args.mode, choices=["state", "action"])
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main(_parse_args())
