"""Replay motion-capture NPZ data on the VD03 robot (kinematic, no physics)."""

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDict
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab.tasks  # noqa: F401  — registers built-in tasks
import vovinamathlete_mjlab.tasks  # noqa: F401  — registers VD03-* tasks
from vovinamathlete_mjlab.utils.motion_dataset import MotionFileEntry, _load_motion_entries_from_yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayConfig:
    motion_file: str = "vovinamathlete_mjlab/assets/motions/vd03/locomotion.npz"
    """NPZ file or YAML dataset config (relative to repo root or absolute).
    Ignored when --motion-dir is set."""
    motion_dir: str | None = None
    """Directory of .npz files. When set, each of the num_envs envs
    independently samples a random file from this folder and plays it on
    its own loop, instead of every env playing the same --motion-file."""
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    loop: bool = True
    """Loop back to frame 0 when a clip ends."""
    fix_xy: bool = True
    """Anchor XY at each env's spawn position; only height (Z) follows the NPZ."""
    speed: float = 1.0
    """Playback speed multiplier (1.0 = real-time)."""
    z_offset: float = -0.07
    """Vertical shift applied to every root position frame (metres).
    Use this to correct a systematic height bias in the NPZ data."""
    seed: int | None = None
    """Random seed for --motion-dir file assignment. None = nondeterministic."""


# ---------------------------------------------------------------------------
# Motion data loader
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _load_npz_files(motion_file: str) -> list[MotionFileEntry]:
    p = _resolve_path(motion_file)
    if p.suffix == ".npz":
        return [MotionFileEntry(str(p))]
    if p.suffix in {".yaml", ".yml"}:
        return _load_motion_entries_from_yaml(p)
    raise ValueError(f"Unsupported motion source: {p}  (expected .npz or .yaml)")


def _sample_motion_dir_entries(
    motion_dir: str, num_envs: int, seed: int | None
) -> list[MotionFileEntry]:
    """Pick one random .npz file per env from a directory (with replacement
    if there are fewer files than envs)."""
    d = _resolve_path(motion_dir)
    npz_files = sorted(d.rglob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found under {d}")
    rng = random.Random(seed)
    chosen = [rng.choice(npz_files) for _ in range(num_envs)]
    print(f"[ReplayMotion] Sampling {num_envs} env(s) from {len(npz_files)} files in {d}")
    for i, f in enumerate(chosen):
        print(f"  env {i}: {f.name}")
    return [MotionFileEntry(str(f)) for f in chosen]


class MotionClip:
    """Concatenated, device-resident motion data from one or more NPZ files."""

    def __init__(self, npz_files: list[MotionFileEntry], device: str) -> None:
        jpos_list, jvel_list, rpos_list, rquat_list, rlv_list, rav_list = [], [], [], [], [], []
        fpss: list[float] = []

        for entry in npz_files:
            f = entry.path
            d = np.load(f)
            fpss.append(float(np.asarray(d["fps"]).reshape(-1)[0]))
            jpos_list.append(torch.tensor(d["joint_pos"], dtype=torch.float32))
            jvel_list.append(torch.tensor(d["joint_vel"], dtype=torch.float32))
            rpos_list.append(torch.tensor(d["body_pos_w"][:, 0, :], dtype=torch.float32))
            rquat_list.append(torch.tensor(d["body_quat_w"][:, 0, :], dtype=torch.float32))
            if "body_lin_vel_w" in d:
                rlv_list.append(torch.tensor(d["body_lin_vel_w"][:, 0, :], dtype=torch.float32))
                rav_list.append(torch.tensor(d["body_ang_vel_w"][:, 0, :], dtype=torch.float32))

        self.joint_pos  = torch.cat(jpos_list,  dim=0).to(device)   # (T, J)
        self.joint_vel  = torch.cat(jvel_list,  dim=0).to(device)
        self.root_pos   = torch.cat(rpos_list,  dim=0).to(device)   # (T, 3)
        self.root_quat  = torch.cat(rquat_list, dim=0).to(device)   # (T, 4) wxyz

        T = self.joint_pos.shape[0]
        if rlv_list:
            self.root_lin_vel = torch.cat(rlv_list, dim=0).to(device)
            self.root_ang_vel = torch.cat(rav_list, dim=0).to(device)
        else:
            self.root_lin_vel = torch.zeros(T, 3, device=device)
            self.root_ang_vel = torch.zeros(T, 3, device=device)

        self.fps         = float(fpss[0]) if fpss else 50.0
        self.num_frames  = T

        names = ", ".join(Path(e.path).name for e in npz_files)
        print(f"[ReplayMotion] {len(npz_files)} file(s) ({names}) → {T} frames "
              f"@ {self.fps:.0f} fps ({T / self.fps:.1f}s total)")


# ---------------------------------------------------------------------------
# Kinematic replay environment wrapper
# ---------------------------------------------------------------------------

class KinematicReplayEnv:
    """Wraps RslRlVecEnvWrapper: replaces env.step() with state injection + forward kinematics.

    Each env holds its own clip (`clips[i]`) and advances/loops independently,
    so different envs can be mid-way through different clips of different
    lengths at the same time. Pass the same clip object for every env to
    replicate the old single-motion behavior.

    The viewer never sees physics glitches because we never call sim.step() —
    only sim.forward() (pure kinematics) after writing each NPZ frame.
    """

    def __init__(
        self,
        wrapped: RslRlVecEnvWrapper,
        clips: list[MotionClip],
        robot,
        fix_xy: bool,
        loop: bool,
        speed: float,
    ) -> None:
        self._env    = wrapped
        self._clips  = clips
        self._robot  = robot
        self._fix_xy = fix_xy
        self._loop   = loop
        self._device = str(wrapped.device)
        self._num_envs = wrapped.num_envs
        assert len(clips) == self._num_envs

        self._frame = [0] * self._num_envs

        # For fix_xy: capture spawn XY after the initial env reset.
        if fix_xy:
            self._spawn_xy = robot.data.root_link_pos_w[:, :2].clone()  # (N, 2)
            self._npz_origin_xy = torch.stack(
                [c.root_pos[0, :2] for c in clips], dim=0
            )  # (N, 2)

        # How many NPZ frames to advance per env.step() call. Assumes all
        # clips share the same fps (standard 50fps output across this repo's
        # conversion pipeline); uses env 0's clip if they don't.
        env_dt = wrapped.unwrapped.step_dt
        npz_dt = 1.0 / clips[0].fps
        self._frames_per_step = max(1, round(speed * env_dt / npz_dt))

        # Proxy viewer-required attributes.
        self.num_envs        = wrapped.num_envs
        self.num_actions     = wrapped.num_actions
        self.device          = wrapped.device
        self.max_episode_length = wrapped.max_episode_length
        self.cfg             = wrapped.cfg
        self.render_mode     = wrapped.render_mode
        self.observation_space = wrapped.observation_space
        self.action_space    = wrapped.action_space
        self.episode_length_buf = wrapped.episode_length_buf

    # --- viewer-required interface ----------------------------------------

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_observations(self) -> TensorDict:
        return self._env.get_observations()

    def reset(self):
        self._frame = [0] * self._num_envs
        return self._env.reset()

    def step(self, _actions: torch.Tensor):
        """Advance each env's NPZ frame independently, write state, run forward kinematics."""
        display_t = []
        for i in range(self._num_envs):
            end_frame = self._frame[i] + self._frames_per_step
            t = min(end_frame - 1, self._clips[i].num_frames - 1)
            display_t.append(t)

            self._frame[i] = end_frame
            if self._frame[i] >= self._clips[i].num_frames:
                if self._loop:
                    self._frame[i] = 0
                else:
                    self._frame[i] = self._clips[i].num_frames  # stay at end

        self._write_frame(display_t)
        self._env.unwrapped.sim.forward()

        if self._fix_xy:
            looped = [
                i for i in range(self._num_envs)
                if self._loop and self._frame[i] == 0
            ]
            if looped:
                idx = torch.tensor(looped, device=self._device, dtype=torch.long)
                self._spawn_xy[idx] = self._robot.data.root_link_pos_w[idx, :2].clone()

        N = self._num_envs
        dev = self._device
        dummy_obs = TensorDict({}, batch_size=[N], device=dev)
        rewards = torch.zeros(N, device=dev)
        dones   = torch.zeros(N, dtype=torch.long, device=dev)
        return dummy_obs, rewards, dones, {}

    def close(self) -> None:
        self._env.close()

    # --- internal ---------------------------------------------------------

    def _write_frame(self, t_per_env: list[int]) -> None:
        N   = self._num_envs
        dev = self._device

        pos    = torch.stack([self._clips[i].root_pos[t_per_env[i]] for i in range(N)], dim=0)
        quat   = torch.stack([self._clips[i].root_quat[t_per_env[i]] for i in range(N)], dim=0)
        lv     = torch.stack([self._clips[i].root_lin_vel[t_per_env[i]] for i in range(N)], dim=0)
        av     = torch.stack([self._clips[i].root_ang_vel[t_per_env[i]] for i in range(N)], dim=0)
        jpos   = torch.stack([self._clips[i].joint_pos[t_per_env[i]] for i in range(N)], dim=0)
        jvel   = torch.stack([self._clips[i].joint_vel[t_per_env[i]] for i in range(N)], dim=0)

        if self._fix_xy:
            xy_delta = pos[:, :2] - self._npz_origin_xy
            pos[:, :2] = self._spawn_xy + xy_delta

        # Root state (N, 13): [pos(3) | quat_wxyz(4) | lin_vel(3) | ang_vel(3)]
        root_state = torch.empty(N, 13, device=dev)
        root_state[:, :3]   = pos
        root_state[:, 3:7]  = quat
        root_state[:, 7:10] = lv
        root_state[:, 10:]  = av
        self._robot.write_root_state_to_sim(root_state)
        self._robot.write_joint_state_to_sim(jpos, jvel)


# ---------------------------------------------------------------------------
# Zero-action policy (obs are written by the env wrapper, not the policy)
# ---------------------------------------------------------------------------

class _ZeroPolicy:
    def __init__(self, num_envs: int, num_actions: int, device: str) -> None:
        self._zeros = torch.zeros(num_envs, num_actions, device=device)

    def __call__(self, obs) -> torch.Tensor:
        return self._zeros


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_replay(cfg: ReplayConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build the tracking env in play mode (no push events, no terrain
    # curriculum). Replay is purely kinematic (state injection + forward
    # kinematics, no sim.step()), so the task only needs to supply the
    # right robot entity.
    env_cfg = load_env_cfg("VD03-Tracking", play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.events = {}  # no domain randomization during replay

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)  # triggers initial reset

    robot = env.scene["robot"]

    # Load motion clip(s), one per env.
    if cfg.motion_dir is not None:
        entries = _sample_motion_dir_entries(cfg.motion_dir, cfg.num_envs, cfg.seed)
        # Cache by source file so identical repeats (num_envs > file count)
        # don't reload/reparse the same npz multiple times.
        cache: dict[str, MotionClip] = {}
        clips: list[MotionClip] = []
        for entry in entries:
            if entry.path not in cache:
                cache[entry.path] = MotionClip([entry], device)
            clips.append(cache[entry.path])
    else:
        npz_files = _load_npz_files(cfg.motion_file)
        clip = MotionClip(npz_files, device)
        clips = [clip] * cfg.num_envs

    # Build kinematic replay env.
    replay_env = KinematicReplayEnv(
        wrapped, clips, robot,
        fix_xy=cfg.fix_xy,
        loop=cfg.loop,
        speed=cfg.speed,
    )

    policy = _ZeroPolicy(cfg.num_envs, wrapped.num_actions, device)

    # Select viewer.
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved = "native" if has_display else "viser"
    else:
        resolved = cfg.viewer

    print(f"[ReplayMotion] Opening {resolved} viewer  "
          f"(loop={cfg.loop}, fix_xy={cfg.fix_xy}, speed={cfg.speed}x)")

    if resolved == "native":
        NativeMujocoViewer(replay_env, policy).run()
    elif resolved == "viser":
        ViserPlayViewer(replay_env, policy).run()
    else:
        raise RuntimeError(f"Unknown viewer: {resolved}")

    env.close()


def main() -> None:
    cfg = tyro.cli(ReplayConfig)
    run_replay(cfg)


if __name__ == "__main__":
    main()
