"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from vovinamathlete_mjlab.tasks.tracking.mdp import MotionCommandCfg as TrackingMotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from vovinamathlete_mjlab.utils.print_config_summary import print_config_summary

_MOTION_COMMAND_CFG_TYPES = (TrackingMotionCommandCfg,)

# geom groups defined in VD03.xml: group 2 = visual mesh, group 3 = collision
# primitives (spheres/cylinders/boxes approximating the mesh for physics).
_GEOM_GROUP_VISUAL = 2
_GEOM_GROUP_COLLISION = 3


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  view: Literal["visual", "collision"] = "visual"
  """Which geom group to render the robot with: 'visual' (the textured
  mesh, group 2) or 'collision' (the physics primitives -- spheres/
  cylinders/boxes -- used for contact, group 3). Both viewers also let you
  toggle groups live at runtime (native: mujoco's built-in Group Enable
  panel; viser: GUI checkboxes) -- this just sets which one is shown at
  launch."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  show_all_envs: bool = False
  """Show debug visualizations (e.g. the reference-motion ghost) for every
  env instead of just the focused one. Native viewer: also toggleable at
  runtime with the 'A' key. Viser viewer: also toggleable via its GUI
  checkbox."""
  export_onnx: bool = True
  """Export the loaded checkpoint's policy to ONNX (trained mode only), next to
  the checkpoint under an 'exported/' subfolder, e.g. playing
  logs/.../model_5000.pt exports logs/.../exported/model_5000.onnx."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], _MOTION_COMMAND_CFG_TYPES
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, _MOTION_COMMAND_CFG_TYPES)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, _MOTION_COMMAND_CFG_TYPES)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  print_config_summary(env_cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.view == "collision":
    # Shape shown is 100% the real visual mesh (group 2) -- the primitive
    # collision geoms (group 3: spheres/cylinders/boxes) are left hidden, not
    # rendered at all. Only a color tint distinguishes this from --view
    # visual, so it's still clear which mode is active.
    model = env.unwrapped.sim.mj_model
    model.geom_rgba[model.geom_group == _GEOM_GROUP_VISUAL] = [0.3, 0.55, 0.9, 1.0]

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    if cfg.export_onnx:
      assert log_dir is not None  # log_dir is set in TRAINED_MODE block
      export_dir = log_dir / "exported"
      export_filename = f"{resume_path.stem}.onnx"
      runner.export_policy_to_onnx(str(export_dir), export_filename)
      print(f"[INFO]: Exported policy to {export_dir / export_filename}")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":

    class _NativeViewerWithGeomView(NativeMujocoViewer):
      def setup(self) -> None:
        super().setup()
        assert self.vopt is not None and self.viewer is not None
        # Primitive collision geoms (group 3) are never rendered -- 'collision'
        # view is the real mesh (group 2), just tinted (see geom_rgba patch
        # above). self.vopt only affects the other (non-focused) envs,
        # rendered via _render_other_env_geoms; the focused env is rendered by
        # the passive viewer itself using its own separate self.viewer.opt --
        # both need setting or only N-1 of the N robots pick up the tint.
        self.vopt.geomgroup[_GEOM_GROUP_VISUAL] = 1
        self.vopt.geomgroup[_GEOM_GROUP_COLLISION] = 0
        self.viewer.opt.geomgroup[_GEOM_GROUP_VISUAL] = 1
        self.viewer.opt.geomgroup[_GEOM_GROUP_COLLISION] = 0

    native_viewer = _NativeViewerWithGeomView(env, policy)
    if cfg.show_all_envs:
      native_viewer._show_all_envs = True
    native_viewer.run()
  elif resolved_viewer == "viser":

    class _ViserPlayViewerCustom(ViserPlayViewer):
      def setup(self) -> None:
        super().setup()
        if cfg.show_all_envs:
          self._scene.show_all_envs = True
        self._scene.geom_groups_visible[_GEOM_GROUP_VISUAL] = True
        self._scene.geom_groups_visible[_GEOM_GROUP_COLLISION] = False

    _ViserPlayViewerCustom(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import vovinamathlete_mjlab.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
