"""Script to train RL agent with RSL-RL."""

import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from vovinamathlete_mjlab.tasks.tracking.mdp import MotionCommandCfg as TrackingMotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from vovinamathlete_mjlab.utils.print_config_summary import print_config_summary

_MOTION_COMMAND_CFG_TYPES = (TrackingMotionCommandCfg,)


def _describe_motion_source(motion_path: Path) -> str:
  if motion_path.is_dir():
    motion_count = len(sorted(motion_path.rglob("*.npz")))
    return f"folder with {motion_count} motion(s): {motion_path}"

  if motion_path.suffix in {".yaml", ".yml"}:
    from vovinamathlete_mjlab.utils.motion_dataset import _load_motion_entries_from_yaml

    entries = _load_motion_entries_from_yaml(motion_path)
    return f"dataset YAML with {len(entries)} motion(s): {motion_path}"

  return f"specific file: {motion_path}"


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = True
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  auto_resume_on_crash: bool = False
  """If True, automatically relaunch training (resuming from the latest
  checkpoint) when the process crashes -- e.g. a rare GPU driver watchdog
  kill from a pathological physics step. A crashed CUDA context can't be
  recovered in-place, so this re-invokes the script as a fresh subprocess."""
  max_crash_retries: int = 10
  """Cap on consecutive crash-relaunches before giving up (only used with
  auto_resume_on_crash)."""
  crash_retry_delay_s: float = 10.0
  """Delay before relaunching after a crash (only used with auto_resume_on_crash)."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], _MOTION_COMMAND_CFG_TYPES
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, _MOTION_COMMAND_CFG_TYPES)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Motion source is a {_describe_motion_source(motion_path)}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    print_config_summary(cfg.env)

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  print(
    f"[INFO] physics_dt={env.physics_dt:.6f}s ({1.0 / env.physics_dt:.1f} Hz)  "
    f"decimation={cfg.env.decimation}  "
    f"step_dt={env.step_dt:.6f}s  policy_freq={1.0 / env.step_dt:.1f} Hz"
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


# Set by _run_supervised on the subprocess it spawns, so that subprocess
# knows to train directly instead of spawning yet another supervisor layer.
_SUPERVISOR_CHILD_ENV_VAR = "_VOVINAMATHLETE_TRAIN_SUPERVISOR_CHILD"


def _run_supervised(argv: list[str], max_retries: int, retry_delay_s: float) -> int:
  """Re-invoke this script as a subprocess, auto-resuming from the latest
  checkpoint if it crashes.

  A crashed CUDA context (e.g. from a GPU driver watchdog kill) can't be
  recovered in the same process, so retrying means launching a fresh
  interpreter. Mirrors what scripts/train_resilient.sh used to do, but as a
  single self-contained flag on train.py.
  """
  child_env = {**os.environ, _SUPERVISOR_CHILD_ENV_VAR: "1"}
  attempt = 0
  while True:
    extra_args = ["--agent.resume", "true"] if attempt > 0 else []
    if attempt > 0:
      print(f"[train] Attempt {attempt}: resuming from latest checkpoint...", flush=True)
    result = subprocess.run(
      [sys.executable, __file__, *argv, *extra_args], env=child_env
    )
    if result.returncode == 0:
      print("[train] Training finished successfully.", flush=True)
      return 0

    attempt += 1
    print(
      f"[train] Training exited with code {result.returncode} "
      f"(attempt {attempt}/{max_retries}).",
      flush=True,
    )
    if attempt >= max_retries:
      print("[train] Max retries reached without a clean finish. Giving up.", flush=True)
      return result.returncode
    time.sleep(retry_delay_s)


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

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  is_supervisor_child = os.environ.get(_SUPERVISOR_CHILD_ENV_VAR) == "1"
  if args.auto_resume_on_crash and not is_supervisor_child:
    exit_code = _run_supervised(
      sys.argv[1:], args.max_crash_retries, args.crash_retry_delay_s
    )
    sys.exit(exit_code)

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
