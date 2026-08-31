import dataclasses

from mjlab.tasks.registry import register_mjlab_task
from vovinamathlete_mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import vd03_flat_tracking_env_cfg, vd03_make_tracking_standing_env_cfg
from .rl_cfg import vd03_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="VD03-Tracking",
  env_cfg=vd03_flat_tracking_env_cfg(),
  play_env_cfg=vd03_flat_tracking_env_cfg(play=True),
  rl_cfg=vd03_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="VD03-Tracking-No-State-Estimation",
  env_cfg=vd03_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=vd03_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=vd03_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="VD03-Tracking-Standing",
  env_cfg=vd03_make_tracking_standing_env_cfg(),
  play_env_cfg=vd03_make_tracking_standing_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    vd03_tracking_ppo_runner_cfg(), experiment_name="vd03_tracking_standing"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
