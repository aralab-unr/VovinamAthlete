from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def apply_body_drag_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  linear_drag: float,
  quadratic_drag: float,
  max_force: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  body_ids = asset_cfg.body_ids
  vel = asset.data.body_link_lin_vel_w[env_ids][:, body_ids]
  speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
  force = -linear_drag * vel - quadratic_drag * speed * vel

  force_norm = torch.linalg.norm(force, dim=-1, keepdim=True)
  scale = torch.clamp(max_force / (force_norm + 1e-8), max=1.0)
  force = force * scale

  torques = torch.zeros_like(force)
  asset.write_external_wrench_to_sim(force, torques, env_ids=env_ids, body_ids=body_ids)


def randomize_physics_scene_gravity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  gravity_distribution_params: tuple[list[float], list[float]],
  operation: str = "abs",
  distribution: str = "uniform",
) -> None:
  if operation != "abs" or distribution != "uniform":
    raise NotImplementedError(
      "randomize_physics_scene_gravity only supports operation='abs', "
      "distribution='uniform'."
    )
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  low = torch.tensor(gravity_distribution_params[0], device=env.device)
  high = torch.tensor(gravity_distribution_params[1], device=env.device)
  samples = sample_uniform(low, high, (len(env_ids), 3), device=env.device)
  env.sim.model.opt.gravity[env_ids] = samples
