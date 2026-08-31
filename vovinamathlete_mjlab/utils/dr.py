from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_inertia_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  entity_indices = asset.indexing.body_ids[asset_cfg.body_ids]
  n_bodies = len(entity_indices)

  default_inertia = env.sim.get_default_field("body_inertia")[entity_indices]

  low, high = ranges
  ratio = torch.empty(
    len(env_ids), n_bodies, device=env.device, dtype=default_inertia.dtype
  ).uniform_(low, high)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  env.sim.model.body_inertia[env_grid, entity_grid] = (
    default_inertia.unsqueeze(0) * ratio.unsqueeze(-1)
  )
