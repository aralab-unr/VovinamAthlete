from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SHOULDER_BODY_NAMES = ("left_shoulder_roll_link", "right_shoulder_roll_link")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_anchor_yaw_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  _, _, ref_yaw = euler_xyz_from_quat(command.anchor_quat_w)
  _, _, robot_yaw = euler_xyz_from_quat(command.robot_anchor_quat_w)
  yaw_error = torch.atan2(
    torch.sin(ref_yaw - robot_yaw), torch.cos(ref_yaw - robot_yaw)
  )
  return torch.exp(-torch.square(yaw_error) / std**2)


def motion_anchor_gravity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  if body_name is None:
    ref_quat_w = command.anchor_quat_w
    robot_quat_w = command.robot_anchor_quat_w
  else:
    body_index = command.cfg.body_names.index(body_name)
    ref_quat_w = command.body_quat_w[:, body_index]
    robot_quat_w = command.robot_body_quat_w[:, body_index]
  ref_gravity_b = quat_apply_inverse(ref_quat_w, asset.data.gravity_vec_w)
  robot_gravity_b = quat_apply_inverse(robot_quat_w, asset.data.gravity_vec_w)
  error = torch.sum(torch.square(ref_gravity_b - robot_gravity_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_body_angular_speed_excess_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_name: str,
  max_speed: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_index = command.cfg.body_names.index(body_name)
  angular_speed = torch.linalg.vector_norm(
    command.robot_body_ang_vel_w[:, body_index], dim=-1
  )
  return torch.square(torch.clamp(angular_speed - max_speed, min=0.0))


def motion_body_angular_speed_excess_relative_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_name: str,
  margin: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_index = command.cfg.body_names.index(body_name)
  robot_speed = torch.linalg.vector_norm(
    command.robot_body_ang_vel_w[:, body_index], dim=-1
  )
  ref_speed = torch.linalg.vector_norm(
    command.body_ang_vel_w[:, body_index], dim=-1
  )
  return torch.square(torch.clamp(robot_speed - (ref_speed + margin), min=0.0))


def motion_body_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.square(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_stand_still_joint_pos_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.1,
  lin_vel_threshold: float = 0.05,
  ang_vel_threshold: float = 0.1,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  standing_mask = (
    (torch.norm(command.anchor_lin_vel_w, dim=-1) < lin_vel_threshold)
    & (torch.norm(command.anchor_ang_vel_w, dim=-1) < ang_vel_threshold)
  ).float()
  error = torch.sum(
    torch.square(command.joint_pos - command.robot_joint_pos), dim=-1
  )
  return torch.exp(-error / std**2) * standing_mask


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force
  force_magnitude = torch.norm(forces, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def ground_contact_force_l2(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force
  force_magnitude = torch.norm(forces, dim=-1)
  return torch.sum(force_magnitude, dim=1)


def actuator_torque_balance_l2(
  env: ManagerBasedRlEnv,
  left_asset_cfg: SceneEntityCfg,
  right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[left_asset_cfg.name]
  left_effort = torch.sum(
    torch.abs(asset.data.actuator_force[:, left_asset_cfg.actuator_ids]), dim=1
  )
  right_effort = torch.sum(
    torch.abs(asset.data.actuator_force[:, right_asset_cfg.actuator_ids]), dim=1
  )
  return torch.square(left_effort - right_effort)


def actuator_stall_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  effort_limit: float,
  saturation_frac: float = 0.9,
  vel_eps: float = 0.1,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  saturated = torque.abs() > (saturation_frac * effort_limit)
  stalled = vel.abs() < vel_eps
  return torch.sum(torch.square(torque) * (saturated & stalled).float(), dim=1)


def penalty_electrical_power_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
  actuator_ids, _ = asset.find_actuators(asset_cfg.joint_names)

  tau = asset.data.actuator_force[:, actuator_ids]
  qd = asset.data.joint_vel[:, joint_ids]

  mech = -tau * qd - 150
  mech_pos = torch.clamp(mech, min=0.0)

  return torch.sum((mech_pos / 500) ** 2, dim=1)


def penalty_relative_shoulder_high(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  shoulder_idx = [command.cfg.body_names.index(n) for n in _SHOULDER_BODY_NAMES]
  return torch.sum(
    torch.square(
      command.body_pos_relative_w[:, shoulder_idx, 2]
      - command.robot_body_pos_w[:, shoulder_idx, 2]
    ),
    dim=-1,
  )


def penalty_relative_root_orientation(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, 0], command.robot_body_quat_w[:, 0]
    )
    ** 2
  )
  return error.squeeze(-1) if error.dim() > 1 else error


class penalty_xy_rate_before_stand:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._command_name = cfg.params["command_name"]
    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    self._shoulder_idx = [
      command.cfg.body_names.index(n) for n in _SHOULDER_BODY_NAMES
    ]
    self._prev_anchor_pos = command.robot_anchor_pos_w.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    command = cast(MotionCommand, self._env.command_manager.get_term(self._command_name))
    self._prev_anchor_pos[env_ids] = command.robot_anchor_pos_w[env_ids]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_threshold: float,
  ) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.norm(
      self._prev_anchor_pos[..., :2] - command.robot_anchor_pos_w[..., :2], dim=1
    )
    diff = torch.norm(
      command.body_pos_relative_w[:, self._shoulder_idx, 2]
      - command.robot_body_pos_w[:, self._shoulder_idx, 2],
      dim=-1,
    )
    result = torch.where(diff > stand_threshold, error, torch.zeros_like(error))
    self._prev_anchor_pos = command.robot_anchor_pos_w.clone()
    return result
