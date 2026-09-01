from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.robot_body_lin_vel_w.view(env.num_envs, -1)


def robot_body_ang_vel_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.robot_body_ang_vel_w.view(env.num_envs, -1)


def _lookahead_clip_end(command: MotionCommand) -> torch.Tensor:
  return (
    command.motion.clip_starts[command.env_clip]
    + command.motion.clip_lengths[command.env_clip]
    - 1
  )


def _lookahead_frame(
  command: MotionCommand, k: int, clip_end: torch.Tensor
) -> torch.Tensor:
  warped_offset = (k * command.speed_scale).round().long()
  return torch.min(command.time_steps + warped_offset, clip_end)


def motion_command_lookahead(
  env: ManagerBasedRlEnv, command_name: str, lookahead_steps: list[int]
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  clip_end = _lookahead_clip_end(command)
  chunks = []
  for k in lookahead_steps:
    future_steps = _lookahead_frame(command, k, clip_end)
    chunks.append(command.motion.joint_pos[future_steps].float())
    chunks.append(
      command.motion.joint_vel[future_steps].float() * command.speed_scale.unsqueeze(-1)
    )
  return torch.cat(chunks, dim=-1)


def motion_anchor_pos_lookahead(
  env: ManagerBasedRlEnv, command_name: str, lookahead_steps: list[int]
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  clip_end = _lookahead_clip_end(command)
  chunks = []
  for k in lookahead_steps:
    future_steps = _lookahead_frame(command, k, clip_end)
    future_pos_w = (
      command.motion.body_pos_w[future_steps, command.motion_anchor_body_index].float()
      + env.scene.env_origins
    )
    future_quat_w = command.motion.body_quat_w[
      future_steps, command.motion_anchor_body_index
    ].float()
    pos, _ = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      future_pos_w,
      future_quat_w,
    )
    chunks.append(pos)
  return torch.cat(chunks, dim=-1)


def motion_anchor_ori_lookahead(
  env: ManagerBasedRlEnv, command_name: str, lookahead_steps: list[int]
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  clip_end = _lookahead_clip_end(command)
  chunks = []
  for k in lookahead_steps:
    future_steps = _lookahead_frame(command, k, clip_end)
    future_pos_w = (
      command.motion.body_pos_w[future_steps, command.motion_anchor_body_index].float()
      + env.scene.env_origins
    )
    future_quat_w = command.motion.body_quat_w[
      future_steps, command.motion_anchor_body_index
    ].float()
    _, ori_q = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      future_pos_w,
      future_quat_w,
    )
    chunks.append(matrix_from_quat(ori_q)[..., :2].reshape(env.num_envs, -1))
  return torch.cat(chunks, dim=-1)
