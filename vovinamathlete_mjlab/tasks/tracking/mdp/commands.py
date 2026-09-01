from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _resolve_motion_files(motion_file: str) -> list[Path]:
  p = Path(motion_file)
  if p.is_dir():
    files = sorted(p.rglob("*.npz"))
    if not files:
      raise FileNotFoundError(f"No .npz files found under directory: {p}")
    return files
  if p.suffix in {".yaml", ".yml"}:
    from vovinamathlete_mjlab.utils.motion_dataset import _load_motion_entries_from_yaml

    entries = _load_motion_entries_from_yaml(p)
    return [Path(e.path) for e in entries]
  return [p]


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    npz_paths = _resolve_motion_files(motion_file)

    jpos_list, jvel_list = [], []
    bpos_list, bquat_list, blv_list, bav_list = [], [], [], []
    clip_lengths = []
    # Only the tracked-body subset of lin/ang vel is ever read (see
    # body_lin_vel_w/body_ang_vel_w below), so slice down to it before it
    # ever becomes a tensor -- the untracked bodies' velocities are never
    # materialized, on top of storing everything as fp16 to fit the corpus
    # in memory. Callers cast back to float32 right after gathering the
    # (tiny, per-env) active batch, so nothing downstream ever sees fp16.
    body_indexes_np = body_indexes.cpu().numpy()
    for p in npz_paths:
      data = np.load(p)
      jpos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float16, device=device))
      jvel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float16, device=device))
      bpos_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float16, device=device))
      bquat_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float16, device=device))
      blv_list.append(
        torch.tensor(
          data["body_lin_vel_w"][:, body_indexes_np], dtype=torch.float16, device=device
        )
      )
      bav_list.append(
        torch.tensor(
          data["body_ang_vel_w"][:, body_indexes_np], dtype=torch.float16, device=device
        )
      )
      clip_lengths.append(data["joint_pos"].shape[0])

    self.joint_pos = torch.cat(jpos_list, dim=0)
    self.joint_vel = torch.cat(jvel_list, dim=0)
    self._body_pos_w = torch.cat(bpos_list, dim=0)
    self._body_quat_w = torch.cat(bquat_list, dim=0)
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = torch.cat(blv_list, dim=0)
    self.body_ang_vel_w = torch.cat(bav_list, dim=0)
    self.time_step_total = self.joint_pos.shape[0]

    self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=device)
    self.clip_starts = torch.cumsum(self.clip_lengths, dim=0) - self.clip_lengths

  def clip_index_for_step(self, steps: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(self.clip_starts, steps, right=True) - 1

    if len(npz_paths) > 1:
      print(
        f"[MotionLoader] Loaded {len(npz_paths)} clips "
        f"({self.time_step_total} total frames) from {motion_file}"
      )


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.env_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.speed_scale = torch.ones(self.num_envs, device=self.device)
    self.frame_phase = torch.zeros(self.num_envs, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    if self.cfg.gravity_curriculum:
      self._true_gravity = env.sim.model.opt.gravity[0].clone()
      self._gravity_scale = self.cfg.gravity_curriculum_start_frac
      self._set_gravity(self._true_gravity * self._gravity_scale)
      self.metrics["gravity_scale"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    self._fk_model: mujoco.MjModel | None = None
    self._fk_data: mujoco.MjData | None = None

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps].float()

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps].float() * self.speed_scale.unsqueeze(-1)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps].float()
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps].float()

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return (
      self.motion.body_lin_vel_w[self.time_steps].float() * self.speed_scale.view(-1, 1, 1)
    )

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return (
      self.motion.body_ang_vel_w[self.time_steps].float() * self.speed_scale.view(-1, 1, 1)
    )

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index].float()
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index].float()

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return (
      self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index].float()
      * self.speed_scale.unsqueeze(-1)
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return (
      self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index].float()
      * self.speed_scale.unsqueeze(-1)
    )

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    if self.cfg.gravity_curriculum:
      self.metrics["gravity_scale"][:] = self._gravity_scale

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _min_body_z_fk(
    self, root_pos: torch.Tensor, root_ori: torch.Tensor, joint_pos: torch.Tensor
  ) -> torch.Tensor:
    if self._fk_model is None:
      self._fk_model = self._env.sim.mj_model
      self._fk_data = mujoco.MjData(self._fk_model)
    model, data = self._fk_model, self._fk_data
    assert data is not None

    indexing = self.robot.indexing
    free_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_adr = indexing.joint_q_adr.cpu().numpy()
    body_ids = indexing.body_ids.cpu().numpy()

    root_pos_np = root_pos.detach().cpu().numpy()
    root_ori_np = root_ori.detach().cpu().numpy()
    joint_pos_np = joint_pos.detach().cpu().numpy()

    n = root_pos_np.shape[0]
    min_z = np.empty(n, dtype=np.float32)
    for i in range(n):
      data.qpos[:] = 0.0
      data.qpos[free_adr[0:3]] = root_pos_np[i]
      data.qpos[free_adr[3:7]] = root_ori_np[i]
      data.qpos[joint_adr] = joint_pos_np[i]
      mujoco.mj_kinematics(model, data)
      min_z[i] = data.xpos[body_ids, 2].min()
    return torch.tensor(min_z, device=self.device, dtype=torch.float32)

  def _set_gravity(self, value: torch.Tensor) -> None:
    gravity_field = self._env.sim.model.opt.gravity
    wp.to_torch(gravity_field.wp_array)[:] = value

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.gravity_curriculum:
      reset_time_outs = getattr(self._env, "reset_time_outs", None)
      if reset_time_outs is not None:
        num_successes = int(reset_time_outs[env_ids].sum().item())
        if num_successes > 0:
          self._gravity_scale = min(
            self._gravity_scale
            + num_successes * self.cfg.gravity_curriculum_step_frac,
            self.cfg.gravity_curriculum_end_frac,
          )
          self._set_gravity(self._true_gravity * self._gravity_scale)

    if self.cfg.sampling_mode == "start":
      num_clips = self.motion.clip_starts.shape[0]
      clip_idx = torch.randint(0, num_clips, (len(env_ids),), device=self.device)
      self.time_steps[env_ids] = self.motion.clip_starts[clip_idx]
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    self.env_clip[env_ids] = self.motion.clip_index_for_step(self.time_steps[env_ids])
    self.speed_scale[env_ids] = sample_uniform(
      self.cfg.speed_scale_range[0],
      self.cfg.speed_scale_range[1],
      (len(env_ids),),
      device=self.device,
    )
    self.frame_phase[env_ids] = self.time_steps[env_ids].float()

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]

    if self.cfg.fall_tilt_deg_range is not None:
      n = len(env_ids)
      lo = math.radians(self.cfg.fall_tilt_deg_range[0])
      hi = math.radians(self.cfg.fall_tilt_deg_range[1])
      tilt_angle = sample_uniform(lo, hi, (n,), device=self.device)
      if self.cfg.fall_azimuth_deg_choices is not None:
        choices_rad = torch.tensor(
          [math.radians(d) for d in self.cfg.fall_azimuth_deg_choices],
          device=self.device,
        )
        pick = torch.randint(0, len(choices_rad), (n,), device=self.device)
        azimuth = choices_rad[pick]
      else:
        azimuth = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
      tilt_axis = torch.stack(
        [azimuth.cos(), azimuth.sin(), torch.zeros_like(azimuth)], dim=-1
      )
      tilt_quat = quat_from_angle_axis(tilt_angle, tilt_axis)
      z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
      forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(n, -1)

      recorded_forward_world = quat_apply(root_ori[env_ids], forward_local)
      recorded_heading = torch.atan2(
        recorded_forward_world[:, 1], recorded_forward_world[:, 0]
      )
      target_heading = recorded_heading + rand_samples[:, 5]

      tilt_forward_world = quat_apply(tilt_quat, forward_local)
      tilt_heading = torch.atan2(tilt_forward_world[:, 1], tilt_forward_world[:, 0])
      yaw_spin_quat = quat_from_angle_axis(target_heading - tilt_heading, z_axis)
      new_root_ori = quat_mul(yaw_spin_quat, tilt_quat)
    else:
      orientations_delta = quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
      )
      new_root_ori = quat_mul(orientations_delta, root_ori[env_ids])
    root_ori[env_ids] = new_root_ori

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )

    _ground_margin = 0.05
    if self.cfg.ground_clearance_mode == "always_ground":
      min_actual_z = self._min_body_z_fk(
        root_pos[env_ids], root_ori[env_ids], joint_pos[env_ids]
      )
      root_pos[env_ids, 2] = root_pos[env_ids, 2] + (_ground_margin - min_actual_z)
    else:
      recorded_root_pos = self.motion._body_pos_w[self.time_steps[env_ids], 0].float()
      recorded_root_quat = self.motion._body_quat_w[self.time_steps[env_ids], 0].float()
      all_body_pos = self.motion._body_pos_w[self.time_steps[env_ids]].float()
      rel_offsets = quat_apply_inverse(
        recorded_root_quat[:, None, :].expand(-1, all_body_pos.shape[1], -1),
        all_body_pos - recorded_root_pos[:, None, :],
      )
      new_world_offsets_z = quat_apply(
        new_root_ori[:, None, :].expand(-1, all_body_pos.shape[1], -1), rel_offsets
      )[..., 2]
      min_new_z_offset = new_world_offsets_z.min(dim=1).values
      required_root_z = _ground_margin - min_new_z_offset
      root_pos[env_ids, 2] = torch.maximum(root_pos[env_ids, 2], required_root_z)

    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    if self.cfg.freeze_steps > 0:
      advancing = (self._env.episode_length_buf >= self.cfg.freeze_steps).float()
      self.frame_phase += self.speed_scale * advancing
    else:
      self.frame_phase += self.speed_scale
    self.time_steps = self.frame_phase.long()
    clip_end_exclusive = (
      self.motion.clip_starts[self.env_clip] + self.motion.clip_lengths[self.env_clip]
    )
    env_ids = torch.where(self.time_steps >= clip_end_exclusive)[0]
    if env_ids.numel() > 0:
      if self.cfg.hold_at_end:
        held = (clip_end_exclusive[env_ids] - 1).float()
        self.time_steps[env_ids] = held.long()
        self.frame_phase[env_ids] = held
      else:
        self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  ground_clearance_mode: Literal["lift_only", "always_ground"] = "lift_only"
  freeze_steps: int = 0
  fall_tilt_deg_range: tuple[float, float] | None = None
  fall_azimuth_deg_choices: tuple[float, ...] | None = None
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  hold_at_end: bool = False
  speed_scale_range: tuple[float, float] = (1.0, 1.0)
  gravity_curriculum: bool = False
  gravity_curriculum_start_frac: float = 0.3
  gravity_curriculum_end_frac: float = 1.05
  gravity_curriculum_step_frac: float = 0.0001

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
