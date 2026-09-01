"""Convert motion-capture CSV files to NPZ format for tracking training.

Single file:

.. code-block:: bash

    python scripts/csv_to_npz.py \\
      --input-file vovinamathlete_mjlab/assets/motions/vd03/chienluocvd03.csv \\
      --output-name chienluocvd03.npz

Whole directory (batch), using many parallel envs per simulation step to
utilize the GPU instead of converting one frame at a time:

.. code-block:: bash

    python scripts/csv_to_npz.py \\
      --input-dir vovinamathlete_mjlab/assets/motions/vd03/csvdata \\
      --output-dir vovinamathlete_mjlab/assets/motions/vd03/npzdata \\
      --batch-size 2048 --device cuda:0

Single file with cleanup options, for noisy video-captured/GMR-retargeted
motion (see MotionLoader for the full set of --smooth/--auto-ground/etc.
flags):

.. code-block:: bash

    python scripts/csv_to_npz.py \\
      --input-file vovinamathlete_mjlab/assets/motions/vd03/csvdata/lie_down.csv \\
      --output-name lie_down.npz \\
      --smooth 11 --auto-ground 0.08 --calibrate-yaw-only --fix-rotation-drift

Converting boneseed-retargeter output (different CSV layout -- header row,
Frame column, centimetres, Euler-degree root rotation, degree joints; see
--csv-format soma in MotionLoader):

.. code-block:: bash

    python scripts/csv_to_npz.py --csv-format soma \\
      --input-dir vovinamathlete_mjlab/assets/motions/boneseedcsv \\
      --output-dir vovinamathlete_mjlab/assets/motions/vd03/npzdata \\
      --batch-size 2048 --device cuda:0 --input-fps 120

This script never opens a viewer/window, so it's headless by default — no
extra flags needed for running on a machine without a display.
"""

import math
import sys
from pathlib import Path
from typing import Any

# Ensures this project's own package is found first on sys.path, ahead of
# any similarly-named sibling project's editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import tyro
import os
from scipy.signal import savgol_filter
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from vovinamathlete_mjlab.tasks.tracking.config.vd03.env_cfgs import vd03_flat_tracking_env_cfg
from vovinamathlete_mjlab.assets.robots.vd03.vd03_constants import VD03_DFS_JOINT_NAMES
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  euler_xyz_from_quat,
  matrix_from_quat,
  quat_conjugate,
  quat_from_angle_axis,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  quat_slerp,
)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    csv_format: str = "raw",
    line_range: tuple[int, int] | None = None,
    smooth_window: int = 1,
    base_smooth_window: int = 0,
    loop_blend_frames: int = 0,
    zero_joint_indices: list[int] | None = None,
    set_joint_map: dict[int, float] | None = None,
    base_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lying_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lock_base_xy: bool = False,
    zero_base_roll: bool = False,
    auto_level_lying: bool = False,
    calibrate_base_rotation: bool = False,
    calibrate_yaw_only: bool = False,
    base_height_offset: float = 0.0,
    auto_ground: float | None = None,
    fix_rotation_drift: bool = False,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.device = device
    if csv_format not in ("raw", "soma", "boneseed"):
      raise ValueError(f"csv_format must be 'raw', 'soma', or 'boneseed', got {csv_format!r}")
    self.csv_format = csv_format
    self.line_range = line_range
    self.smooth_window = smooth_window
    self.base_smooth_window = base_smooth_window
    self.loop_blend_frames = loop_blend_frames
    self.zero_joint_indices = zero_joint_indices or []
    self.set_joint_map = set_joint_map or {}
    self.base_rotation_offset = base_rotation_offset
    self.lying_rotation_offset = lying_rotation_offset
    self.lock_base_xy = lock_base_xy
    self.zero_base_roll = zero_base_roll
    self.auto_level_lying = auto_level_lying
    self.calibrate_base_rotation = calibrate_base_rotation
    self.calibrate_yaw_only = calibrate_yaw_only
    self.base_height_offset = base_height_offset
    self.auto_ground = auto_ground
    self.fix_rotation_drift = fix_rotation_drift

    self._load_motion()
    self._interpolate_motion()
    self._smooth_motion()
    self._loop_blend()
    self._compute_velocities()

  # --- loading + cleanup ---------------------------------------------------

  def _load_motion(self):
    """Loads the motion from the csv file, then applies cleanup options in a
    fixed order (position/rotation offsets -> leveling/calibration ->
    ground/height -> drift report -> joint overrides)."""
    if self.csv_format in ("soma", "boneseed"):
      self._load_boneseed_csv()
    else:
      self._load_raw_csv()

    self.input_frames = self.motion_base_poss_input.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt
    print(
      f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, "
      f"frames: {self.input_frames}"
    )

    self._apply_base_rotation_offset()
    self._apply_lock_base_xy()
    self._apply_zero_base_roll()
    self._apply_auto_level_lying()
    self._apply_lying_rotation_offset()
    self._apply_calibrate_base_rotation()
    self._apply_calibrate_yaw_only()
    self._apply_auto_ground()
    self._apply_base_height_offset()
    self._report_and_fix_rotation_drift()
    self._apply_zero_and_set_joints()

  def _load_raw_csv(self):
    """--csv-format raw (default): headerless rows of
    [x, y, z, qx, qy, qz, qw, joint0, joint1, ...] -- position in metres,
    root rotation as an xyzw quaternion, joints in radians."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    self.motion_dof_poss_input = motion[:, 7:]

  def _load_boneseed_csv(self):
    """--csv-format soma/boneseed: the boneseed-retargeter CSV layout (also
    produced by soma_retargeter, same columns) -- a header row, then a
    leading Frame index, [root_translateX/Y/Z, root_rotateX/Y/Z, joint0,
    joint1, ...]. Position is in CENTIMETRES, root rotation is extrinsic-XYZ
    Euler DEGREES (not a quaternion), and joints are in DEGREES. Joint
    columns must already be in the robot's DFS joint order (this project's
    boneseed CSVs use '<joint_name>_dof' headers matching VD03_DFS_JOINT_NAMES)."""
    if self.line_range is not None:
      raise ValueError("--line-range is not supported with --csv-format soma/boneseed.")
    motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",", skiprows=1))
    motion = motion.to(torch.float32).to(self.device)

    self.motion_base_poss_input = motion[:, 1:4] * 0.01  # cm -> m
    euler_rad = torch.deg2rad(motion[:, 4:7])
    self.motion_base_rots_input = quat_from_euler_xyz(
      euler_rad[:, 0], euler_rad[:, 1], euler_rad[:, 2]
    )
    self.motion_dof_poss_input = torch.deg2rad(motion[:, 7:])

  def _euler_deg_to_quat(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> torch.Tensor:
    """Build a single (1, 4) wxyz quaternion from Euler angles in degrees."""
    roll = torch.tensor([math.radians(roll_deg)], device=self.device)
    pitch = torch.tensor([math.radians(pitch_deg)], device=self.device)
    yaw = torch.tensor([math.radians(yaw_deg)], device=self.device)
    return quat_from_euler_xyz(roll, pitch, yaw)

  def _apply_base_rotation_offset(self):
    """Constant rotation offset (degrees) applied to ALL frames. Fixes a
    systematic tilt from e.g. the video ground plane."""
    roll_deg, pitch_deg, yaw_deg = self.base_rotation_offset
    if roll_deg == 0.0 and pitch_deg == 0.0 and yaw_deg == 0.0:
      return
    offset = self._euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg)
    q = self.motion_base_rots_input
    q = quat_mul(offset.expand(q.shape[0], -1), q)  # pre-multiply: world-frame offset
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(
      f"  Base rotation offset applied: roll={roll_deg:+.1f}° "
      f"pitch={pitch_deg:+.1f}° yaw={yaw_deg:+.1f}°"
    )

  def _apply_lock_base_xy(self):
    """Fix base X,Y to the first frame value throughout. For stationary
    motions (lie-down, get-up) where video mocap drifts in XY."""
    if not self.lock_base_xy:
      return
    self.motion_base_poss_input[:, 0] = self.motion_base_poss_input[0, 0]
    self.motion_base_poss_input[:, 1] = self.motion_base_poss_input[0, 1]
    print(
      f"  Locked base XY to first frame: "
      f"x={self.motion_base_poss_input[0, 0].item():.4f}, "
      f"y={self.motion_base_poss_input[0, 1].item():.4f}"
    )

  def _apply_zero_base_roll(self):
    """Zero out the roll (X-axis rotation) of the base quaternion every
    frame. Fixes sideways tilt from video capture that causes arms to clip
    underground."""
    if not self.zero_base_roll:
      return
    roll, pitch, yaw = euler_xyz_from_quat(self.motion_base_rots_input)
    roll_before = roll.abs()
    zero = torch.zeros_like(roll)
    q = quat_from_euler_xyz(zero, pitch, yaw)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(
      f"  Zeroed base roll (was {roll_before.mean().item():.3f} rad avg, "
      f"max {roll_before.max().item():.3f} rad)"
    )

  def _apply_auto_level_lying(self):
    """Detect lying-down frames (base Z in bottom 30% of range), compute the
    average lateral tilt of the base in world space (gimbal-lock safe, works
    even at large pitch), and apply a single global rotation to remove it.
    Fixes the 'one arm underground one arm floating' problem from video
    mocap."""
    if not self.auto_level_lying:
      return
    z = self.motion_base_poss_input[:, 2]
    threshold = z.min() + (z.max() - z.min()) * 0.30
    lying_mask = z <= threshold
    if lying_mask.sum() < 3:
      print("  [WARN] auto_level_lying: not enough lying-down frames detected")
      return

    q_lying = self.motion_base_rots_input[lying_mask]
    q_avg = q_lying.mean(dim=0)
    q_avg = q_avg / q_avg.norm().clamp_min(1e-8)

    # Use body-Y (left-right axis): when lying on the back, body-X points
    # toward the ceiling so levelling it would wrongly tilt the robot;
    # body-Y is the lateral axis and should be horizontal when lying flat.
    # matrix_from_quat(q)[:, 1] (column 1) is body-Y expressed in world.
    by_world = matrix_from_quat(q_avg.unsqueeze(0))[0, :, 1]

    by_flat = torch.stack([by_world[0], by_world[1], torch.zeros((), device=self.device)])
    by_flat = by_flat / by_flat.norm().clamp_min(1e-6)

    cos_a = (by_world * by_flat).sum().clamp(-1.0, 1.0)
    axis = torch.linalg.cross(by_world, by_flat)
    axis = axis / axis.norm().clamp_min(1e-8)
    angle = torch.acos(cos_a)
    corr = quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0))[0]

    q = self.motion_base_rots_input
    q = quat_mul(corr.expand(q.shape[0], -1), q)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    tilt = math.degrees(math.asin(float(by_world[2].clamp(-1, 1))))
    print(
      f"  Auto-level lying (body-Y): tilt was {tilt:.1f}°, "
      f"corrected by {math.degrees(float(angle)):.1f}°"
    )

  def _apply_lying_rotation_offset(self):
    """Rotation offset (degrees) applied ONLY during the lying-down phase,
    blending smoothly to zero as the robot rises. Useful when
    --auto_level_lying is not enough and the lying pose still has a
    residual tilt."""
    roll_d, pitch_d, yaw_d = self.lying_rotation_offset
    if roll_d == 0.0 and pitch_d == 0.0 and yaw_d == 0.0:
      return
    offset_q = self._euler_deg_to_quat(roll_d, pitch_d, yaw_d)[0]
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    z_pos = self.motion_base_poss_input[:, 2]
    z_min, z_max = z_pos.min(), z_pos.max()
    z_range = (z_max - z_min).clamp_min(1e-4)
    z_norm = (z_pos - z_min) / z_range
    weight = torch.clamp(1.0 - z_norm * 2.0, 0.0, 1.0)
    weight = (1.0 - torch.cos(torch.pi * weight)) / 2.0  # smooth 0->1

    n = self.motion_base_rots_input.shape[0]
    corr_q = self._slerp(
      identity_q.unsqueeze(0).expand(n, -1),
      offset_q.unsqueeze(0).expand(n, -1),
      weight,
    )
    q = quat_mul(corr_q, self.motion_base_rots_input)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(
      f"  Lying rotation offset applied: roll={roll_d:+.1f}° pitch={pitch_d:+.1f}° "
      f"yaw={yaw_d:+.1f}° (max weight={weight.max().item():.2f} at lying frames)"
    )

  def _apply_calibrate_base_rotation(self):
    """Read the base link quaternion at frame 0, invert it, and pre-multiply
    all frames by that inverse so the motion starts at identity rotation."""
    if not self.calibrate_base_rotation:
      return
    q0_inv = quat_inv(self.motion_base_rots_input[0])
    q = quat_mul(q0_inv.unsqueeze(0).expand(self.motion_base_rots_input.shape[0], -1),
                 self.motion_base_rots_input)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print("  Calibrated base rotation: frame-0 zeroed to identity")

  def _apply_calibrate_yaw_only(self):
    """Remove only the yaw (Z-axis) offset from frame 0, keeping pitch/roll
    intact. Use instead of --calibrate_base_rotation for lying-down motions
    where pitch is intentional but the capture facing direction is wrong."""
    if not self.calibrate_yaw_only:
      return
    _, _, yaw0 = euler_xyz_from_quat(self.motion_base_rots_input[0:1])
    inv_yaw_q = self._euler_deg_to_quat(0.0, 0.0, math.degrees(-float(yaw0[0])))
    q = quat_mul(inv_yaw_q.expand(self.motion_base_rots_input.shape[0], -1),
                 self.motion_base_rots_input)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(f"  Calibrated yaw only: removed {math.degrees(float(yaw0[0])):.1f}° from frame-0 yaw")

  def _apply_auto_ground(self):
    """Auto-calibrate Z so the minimum base height in the motion equals
    --auto_ground (metres). Fixes video-captured motions where the ground
    level is inconsistent: the robot clips underground when lying down but
    floats when standing."""
    if self.auto_ground is None:
      return
    min_z = self.motion_base_poss_input[:, 2].min().item()
    correction = self.auto_ground - min_z
    self.motion_base_poss_input[:, 2] += correction
    max_z = self.motion_base_poss_input[:, 2].max().item()
    print(
      f"  Auto-ground calibration: min z was {min_z:.4f} m -> offset {correction:+.4f} m "
      f"(lying height = {self.auto_ground:.4f} m, standing height ~= {max_z:.4f} m)"
    )

  def _apply_base_height_offset(self):
    """Constant offset added to the base link z position (applied after
    --auto_ground)."""
    if self.base_height_offset == 0.0:
      return
    self.motion_base_poss_input[:, 2] += self.base_height_offset
    print(f"  Base height offset applied: {self.base_height_offset:+.4f} m")

  def _report_and_fix_rotation_drift(self):
    """Always report the rotation drift between the first and last frame;
    optionally (--fix_rotation_drift) spread a linearly-increasing
    correction across all frames so the motion ends at the same orientation
    it started, preventing a jump when looping."""
    q0 = self.motion_base_rots_input[0]
    qN = self.motion_base_rots_input[-1]
    delta = quat_mul(qN, quat_conjugate(q0))
    delta = delta / delta.norm().clamp_min(1e-8)
    drift_angle = 2.0 * math.acos(float(delta[0].clamp(-1.0, 1.0)))
    drift_deg = math.degrees(drift_angle)
    print(f"  Rotation drift (first->last): {drift_deg:.2f}°")

    if not self.fix_rotation_drift:
      return
    n = self.motion_base_rots_input.shape[0]
    inv_delta = quat_conjugate(delta)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    alphas = torch.linspace(0.0, 1.0, n, device=self.device)
    corr_q = self._slerp(
      identity.unsqueeze(0).expand(n, -1),
      inv_delta.unsqueeze(0).expand(n, -1),
      alphas,
    )
    # Post-multiply each frame: q_new[i] = q[i] * corr[i]
    q = quat_mul(self.motion_base_rots_input, corr_q)
    self.motion_base_rots_input = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(f"  Rotation drift corrected: spread {drift_deg:.2f}° correction linearly across {n} frames")

  def _apply_zero_and_set_joints(self):
    if self.zero_joint_indices:
      self.motion_dof_poss_input[:, self.zero_joint_indices] = 0.0
      print(f"  Zeroed joint indices: {self.zero_joint_indices}")
    if self.set_joint_map:
      for idx, val in self.set_joint_map.items():
        self.motion_dof_poss_input[:, idx] = val
      print(f"  Set joint values: {self.set_joint_map}")

  # --- interpolation / smoothing / looping ----------------------------------

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(
      index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device)
    )
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _smooth_motion(self):
    """Savitzky-Golay smoothing. --smooth (joints) and --base-smooth-window
    (base pos/rot, defaults to --smooth if 0) are independent since
    video-captured root motion is often much noisier than joint angles."""
    w_joint = self.smooth_window
    w_base = self.base_smooth_window if self.base_smooth_window > 0 else w_joint

    def _sg(tensor: torch.Tensor, w: int) -> torch.Tensor:
      if w <= 1:
        return tensor
      if w % 2 == 0:
        w += 1
      polyorder = min(3, w - 1)
      arr = savgol_filter(tensor.cpu().numpy(), window_length=w, polyorder=polyorder, axis=0)
      return torch.tensor(arr, dtype=tensor.dtype, device=tensor.device)

    self.motion_base_poss = _sg(self.motion_base_poss, w_base)
    smoothed_q = _sg(self.motion_base_rots, w_base)
    self.motion_base_rots = smoothed_q / smoothed_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    self.motion_dof_poss = _sg(self.motion_dof_poss, w_joint)

    if w_joint > 1 or w_base > 1:
      vel_before = self.motion_dof_poss.diff(dim=0).abs().max().item() * self.output_fps
      base_vel = self.motion_base_poss.diff(dim=0).abs().max().item() * self.output_fps
      print(
        f"  Smoothing: joints window={w_joint} (max vel spike {vel_before:.1f} rad/s), "
        f"base window={w_base} (max pos derivative {base_vel:.4f} m/s)"
      )

  def _loop_blend(self):
    """Crossfade the tail of the motion toward the head to remove the
    loop-point snatch. Over the last --loop_blend_frames frames a cosine
    weight rises from 0 -> 1, blending each channel from its current value
    toward the corresponding frame at the start of the clip."""
    k = self.loop_blend_frames
    if k <= 0:
      return
    k = min(k, self.output_frames // 2)

    t = torch.linspace(0.0, 1.0, k, device=self.device)
    w = (1.0 - torch.cos(torch.pi * t)) / 2.0

    w_j = w.unsqueeze(1)
    self.motion_dof_poss[-k:] = (1.0 - w_j) * self.motion_dof_poss[-k:] + w_j * self.motion_dof_poss[:k]
    self.motion_base_poss[-k:] = (1.0 - w_j) * self.motion_base_poss[-k:] + w_j * self.motion_base_poss[:k]
    self.motion_base_rots[-k:] = self._slerp(self.motion_base_rots[-k:], self.motion_base_rots[:k], w)

    print(
      f"  Loop blend applied: last {k} frames crossfaded toward first {k} frames "
      f"({k / self.output_fps:.2f} s window)"
    )

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega


def convert_file(
  sim: Simulation,
  scene: Scene,
  joint_names: list[str],
  input_file: str,
  input_fps: float,
  output_fps: float,
  output_path: str,
  line_range: tuple[int, int] | None,
  csv_format: str = "raw",
  smooth: int = 1,
  base_smooth_window: int = 0,
  loop_blend_frames: int = 0,
  zero_joints: list[str] | None = None,
  set_joint_values: list[str] | None = None,
  base_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
  lying_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
  lock_base_xy: bool = False,
  zero_base_roll: bool = False,
  auto_level_lying: bool = False,
  calibrate_base_rotation: bool = False,
  calibrate_yaw_only: bool = False,
  base_height_offset: float = 0.0,
  auto_ground: float | None = None,
  fix_rotation_drift: bool = False,
) -> None:
  """Convert one CSV motion to an npz, reusing the already-compiled scene/sim.

  Processes `scene.num_envs` frames per simulation step in parallel (each env
  holds one motion frame), instead of one frame at a time, so the GPU is
  actually doing batched work.
  """
  # Resolve --zero-joints names to column indices in the DFS joint list.
  zero_indices = []
  for name in zero_joints or []:
    if name in joint_names:
      zero_indices.append(joint_names.index(name))
    else:
      print(f"[WARN]: --zero-joints '{name}' not found in joint list, skipping.")

  # Resolve --set-joint-values (format: joint_name=value).
  set_joint_map: dict[int, float] = {}
  for item in set_joint_values or []:
    if "=" not in item:
      print(f"[WARN]: --set-joint-values '{item}' must be in JOINT_NAME=VALUE format, skipping.")
      continue
    name, _, raw_val = item.partition("=")
    if name not in joint_names:
      print(f"[WARN]: --set-joint-values '{name}' not found in joint list, skipping.")
      continue
    set_joint_map[joint_names.index(name)] = float(raw_val)

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    csv_format=csv_format,
    line_range=line_range,
    smooth_window=smooth,
    base_smooth_window=base_smooth_window,
    loop_blend_frames=loop_blend_frames,
    zero_joint_indices=zero_indices,
    set_joint_map=set_joint_map,
    base_rotation_offset=base_rotation_offset,
    lying_rotation_offset=lying_rotation_offset,
    lock_base_xy=lock_base_xy,
    zero_base_roll=zero_base_roll,
    auto_level_lying=auto_level_lying,
    calibrate_base_rotation=calibrate_base_rotation,
    calibrate_yaw_only=calibrate_yaw_only,
    base_height_offset=base_height_offset,
    auto_ground=auto_ground,
    fix_rotation_drift=fix_rotation_drift,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  if motion.motion_dof_poss.shape[1] != len(joint_names):
    raise ValueError(
      f"CSV has {motion.motion_dof_poss.shape[1]} joint columns, but the robot "
      f"expects {len(joint_names)} joints. Expected order: {joint_names}"
    )

  output_frames = motion.output_frames
  batch_size = scene.num_envs
  num_robot_joints = robot.data.joint_pos.shape[1]
  num_bodies = robot.data.body_link_pos_w.shape[1]

  log: dict[str, Any] = {
    "fps": np.asarray([output_fps], dtype=np.float32),
    "joint_pos": np.empty((output_frames, num_robot_joints), dtype=np.float32),
    "joint_vel": np.empty((output_frames, num_robot_joints), dtype=np.float32),
    "body_pos_w": np.empty((output_frames, num_bodies, 3), dtype=np.float32),
    "body_quat_w": np.empty((output_frames, num_bodies, 4), dtype=np.float32),
    "body_lin_vel_w": np.empty((output_frames, num_bodies, 3), dtype=np.float32),
    "body_ang_vel_w": np.empty((output_frames, num_bodies, 3), dtype=np.float32),
  }

  scene.reset()

  for start in range(0, output_frames, batch_size):
    end = min(start + batch_size, output_frames)
    frame_count = end - start
    frame_slice = slice(start, end)

    # Set root state (env-slot i holds frame start+i).
    root_states = robot.data.default_root_state.clone()
    root_states[:frame_count, 0:3] = motion.motion_base_poss[frame_slice]
    root_states[:frame_count, :2] += scene.env_origins[:frame_count, :2]
    root_states[:frame_count, 3:7] = motion.motion_base_rots[frame_slice]
    root_states[:frame_count, 7:10] = motion.motion_base_lin_vels[frame_slice]
    root_states[:frame_count, 10:] = motion.motion_base_ang_vels[frame_slice]
    robot.write_root_state_to_sim(root_states)

    # Set joint state.
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:frame_count, robot_joint_indexes] = motion.motion_dof_poss[frame_slice]
    joint_vel[:frame_count, robot_joint_indexes] = motion.motion_dof_vels[frame_slice]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()  # Kinematics only — no physics stepping.
    scene.update(sim.mj_model.opt.timestep)

    body_pos_w = robot.data.body_link_pos_w[:frame_count].detach().cpu().numpy().copy()
    body_pos_w -= scene.env_origins[:frame_count].detach().cpu().numpy()[:, None, :]

    log["joint_pos"][frame_slice] = robot.data.joint_pos[:frame_count].detach().cpu().numpy()
    log["joint_vel"][frame_slice] = robot.data.joint_vel[:frame_count].detach().cpu().numpy()
    log["body_pos_w"][frame_slice] = body_pos_w
    log["body_quat_w"][frame_slice] = robot.data.body_link_quat_w[:frame_count].detach().cpu().numpy()
    log["body_lin_vel_w"][frame_slice] = robot.data.body_link_lin_vel_w[:frame_count].detach().cpu().numpy()
    log["body_ang_vel_w"][frame_slice] = robot.data.body_link_ang_vel_w[:frame_count].detach().cpu().numpy()

    torch.testing.assert_close(
      robot.data.body_link_lin_vel_w[:frame_count, 0],
      motion.motion_base_lin_vels[frame_slice],
    )
    torch.testing.assert_close(
      robot.data.body_link_ang_vel_w[:frame_count, 0],
      motion.motion_base_ang_vels[frame_slice],
    )

  np.savez(output_path, **log)  # type: ignore[arg-type]


def main(
  input_file: str | None = None,
  output_name: str | None = None,
  input_dir: str | None = None,
  output_dir: str | None = None,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  csv_format: str = "raw",
  line_range: tuple[int, int] | None = None,
  batch_size: int = 2048,
  overwrite: bool = False,
  smooth: int = 1,
  base_smooth_window: int = 0,
  loop_blend_frames: int = 0,
  zero_joints: list[str] | None = None,
  set_joint_values: list[str] | None = None,
  base_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
  lying_rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
  lock_base_xy: bool = False,
  zero_base_roll: bool = False,
  auto_level_lying: bool = False,
  calibrate_base_rotation: bool = False,
  calibrate_yaw_only: bool = False,
  base_height_offset: float = 0.0,
  auto_ground: float | None = None,
  fix_rotation_drift: bool = False,
):
  """Convert motion CSV file(s) to npz.

  Args:
    input_file: Path to a single input CSV file (single-file mode).
    output_name: Path to the output npz file (single-file mode).
    input_dir: Directory searched recursively for '*.csv' files (batch mode).
    output_dir: Directory to write '<stem>.npz' files into (batch mode);
      also used as the default output location for single-file mode.
    input_fps: Frame rate of the CSV file(s).
    output_fps: Desired output frame rate.
    device: Device to use.
    csv_format: Layout of the input CSV file(s). 'raw' (default): headerless
      [x, y, z, qx, qy, qz, qw, joint...] rows, position in metres, root
      rotation as an xyzw quaternion, joints in radians. 'soma'/'boneseed':
      the boneseed-retargeter CSV layout -- header row, leading Frame index
      column, position in centimetres, root rotation as extrinsic-XYZ Euler
      degrees, joints in degrees. Not compatible with --line-range.
    line_range: Range of lines to process from the CSV file (single-file mode only).
    batch_size: Number of motion frames converted per simulation step. Higher
      utilizes more GPU parallelism at the cost of more device memory.
    overwrite: Batch mode: re-convert files whose output npz already exists.
    smooth: Savitzky-Golay smoothing window for joints (odd integer). 1 = no
      smoothing. Typical values: 11 for 30fps input, 21 for 50fps output.
    base_smooth_window: Savitzky-Golay window applied ONLY to base position
      and rotation, independent of --smooth. 0 = use the same window as
      --smooth. Use a larger value (e.g. 51, 101) for video-captured data
      where root motion is much noisier than joint angles.
    loop_blend_frames: Number of frames at the end of the motion to
      crossfade toward the beginning, so the loop plays without a snatch.
      0 = disabled. Typical: 10-30 frames at 50 fps (0.2-0.6 s). Requires
      that the first and last poses are already close.
    zero_joints: Joint names to force to 0 throughout the motion (e.g.
      left_hand_joint). Applied before interpolation and smoothing.
    set_joint_values: Set specific joints to a fixed value throughout the
      motion, e.g. ["left_wrist_pitch_joint=-0.3"]. Applied before
      interpolation and smoothing.
    base_rotation_offset: Apply a constant rotation offset (degrees) to the
      base quaternion for ALL frames: (roll, pitch, yaw). Use to correct a
      systematic tilt from the video ground plane.
    lying_rotation_offset: Apply a rotation offset (degrees) ONLY during the
      lying-down phase, blending smoothly to zero as the robot rises: (roll,
      pitch, yaw). Useful when --auto-level-lying is not enough.
    lock_base_xy: Fix base X,Y to the first frame value throughout the
      motion. Use for stationary motions (lie-down, get-up) where video
      mocap drifts in XY.
    zero_base_roll: Zero out the roll (X-axis rotation) of the base
      quaternion every frame. Fixes sideways tilt from video capture that
      causes arms to clip underground.
    auto_level_lying: Detect lying-down frames (base Z in bottom 30%% of
      range), compute the average lateral tilt of the base in world space,
      and apply a single global rotation to remove it.
    calibrate_base_rotation: Read the base link quaternion at frame 0,
      invert it, and pre-multiply all frames by that inverse so the motion
      starts at identity rotation (0,0,0).
    calibrate_yaw_only: Remove only the yaw (Z-axis) offset from frame 0.
      Keeps pitch and roll intact.
    base_height_offset: Constant offset added to the base link z position
      (metres). Use negative to lower the robot.
    auto_ground: Auto-calibrate Z so the minimum base height in the motion
      equals this value (metres). Applied before --base-height-offset.
    fix_rotation_drift: Compute the rotation difference between the first
      and last frame, then apply a linearly increasing correction across
      all frames so the motion ends at the same orientation it started.
  """
  if input_dir is not None:
    if output_dir is None:
      raise ValueError("--output-dir is required with --input-dir")
  elif input_file is not None:
    if output_name is None:
      output_name = str(Path(input_file).with_suffix(".npz").name)
  else:
    raise ValueError("Provide --input-file (single) or --input-dir/--output-dir (batch)")
  if batch_size <= 0:
    raise ValueError(f"--batch-size must be positive, got {batch_size}")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene_cfg: SceneCfg = vd03_flat_tracking_env_cfg().scene  # 27 Dof
  joint_names = list(VD03_DFS_JOINT_NAMES)
  default_output_dir = "./vovinamathlete_mjlab/assets/motions/vd03"

  final_output_dir = output_dir or default_output_dir
  os.makedirs(final_output_dir, exist_ok=True)

  # Build the scene/sim ONCE with num_envs=batch_size and reuse across every
  # file — recompiling per file (as single-file conversion effectively did
  # with num_envs=1 in a loop) would waste most of the wall-clock time on
  # repeated model compilation instead of GPU-parallel frame conversion.
  scene_cfg.num_envs = batch_size
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=batch_size, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  common_kwargs = dict(
    csv_format=csv_format,
    smooth=smooth,
    base_smooth_window=base_smooth_window,
    loop_blend_frames=loop_blend_frames,
    zero_joints=zero_joints,
    set_joint_values=set_joint_values,
    base_rotation_offset=base_rotation_offset,
    lying_rotation_offset=lying_rotation_offset,
    lock_base_xy=lock_base_xy,
    zero_base_roll=zero_base_roll,
    auto_level_lying=auto_level_lying,
    calibrate_base_rotation=calibrate_base_rotation,
    calibrate_yaw_only=calibrate_yaw_only,
    base_height_offset=base_height_offset,
    auto_ground=auto_ground,
    fix_rotation_drift=fix_rotation_drift,
  )

  if input_dir is not None:
    input_dir_path = Path(input_dir).expanduser().resolve()
    output_dir_path = Path(final_output_dir).expanduser().resolve()
    csv_files = sorted(input_dir_path.rglob("*.csv"))
    print(f"[INFO]: Found {len(csv_files)} CSV files under {input_dir_path}")
    converted = skipped = failed = 0
    pbar = tqdm(csv_files, desc="Converting", unit="file", ncols=100)
    for csv_path in pbar:
      npz_path = output_dir_path / f"{csv_path.stem}.npz"
      if npz_path.exists() and not overwrite:
        skipped += 1
        continue
      pbar.set_postfix_str(csv_path.name[:40])
      try:
        convert_file(
          sim=sim,
          scene=scene,
          joint_names=joint_names,
          input_file=str(csv_path),
          input_fps=input_fps,
          output_fps=output_fps,
          output_path=str(npz_path),
          line_range=None,
          **common_kwargs,
        )
        converted += 1
      except Exception as exc:  # one bad file must not abort the batch
        failed += 1
        print(f"\n[WARN]: FAILED {csv_path.name}: {exc}")
    print(f"[INFO]: Done. converted={converted}, skipped(existing)={skipped}, failed={failed}")
  else:
    assert input_file is not None and output_name is not None
    if not output_name.endswith(".npz"):
      output_name += ".npz"
    output_path = os.path.join(final_output_dir, output_name)
    convert_file(
      sim=sim,
      scene=scene,
      joint_names=joint_names,
      input_file=input_file,
      input_fps=input_fps,
      output_fps=output_fps,
      output_path=output_path,
      line_range=line_range,
      **common_kwargs,
    )
    print(f"[INFO]: Motion npz file saved to {output_path}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
