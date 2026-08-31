from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import vovinamathlete_mjlab.tasks.tracking.mdp as mdp
from vovinamathlete_mjlab.utils.dr import body_inertia_scale

VELOCITY_RANGE = {
  "x": (-0.75, 0.75),
  "y": (-0.75, 0.75),
  "z": (-0.4, 0.4),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_tracking_standing_env_cfg() -> ManagerBasedRlEnvCfg:
  _lookahead_steps = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}, history_length=5
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
      history_length=5,
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=5,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=5,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      params={"biased": True},
      history_length=5,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, history_length=5),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=5,
    ),
    "motion_command_lookahead": ObservationTermCfg(
      func=mdp.motion_command_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=1,
    ),
    "motion_anchor_pos_lookahead": ObservationTermCfg(
      func=mdp.motion_anchor_pos_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      noise=Unoise(n_min=-0.25, n_max=0.25),
      history_length=1,
    ),
    "motion_anchor_ori_lookahead": ObservationTermCfg(
      func=mdp.motion_anchor_ori_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=1,
    ),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}, history_length=5
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, history_length=5
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, history_length=5
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}, history_length=5
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}, history_length=5
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, history_length=5
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, history_length=5
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, history_length=5),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, history_length=5),
    "actions": ObservationTermCfg(func=mdp.last_action, history_length=5),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      history_length=5,
    ),
    "motion_command_lookahead": ObservationTermCfg(
      func=mdp.motion_command_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      history_length=1,
    ),
    "motion_anchor_pos_lookahead": ObservationTermCfg(
      func=mdp.motion_anchor_pos_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      history_length=1,
    ),
    "motion_anchor_ori_lookahead": ObservationTermCfg(
      func=mdp.motion_anchor_ori_lookahead,
      params={"command_name": "motion", "lookahead_steps": _lookahead_steps},
      history_length=1,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=None,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      nan_policy="sanitize",
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "motion": mdp.MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-1.57, 1.57),
        "pitch": (-1.57, 1.57),
        "yaw": (-1.0, 1.0),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-1.25, 1.25),
      gravity_curriculum=True,
      gravity_curriculum_start_frac=1.0,
      gravity_curriculum_end_frac=1.05,
      gravity_curriculum_step_frac=0.0000005,
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  events = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(00, 6.0),
      params={
        "velocity_range": {
          "x": (-0.75, 0.75),
          "y": (-0.75, 0.75),
          "z": (-0.25, 0.25),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(
            r"collision_left_ankle_roll_link_.*",
            r"collision_right_ankle_roll_link_.*",
          ),
        ),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.035, 0.035),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis_link", "torso_link")),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "inertia_randomize": EventTermCfg(
      mode="startup",
      func=body_inertia_scale,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "ranges": (0.7, 1.3),
      },
    ),
    "mass_randomize": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "ranges": (0.75, 1.25),
        "operation": "scale",
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "ranges": (0.75, 1.25),
        "operation": "scale",
      },
    ),
    "drag_force": EventTermCfg(
      mode="step",
      func=mdp.apply_body_drag_force,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "linear_drag": 2.0,
        "quadratic_drag": 0.0,
        "max_force": 25.0,
      },
    ),
    "leg_actuator_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 3]),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "upper_body_actuator_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[2, 4, 5]),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
  }

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "electrical_power_cost": RewardTermCfg(
      func=mdp.penalty_electrical_power_cost,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
    ),
    "penalty_relative_shoulder_high": RewardTermCfg(
      func=mdp.penalty_relative_shoulder_high,
      weight=-2.0,
      params={"command_name": "motion"},
    ),
    "penalty_relative_root_orientation": RewardTermCfg(
      func=mdp.penalty_relative_root_orientation,
      weight=-0.5,
      params={"command_name": "motion"},
    ),
    "penalty_xy_rate_before_stand": RewardTermCfg(
      func=mdp.penalty_xy_rate_before_stand,
      weight=-1.0,
      params={"command_name": "motion", "stand_threshold": 0.1},
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "tracking_failure": TerminationTermCfg(
      func=mdp.TolerantTermination(
        bad_tracking_time_threshold_s=4.0,
        command_name="motion",
        terms=[
          (
            "anchor_pos_z",
            mdp.bad_anchor_pos_z_only,
            {"command_name": "motion", "threshold": 0.35},
          ),
          (
            "anchor_ori",
            mdp.bad_anchor_ori,
            {
              "asset_cfg": SceneEntityCfg("robot"),
              "command_name": "motion",
              "threshold": 0.8,
            },
          ),
          (
            "ee_body_pos_z",
            mdp.bad_motion_body_pos_z_only,
            {"command_name": "motion", "threshold": 0.35, "body_names": ()},
          ),
        ],
      ),
      params={},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=10,
    episode_length_s=10.0,
  )
