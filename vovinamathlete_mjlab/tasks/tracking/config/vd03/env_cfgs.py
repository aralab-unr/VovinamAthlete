from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from vovinamathlete_mjlab.assets.robots import get_vd03_robot_cfg
from vovinamathlete_mjlab.tasks.tracking.mdp import MotionCommandCfg
from vovinamathlete_mjlab.tasks.tracking.mdp.terminations import TolerantTermination
from vovinamathlete_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from vovinamathlete_mjlab.tasks.tracking.tracking_standing_env_cfg import (
  make_tracking_standing_env_cfg,
)

EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_hand_link",
  "right_hand_link",
)


def vd03_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128

  cfg.scene.entities = {"robot": get_vd03_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  head_ground_cfg = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="body", pattern="head_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg, head_ground_cfg)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_link",
  )

  motion_cmd.sampling_mode = "uniform"

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_hand_link",
    "right_hand_link",
  )

  cfg.viewer.body_name = "torso_link"

  cfg.terminations["nan_guard"] = TerminationTermCfg(func=envs_mdp.nan_detection)

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel", "motion_anchor_pos_lookahead"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=cfg.observations["actor"].history_length,
    )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

  return cfg


def vd03_make_tracking_standing_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_standing_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128

  cfg.scene.entities = {"robot": get_vd03_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  head_ground_cfg = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="body", pattern="head_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg, head_ground_cfg)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_link",
  )
  motion_cmd.sampling_mode = "uniform"

  tolerant = cfg.terminations["tracking_failure"].func
  assert isinstance(tolerant, TolerantTermination)
  for name, _func, params in tolerant.terms:
    if name == "ee_body_pos_z":
      params["body_names"] = EE_BODY_NAMES

  cfg.viewer.body_name = "torso_link"

  cfg.terminations["nan_guard"] = TerminationTermCfg(func=envs_mdp.nan_detection)

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel", "motion_anchor_pos_lookahead"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=cfg.observations["actor"].history_length,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.gravity_curriculum = False

  return cfg
