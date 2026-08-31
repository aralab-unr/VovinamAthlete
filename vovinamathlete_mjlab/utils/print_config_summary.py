from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


def _unwrap_actuator(act):
  base = getattr(act, "base_cfg", None)
  return base if base is not None else act


def _format_value(value):
  cls_name = type(value).__name__
  if cls_name == "SceneEntityCfg":
    selectors = {
      field: getattr(value, field)
      for field in (
        "joint_names", "body_names", "geom_names", "site_names",
        "actuator_names", "tendon_names",
      )
      if getattr(value, field) is not None
    }
    return f"SceneEntityCfg(name={value.name!r}, {selectors})"
  return value


def print_config_summary(env_cfg: "ManagerBasedRlEnvCfg") -> None:
  print("\n" + "=" * 88)
  print("[CONFIG SUMMARY]")
  print("=" * 88)

  joint_pos_action = env_cfg.actions.get("joint_pos")
  print("\nAction scale:")
  if joint_pos_action is None:
    print("  (no 'joint_pos' action term found)")
  else:
    scale = joint_pos_action.scale
    if isinstance(scale, dict):
      for pattern, value in scale.items():
        print(f"  {pattern:35s} {value:.6f}")
    else:
      print(f"  flat scale = {scale}")

  robot_cfg = env_cfg.scene.entities.get("robot")
  print("\nDefault pose (init_state.joint_pos):")
  if robot_cfg is None:
    print("  (no 'robot' entity found)")
  else:
    joint_pos = robot_cfg.init_state.joint_pos
    if not joint_pos:
      print("  (none set; falls back to model keyframe)")
    else:
      for pattern, value in joint_pos.items():
        print(f"  {pattern:35s} {value:.6f}")

  print("\nActuator groups (kp / kd / armature / frictionloss / effort_limit):")
  if robot_cfg is None or robot_cfg.articulation is None:
    print("  (no articulated 'robot' entity found)")
  else:
    for act in robot_cfg.articulation.actuators:
      base = _unwrap_actuator(act)
      targets = ", ".join(base.target_names_expr)
      kp = getattr(base, "stiffness", None)
      kd = getattr(base, "damping", None)
      armature = getattr(base, "armature", None)
      frictionloss = getattr(base, "frictionloss", None)
      effort_limit = getattr(base, "effort_limit", None)
      print(f"  {targets}")
      print(
        f"    kp={kp}  kd={kd}  armature={armature}  "
        f"frictionloss={frictionloss}  effort_limit={effort_limit}"
      )

  print("\nDomain randomization / events:")
  if not env_cfg.events:
    print("  (no events configured)")
  else:
    for name, ev in env_cfg.events.items():
      func_name = getattr(ev.func, "__name__", str(ev.func))
      print(f"  {name}  (mode={ev.mode}, func={func_name})")
      for key, value in ev.params.items():
        print(f"    {key} = {_format_value(value)}")

  print("=" * 88 + "\n")
