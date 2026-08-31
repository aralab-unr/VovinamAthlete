import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

_VD03_XML = Path(__file__).parent / "xml" / "RD03.xml"
assert _VD03_XML.exists(), f"VD03 XML not found: {_VD03_XML}"

VD03_DFS_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_hand_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_hand_joint",
)

VD03_DFS_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(VD03_DFS_JOINT_NAMES),
    preserve_order=True,
)

ACTUATOR_DELAY_RANGE = (6, 20)
_ACTUATOR_DELAY_UPDATE_PERIOD = 1_000_000


def _with_actuator_delay(base_cfg: BuiltinPositionActuatorCfg) -> DelayedActuatorCfg:
    return DelayedActuatorCfg(
        base_cfg=base_cfg,
        delay_target="position",
        delay_min_lag=ACTUATOR_DELAY_RANGE[0],
        delay_max_lag=ACTUATOR_DELAY_RANGE[1],
        delay_update_period=_ACTUATOR_DELAY_UPDATE_PERIOD,
        delay_per_env_phase=False,
    )


_ACT_LEG_STRONG = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r".*_hip_pitch_joint",
        r".*_hip_roll_joint",
        r".*_knee_joint",
    ),
    stiffness=167.476,
    damping=8.201,
    effort_limit=135.0,
    armature=0.025101925,
    frictionloss=0.01,
))

_ACT_HIP_YAW = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r".*_hip_yaw_joint",
    ),
    stiffness=130.181,
    damping=4.604,
    effort_limit=85.0,
    armature=0.01017752,
    frictionloss=0.01,
))

_ACT_WAIST_YAW = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r"waist_yaw_joint",
    ),
    stiffness=254.468,
    damping=5.627,
    effort_limit=85.0,
    armature=0.01017752,
    frictionloss=0.01,
))

_ACT_ANKLE = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r".*_ankle_pitch_joint",
        r".*_ankle_roll_joint",
    ),
    stiffness=48.167,
    damping=2.359,
    effort_limit=50.0,
    armature=0.00721945,
    frictionloss=0.01,
))

_ACT_ARM = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r".*_shoulder_pitch_joint",
        r".*_shoulder_roll_joint",
        r".*_shoulder_yaw_joint",
        r".*_elbow_joint",
        r".*_wrist_roll_joint",
    ),
    stiffness=82.084,
    damping=2.177,
    effort_limit=25.0,
    armature=0.003609725,
    frictionloss=0.01,
))

_ACT_HAND = _with_actuator_delay(BuiltinPositionActuatorCfg(
    target_names_expr=(
        r".*_wrist_pitch_joint",
        r".*_hand_joint",
    ),
    stiffness=37.751,
    damping=1.602,
    effort_limit=10.0,
    armature=0.00425,
    frictionloss=0.01,
))

VD03_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _ACT_LEG_STRONG,
        _ACT_HIP_YAW,
        _ACT_WAIST_YAW,
        _ACT_ANKLE,
        _ACT_ARM,
        _ACT_HAND,
    ),
    soft_joint_pos_limit_factor=0.9,
)

VD03_ACTION_SCALE: dict[str, float] = {}
for _act in VD03_ARTICULATION.actuators:
    if isinstance(_act, DelayedActuatorCfg):
        _act = _act.base_cfg
    assert isinstance(_act, BuiltinPositionActuatorCfg)
    assert _act.effort_limit is not None
    _scale = 0.25 * _act.effort_limit / _act.stiffness
    for _expr in _act.target_names_expr:
        VD03_ACTION_SCALE[_expr] = _scale


def get_spec(no_head: bool = False) -> mujoco.MjSpec:
    tree = ET.parse(str(_VD03_XML))
    root = tree.getroot()
    for act_elem in list(root.findall("actuator")):
        root.remove(act_elem)

    if no_head:
        head_elem = root.find(".//body[@name='head_link']")
        assert head_elem is not None, "RD03.xml has no body named head_link"
        parent_map = {child: parent for parent in root.iter() for child in parent}
        parent_map[head_elem].remove(head_elem)

    sensor_elem = root.find("sensor")
    assert sensor_elem is not None, "RD03.xml has no <sensor> block"
    for gyro in sensor_elem.findall("gyro"):
        if gyro.get("name") == "angular-velocity":
            gyro.set("name", "imu_ang_vel")
    for accel in sensor_elem.findall("accelerometer"):
        if accel.get("name") == "linear-acceleration":
            accel.set("name", "imu_lin_acc")
    if sensor_elem.find("velocimeter") is None:
        ET.SubElement(
            sensor_elem, "velocimeter", {"name": "imu_lin_vel", "site": "imu_sensor"}
        )
    if sensor_elem.find("subtreeangmom") is None:
        ET.SubElement(
            sensor_elem,
            "subtreeangmom",
            {"name": "root_angmom", "body": "pelvis_link"},
        )

    for side in ("left", "right"):
        body_name = f"{side}_ankle_roll_link"
        body_elem = root.find(f".//body[@name='{body_name}']")
        assert body_elem is not None, f"RD03.xml has no body named {body_name}"
        box_geom = None
        for geom in body_elem.findall("geom"):
            if geom.get("class") == "collision" and geom.get("type") == "box":
                box_geom = geom
                break
        assert box_geom is not None, f"{body_name} has no box collision geom"
        ET.SubElement(
            body_elem,
            "site",
            {"name": f"{side}_foot", "pos": box_geom.get("pos", "0 0 0")},
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=_VD03_XML.parent)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            tree.write(f, xml_declaration=True, encoding="unicode")
        spec = mujoco.MjSpec.from_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    collision_geom_counts: dict[str, int] = {}

    def _process(body: mujoco.MjsBody) -> None:
        for geom in body.geoms:
            if geom.group == 3:
                geom.rgba[3] = 0.0
                body_name = body.name
                idx = collision_geom_counts.get(body_name, 0)
                geom.name = f"collision_{body_name}_{idx}"
                collision_geom_counts[body_name] = idx + 1
        for child in body.bodies:
            _process(child)

    _process(spec.worldbody)
    return spec


_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.725),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "left_hip_pitch_joint": -0.26,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.52,
        "left_ankle_pitch_joint": -0.26,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.26,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.52,
        "right_ankle_pitch_joint": -0.26,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.26,
        "left_shoulder_roll_joint": 0.26,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.7,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_hand_joint": 0.0,
        "right_shoulder_pitch_joint": 0.26,
        "right_shoulder_roll_joint": -0.26,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.7,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_hand_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_vd03_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=_INIT_STATE,
        spec_fn=get_spec,
        articulation=VD03_ARTICULATION,
    )


def get_vd03_robot_cfg_no_head() -> EntityCfg:
    return EntityCfg(
        init_state=_INIT_STATE,
        spec_fn=lambda: get_spec(no_head=True),
        articulation=VD03_ARTICULATION,
    )

