import robosuite as suite

env = suite.make(
    "Lift",
    robots=["Panda"],
    has_renderer=False,
    has_offscreen_renderer=False,
    render_visual_mesh=False,
    render_collision_mesh=False,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=100,

)
import collections
from dm_control import mjcf
from dm_control import composer, viewer
from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.manipulation.shared import observations

import os


EYE_IN_HAND_VIEW = "robot0_eye_in_hand"

TASK_VIEW = "robot0_robotview"
FRONT_VIEW = "frontview"

_ROPE_XML_PATH = os.path.abspath("models/rope/rope.xml")

_CAMERA = observations.CameraObservableSpec(
    height=240,
    width=480,
    enabled=True,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None,
)

_PERFECT_FEATURES = observations.ObservationSettings(
    proprio=observations._ENABLED_FEATURE,
    ftt=observations._ENABLED_FTT,
    prop_pose=observations._ENABLED_FEATURE,
    camera=_CAMERA,
)


class WorkspaceEntity(composer.ModelWrapperEntity):
    def _build(self, mjcf_model):
        self._mjcf_model = mjcf_model
        self._mjcf_root = mjcf_model


class RopeEntity(composer.ModelWrapperEntity):
    def _build(self):
        rope = mjcf.from_file(_ROPE_XML_PATH)
        self._mjcf_model = rope
        self._mjcf_root = rope
        self._bodies = self._mjcf_model.find_all("body")

    @property
    def bodies(self):
        return self._bodies

    def _build_observables(self):
        return RopeObservables(self)


class RopeObservables(composer.Observables):
    @define.observable
    def position(self):
        return observable.MJCFFeature("xpos", self._entity.bodies)

    @define.observable
    def orientation(self):
        return observable.MJCFFeature("xquat", self._entity.bodies)


class RopeWrapTask(composer.Task):
    def __init__(self, workspace, rope, obs_settings):
        self._root_entity = workspace
        self._rope = rope
        table_site = self._root_entity.mjcf_model.find("site", "table_top")
        self._root_entity.attach(self._rope, table_site)
        rope.observables.set_options(obs_settings.prop_pose._asdict())

        self._observables = rope.observables.as_dict()
        cameras = self._root_entity.mjcf_model.find_all("camera")
        camera_obs_dict = collections.OrderedDict()
        for cam in cameras:
            camera_obs_dict[cam.name] = observable.MJCFCamera(cam)
            camera_obs_dict[cam.name].configure(**obs_settings.camera._asdict())
        self._observables.update(camera_obs_dict)

    @property
    def root_entity(self):
        return self._root_entity

    def get_reward(self, physics):
        return 0.0

    @property
    def rope(self):
        return self._rope

    @property
    def observables(self):
        return self._observables


def rope_env():
    global env
    env.reset()
    env_xml = env.model.get_xml()

    world = mjcf.from_xml_string(env_xml)
    cube = world.find("body", "cube_main")
    grip_vis = world.find("site", "gripper0_grip_site_cylinder")
    geoms = world.find_all("geom")
    for geom in geoms:
        if "collision" in geom.name:
            geom.rgba = [0, 0, 0, 0]
    grip_vis.remove()
    cube.remove()

    rope_entity = RopeEntity()
    world_entity = WorkspaceEntity(world)

    task = RopeWrapTask(world_entity, rope_entity, _PERFECT_FEATURES)
    task.control_timestep = 0.1
    env = composer.Environment(task)

    return env


if __name__ == "__main__":
    env = rope_env()
    viewer.launch(env)
