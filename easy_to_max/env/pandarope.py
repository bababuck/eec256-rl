from robosuite.models import MujocoWorldBase
import numpy as np

world = MujocoWorldBase()

from robosuite.models.robots import Panda

mujoco_robot = Panda()

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# ADD Rope here from xml file
"""
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
class RopeObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("objects/rope.xml"), name=name)


rope = RopeObject(name="rope")
rope.set("pos", "0 0 0")
world.worldbody.append(rope)
"""
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)


model = world.get_model(mode="mujoco")

import mujoco
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

sim = MjSim(model)
viewer = OpenCVRenderer(sim)
viewer.render()
