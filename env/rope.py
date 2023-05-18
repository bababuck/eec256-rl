import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {}


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment corresponds to the rope enviroment described in our project.

    The rope consist of 8 independent segments coneected at their ends.

    The rope always begins in a straight line.

    Actions involve moving a joint to another location, the rest of the rope will move appropriately.

    ### Notes

    The problem parameters are:
    Problem parameters:
    * *n*: number of rope segments
    * *m<sub>i</sub>*: mass of each segment
    * *l<sub>i</sub>*: length of each segment
    * *k*: friction coefficient

    ### Action Space
    An action is dragging a segment to another location. We move the center of the segment.

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Unit         |
    |-----|------------------------------------|-------------|-------------|----------------------------------|--------------|
    | 0   | Segment to move                    | 0           | n           |                                  | None         |
    | 1   | Distance to move                   | -1          | 1           |                                  | distance (m) |

    ### Observation Space

    By default, the observation is a `ndarray` with shape `(18,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ------------------------ |
    | i   | x-coordinate of end 0                | -10  | 10  |                                  | location (m)             |
    | i   | y-coordinate of end 0                | -10  | 10  |                                  | location (m)             |

    ### Rewards
    The reward consists of:
    - *circular_reward*: how far is each angle from 135 degrees (octagon).

    ### Starting State
    All ropes start in a line.

    ### Episode End
    The episode truncates when the episode length is greater than 1000 or if all angles fall in range 125 - 145 degrees.

    ```
    env = gym.make('Rope')
    ```

    Followed template from https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v4.py
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self, "rope.xml", 4, observation_space=observation_space, **kwargs
        )

    def _get_angles_reward(self):
        return 0 # don't care about this actually

    def step(self, action):
        angles_reward_before = self._get_angles_reward()
        self.do_simulation(action, self.frame_skip)
        angles_reward_after = self._get_angles_reward()

        reward = angles_reward_after - angles_reward_before
        observation = self._get_obs()
        reward = forward_reward - ctrl_cost

        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()

        observation = position.ravel()
        return observation

    def reset_model(self):
        self.set_state(self.init_qpos, slef.init_qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)