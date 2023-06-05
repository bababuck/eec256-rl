from .random_agent import RandomAgent
import gymnasium as gym
import numpy as np
import math
import time


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def to_action(input_action):
    x = input_action[0]
    b = input_action[1]
    #  0:Up, 1:Down, 2:Left, 3:Right
    if b == 0:
        y, z = -0.5, 0
    elif b == 1:
        y, z = 0.5, 0
    elif b == 2:
        y, z = 0, -0.5
    elif b == 3:
        y, z = 0, 0.5
    else:
        raise ValueError("The second element of the input list should be 0, 1, 2, or 3.")

    return [x, y, z]


def one_to_2d(one_d):
    two_d_back = [one_d // 4, one_d % 4]
    return two_d_back


class ControlEnv():
    """ Interface for our RL agent to interact with the enviroment. """

    NUM_RND_ACTIONS = 50 # Number of random actions upon state reset

    def __init__(self, add_randomness=False):
        """
        Random agent will be used to randomly initialize the state.
        """
        self.env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode = "human")
        self.count = 0
        if add_randomness:
            self.random_agent = RandomAgent(self.env.action_space.n)
        else:
            self.random_agent = None

    @property
    def action_space(self):
        return 32

    @property
    def observation_space(self):
        return 16

    def step(self, action):
        """ Take one action in the simulation.

        Inputs:
        action - action for the agent to take
               - encoded as [segment, dx, dy]

        Outputs:
        observation - state of enviroment following the action
        reward - reward from the prior action
        done - is episode complete
        """
        if isinstance(action, int):
            action = one_to_2d(action)
        action = to_action(action)
        observation, reward, done = self.do_actions(action)
        return observation, reward, done

    def do_actions(self, action):
        # Get the current rope states
        curr_state = self.get_rope_states()
        curr_rope_x = np.array(curr_state[::2])  # This gets every other element, starting from 0, so all x coordinates.
        curr_rope_y = np.array(curr_state[1::2])  # This gets every other element, starting from 1, so all y coordinates.
        curr_area = poly_area(curr_rope_x, curr_rope_y)

        # First move to picked location
        new_x, new_y, _ = self.env.get_rope_pos(action[0])
        curr_x, curr_y, _ = self.env.get_gripper_xpos()
        while (curr_x > new_x + 0.001) or (curr_x < new_x - 0.001) or (curr_y > new_y + 0.001) or (curr_y < new_y - 0.001):
            curr_x, curr_y, _ = self.env.get_gripper_xpos()
            move_x = min(1, (new_x - curr_x)*20)
            move_y = min(1, (new_y - curr_y)*20)
            self.env.step(np.array([move_x, move_y, 0, 0]))
        # Then lower
        self.env.step(np.array([0, 0, -0.75, 0]))
        # Then move
        dx = action[1]/20
        dy = action[2]/20
        for i in range(20):
            self.env.step(np.array([dx, dy, 0, 0])) 

#        dx = action[1]
#        dy = action[2]
#        self.env.step(np.array([dx, dy, 0, 0])) 
#        curr_x, curr_y, _ = self.env.get_gripper_xpos()

        # Then raise
        self.reset_gripper_height()
        self.count += 1

        # New area
        new_state = self.get_rope_states()
        new_rope_x = np.array(new_state[::2])  # This gets every other element, starting from 0, so all x coordinates.
        new_rope_y = np.array(new_state[1::2])  # This gets every other element, starting from 1, so all y coordinates.
        new_area = poly_area(new_rope_x, new_rope_y)

        # Calculate reward
        # Initial state [1.2499990569368231, 0.75, 1.2799964178638663, 0.75, 1.3099998632539256, 0.75, 1.3399998312433623, 0.75, 1.3699998721215199, 0.75,
        # 1.3999998901256643, 0.75, 1.429999854380289, 0.75, 1.4600019932462143, 0.75]
        # Perimeter 0.24
        radius = 0.21 / (2 * math.pi)
        circle_area = math.pi * radius ** 2

        # If the polygon is open, area is larger than circle area.
        # Can change to higher order. Maybe add step penalty. Can be related to action length
        reward = abs(curr_area - circle_area) - abs(new_area - circle_area)
        reward = reward * 100  # Scale up
        """
        if curr_area > circle_area:
            reward = curr_area - new_area
        else:
            reward = new_area - curr_area
        """
        return new_state, reward, self.count > 100

    # Get the area of polygon covered by the rope to calculate reward

    def get_area(self):
        new_state = self.get_rope_states()
        new_rope_x = np.array(new_state[::2])  # This gets every other element, starting from 0, so all x coordinates.
        new_rope_y = np.array(new_state[1::2])  # This gets every other element, starting from 1, so all y coordinates.
        new_area = poly_area(new_rope_x, new_rope_y)
        return new_area


    def get_rope_states(self):
        state = []
        for i in range(8):
            x, y, _ = self.env.get_rope_pos(i)
            state.append(x)
            state.append(y)
        return state

    def reset_gripper_height(self):
        _, _, curr_z = self.env.get_gripper_xpos()
        new_z = 0.475
        while (curr_z > new_z + 0.001) or (curr_z < new_z - 0.001):
            _, _, curr_z = self.env.get_gripper_xpos()
            move_z = min(1, (new_z - curr_z)*20)
            self.env.step(np.array([0, 0, move_z, 0]))

    def reset(self):
        """ Reset the enviroment.

        Outputs:
        observation - state of enviroment following the reset
        """
        observation, info = self.env.reset()
        # Set gripper height
        self.reset_gripper_height()

        self.count = 0
        if self.random_agent is not None:
            for i in range(self.NUM_RND_ACTIONS):
                random_action = self.random_agent.get_action()
                self.step(random_action)
        return observation

    def render(self):
        self.env.render()

    def end_render(self):
        self.env.close()
