
from .random_agent import RandomAgent
import gymnasium as gym
import numpy as np
from agents.agent import Agent
from agents.trainer import Trainer
from utils.utils import normalize_states
import time
import torch

class ControlEnv():
    """ Interface for our RL agent to interact with the enviroment. """

    ACTION_THRESHOLD=0.001 # How accurate do we want our actions to be

    def __init__(self, add_randomness=False, sleep_time=0):
        """
        Random agent will be used to randomly initialize the state.
        """
        self.env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode = "human")
        self.count = 0
        if add_randomness:
            self.random_agent = True
        else:
            self.random_agent = False
        np.random.seed(0)
        self.sleep_time = sleep_time

    def step(self, segment, action, reset=False):
        """ Take one action in the simulation.

        Inputs:
        seg    - segment to grab
        action - action for the agent to take
               - encoded as [segment, dx, dy]

        Outputs:
        observation - state of enviroment following the action
        reward - reward from the prior action
        done - is episode complete
        """
        if reset:
            # 8 segment options in underlying environment, but 4 options provided by the wrapper
            seg = {0:0, 1:2, 2:4, 3:7}[segment]
        else:
            seg = {0:0, 1:7}[segment] # One hot

        # First move to picked location
        new_x, new_y, _ = self.env.get_rope_pos(seg)
        curr_x, curr_y, _ = self.env.get_gripper_xpos()
        while (curr_x > new_x + self.ACTION_THRESHOLD) or (curr_x < new_x - self.ACTION_THRESHOLD) or (curr_y > new_y + self.ACTION_THRESHOLD) or (curr_y < new_y - self.ACTION_THRESHOLD):
            curr_x, curr_y, _ = self.env.get_gripper_xpos()
            move_x = min(1, (new_x - curr_x)*20)
            move_y = min(1, (new_y - curr_y)*20)
            self.env.step(np.array([move_x, move_y, 0, 0]))

        # Then lower the gripper
        self.env.step(np.array([0, 0, -0.75, 0]))

        # Then move the speicifed distance
        dx = action[0]
        dy = action[1]
        dx = dx/20
        dy = dy/20
        for i in range(20):
            self.env.step(np.array([dx, dy, 0, 0])) 
            if not reset:
                self.render()
                time.sleep(self.sleep_time)

        # Then raise the gripper
        self.reset_gripper_height()

        if not reset:
            self.count += 1
        state = self.get_rope_states() 

        # Rollout is done if:
        # - more steps than MAX_EPISODE_STEPS
        # - any segment is no longer reachable by the gripper
        #      - This range is roughly empirically determined   
        done = self.count > 50
        for i in range(1,12,2):
            if state[i] < 0.4 or state[i] > 1.1:
                done = True
        for i in range(0,12,2):
            if state[i] < 1.13  or state[i] > 1.47:
                done = True

        normalize_states(state)

        # Never give reward
        reward = 0

        return state, reward, done

    def get_rope_states(self):
        """ Get the state of all the ropes. """
        state = []
        for i in [0, 2, 3, 4, 5, 7]:
            x, y, _ = self.env.get_rope_pos(i)
            state.append(x)
            state.append(y)
        return np.array(state)

    def reset_gripper_height(self):
        """ Move the gripper to a set height above the rope.
        Can move freely in the x-y plane at this height.
        """
        _, _, curr_z = self.env.get_gripper_xpos()
        new_z = 0.475
        while (curr_z > new_z + self.ACTION_THRESHOLD) or (curr_z < new_z - self.ACTION_THRESHOLD):
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

        setups = [[1, 3, 2, 2, 3, 1, 0, 0, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1],
              [2, 2, 3, 3, 1, 2, 3, 1, 3, 1, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 3, 1, 1, 1],
              [1, 3, 1, 3, 3, 3, 0, 2, 2, 3, 3, 1, 0, 0, 0, 0, 3, 1, 2, 1, 3, 3, 0, 0, 3, 2, 3, 3],
              [3, 3, 1, 2, 0, 0, 0, 0, 3, 1, 1, 2, 1, 3],
              [0, 2, 1, 3, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1],
              [1, 2, 2, 2, 1, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 2, 0],
              [3, 3, 3, 1, 3, 1, 2, 0, 3, 1, 2, 0, 0, 0, 1, 2, 3, 3, 3, 0],
              [3, 3, 1, 2, 1, 2, 0, 0, 3, 3, 3, 1, 3, 1, 0, 0, 3, 1, 3, 1, 3, 3, 2, 0, 0, 0, 2, 0],
              [1, 3, 2, 2, 2, 2, 1, 3, 1, 3, 1, 3, 0, 2, 2, 0, 0, 2, 3, 1, 0, 0, 3, 1, 3, 1, 0, 0],                                                                                                                                         
              [1, 2, 1, 3, 1, 2, 1, 2, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 3, 1],
              [2, 2, 0, 0, 1, 2, 1, 2, 3, 1, 1, 0, 3, 0, 3, 1],
              [2, 2, 0, 0, 3, 1, 3, 3, 3, 1, 1, 3, 2, 1, 0, 0, 3, 2, 0, 0, 3, 3, 0, 0],
              [0, 2, 2, 2, 3, 1, 2, 2, 3, 3, 3, 1, 3, 1, 0, 0, 3, 3, 1, 2, 3, 1],
              [2, 2, 2, 2, 0, 2, 1, 2, 3, 1, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 3, 1],
              [0, 2, 0, 2, 1, 3, 3, 1, 1, 3, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0]]

        if self.random_agent:
            setup = setups[np.random.randint(0,15)]
            for i in range(0, np.random.randint(0, len(setup)), 2):
                directions = [0, 0]
                if setup[i + 1] == 0:
                    directions = [1, 0]
                elif setup[i + 1] == 1:
                    directions = [-1, 0]
                elif setup[i + 1] == 2:
                    directions = [0, 1]
                elif setup[i + 1] == 3:
                    directions = [0, -1]
                self.step(setup[i], directions, True)

        state = self.get_rope_states()
        normalize_states(state)
        return state

    def render(self):
        self.env.render()

    def end_render(self):
        self.env.close()
