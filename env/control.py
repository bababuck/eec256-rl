from .random_agent import RandomAgent
import gymnasium as gym

class ControlEnv():
    """ Interface for our RL agent to interact with the enviroment. """

    NUM_RND_ACTIONS = 50 # Number of random actions upon state reset

    def __init__(self, env_name, add_randomness=False):
        """
        Random agent will be used to randomly initialize the state.
        """
        self.env = gym.make(env_name, render_mode = "human")
        if add_randomness:
            self.random_agent = RandomAgent(self.env.action_space.n)
        else:
            self.random_agent = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def step(self, action):
        """ Take one action in the simulation.

        Inputs:
        action - action for the agent to take

        Outputs:
        observation - state of enviroment following the action
        reward - reward from the prior action
        done - is episode complete
        """
        observation, reward, done, truncated, info = self.env.step(action)
        return observation, reward, done

    def reset(self):
        """ Reset the enviroment.

        Outputs:
        observation - state of enviroment following the reset
        """
        observation, info = self.env.reset()
        if self.random_agent is not None:
            for i in range(self.NUM_RND_ACTIONS):
                random_action = self.random_agent.get_action()
                self.step(random_action)
        return observation

    def render(self):
        self.env.render()