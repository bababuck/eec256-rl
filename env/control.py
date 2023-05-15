from agent.random_agent import RandomAgent
import gymnasium as gym

class ControlEnv:
    """ Interface for our RL agent to interact with the enviroment. """

    NUM_RND_ACTIONS - 50 # Number of random actions upon state reset

    def __init__():
        """

        Random agent will be used to randomly initialize the state.
        """
        self.random_agent = RandomAgent(num_actions)
        self.env = gym.make("simple_rope", render_mode="human")

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
        for i in range(NUM_RND_ACTIONS):
            self.
        return observation