class ControlEnv:
    """ Interface for our RL agent to interact with the enviroment. """

    def __init__():

    def step(self, action):
        """ Take one action in the simulation.

        Inputs:
        action - action for the agent to take

        Outputs:
        observation - state of enviroment following the action
        reward - reward from the prior action
        done - is episode complete
        """

    def reset(self):
        """ Reset the enviroment.

        Outputs:
        observation - state of enviroment following the reset
        """
