from env.control import Env

class Trainer():
    """ Class for training the RL agent. """

    def __init__(self, params):
        """ Create the gym enviroment and agent. """
        self.env = Env()
        self.agent = Agent()

    def training_loop(self, rollouts):
        """ Train the agent.

        Inputs:
        rollouts - number of iterations to train for
        """
        for itr in range(iterations):
            self.simulate()
            
    def simulate():
        """ Run one rollout. ""