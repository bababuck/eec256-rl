import Agent from agent
import random

class RandomAgent():
    """ Agent class that gives random actions based on current state.

    Will use for initializing rope state.
    """
    def __init__(self, action_size):
        """ Init network and optimizer. """
        self.num_actions = action_size

    def get_action(self):
        """ Return an action randomly.

        Outputs:
        action - action to perform
        """
        return get_random_action(self)

    def get_random_action(self):
        """ Return a random aciton. """
        return random.randint(0, self.num_actions-1)