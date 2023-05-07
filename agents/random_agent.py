import Agent from agent

class RandomAgent(Agent):
    """ Agent class that gives random actions based on current state.

    Will use for initializing rope state.
    """

    def __init__(self):
        """ Init policy and value functions. """

    def get_action(self, state, training):
        """ Return an action based on the current state.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - action to perform
        """
        return get_random_action(self, state)

    def get_random_action(self, state):
        """ Return a random aciton. """