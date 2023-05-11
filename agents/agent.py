class Agent():
    """ Agent class that gives actions based on current state. """

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

    def update(self, states, actions, rewards):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ )] − H(τ ) 

        Inputs:
        states - sequence of observed states
        actions - sequence of performed actions
        rewards - sequence of rewards from the performed actions
        """
        entropy = forward(states) * log(forward(states))
        minimize(rewards - entropy)