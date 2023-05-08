class PolicyNetwork(nn.Module):
    """ Policy function parameterized by a NN. """

    def __init__(self, observation_space_size, action_space_size):
        """ Initialize the network as desired. """
        super(PolicyNetwork, self).__init__()

    def forward(self, x):
        """ Pass data through the network.

        Outputs:
        action_probabilities : probability of each action possibility
        """
        return action_probabilities

class ValueNetwork(nn.Module):
    """ Value function parameterized by a NN. """

    def __init__(self, observation_space_size):
        """ Initialize the network as desired. 

        Output is single dimensioned (Value).
        """
        super(ValueNetwork, self).__init__()

    def forward(self, state):
        """ Pass data through the network.

        Outputs:
        state_value : estimated value of the given state
        """
        return state_value
    