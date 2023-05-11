import numpy as np

class Batch:
    def __init__(self, load_file=None, states=[], probs=[], actions=[]):
        if load_file:
            loaded_data = np.load(load_file)
            self.states = loaded_data[0]
            self.probs = loaded_data[1]
            self.actions = loaded_data[2]
        else:
            self.states = np.array(states)
            self.probs = np.array(probs)
            self.actions = np.array(actions)

    def sample(self):
        idx = np.random.choice(len(states), DEMO_BATCH)
        return np.concatenate((states, probs, actions), axis=1)[idx]

    def extend(self, other):
        """ Combine with a second batch.

        Inputs:
        other - another batch type object
        """
        self.states = np.concatenate(self.states, other.states, axis=1)
        self.probs = np.concatenate(self.probs, other.probs, axis=1)
        self.actions = np.concatenate(self.actions, other.actions, axis=1)
