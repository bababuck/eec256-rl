import numpy as np
import pytorch as torch

class Batch:
    def __init__(self, load_file=None, states=[], probs=[], actions=[]):
        if load_file:
            loaded_data = np.load(load_file)
            self.states = torch.tensor(loaded_data[0])
            self.probs = torch.tensor(loaded_data[1])
            self.actions = torch.tensor(loaded_data[2])
        else:
            self.states = torch.tensor(states)
            self.probs = torch.tensor(probs)
            self.actions = torch.tensor(actions)

    def sample(self, count):
        idx = np.random.choice(len(states), count)
        sampled_batch = Batch()
        sampled_batch.states = states[idx]
        sampled_batch.probs = probs[idx]
        sampled_batch.actions = actions[idx]
        return sampled_batch

    def extend(self, other):
        """ Combine with a second batch.

        Inputs:
        other - another batch type object
        """
        self.states = torch.cat(self.states, other.states, axis=1)
        self.probs = torch.cat(self.probs, other.probs, axis=1)
        self.actions = torch.cat(self.actions, other.actions, axis=1)
