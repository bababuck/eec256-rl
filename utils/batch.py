import numpy as np

class Batch:
    def __init__(self, load_file=None):
        if load_file:
            loaded_data = np.load(load_file)
            self.states = loaded_data[0]
            self.probs = loaded_data[1]
            self.actions = loaded_data[2]
        else:
            self.states = np.array([])
            self.probs = np.array([])
            self.actions = np.array([])

    def sample(self):
        idx = np.random.choice(len(states), DEMO_BATCH)
        return np.concatenate((states, probs, actions), axis=1)[idx]