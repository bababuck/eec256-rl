
import numpy as np
import torch
import random

class Batch:
    def __init__(self, load_file=None, states=[], probs=[], actions=[]):
        if load_file:
            self.states, self.actions = self.load_file(load_file)
            print("\n Shape: \n", np.shape(self.states))
            self.probs = np.ones((np.shape(self.states)[0], 1))
        else:
            self.actions = np.array(actions, dtype=int)
            if actions == []:
                self.states = np.zeros((0, 16))
                self.probs = np.zeros((0, 1))
            else:
                self.states = np.array(states)
                self.probs = np.array(probs).reshape(-1, 1)
        # actions = np.zeros((self.actions.size, 32))
        # actions[np.arange(self.actions.size), self.actions] = 1

    def sample(self, count):
        sampled_batch = Batch()
        if (len(self.states)):
            idx = np.random.choice(len(self.states), count)
            sampled_batch.states = self.states[idx]
            sampled_batch.probs = self.probs[idx]
            sampled_batch.actions = self.actions[idx]
        return sampled_batch

    def extend(self, other):
        """ Combine with a second batch.

        Inputs:
        other - another batch type object
        """
        self.states = np.concatenate((self.states, other.states), axis=0)
        self.probs = np.concatenate((self.probs, other.probs), axis=0)
        if isinstance(other.actions[0], np.ndarray):
            add_actions = other.actions[:, 0]
            print("\n Other actions added : \n", add_actions)
        else:
            add_actions = other.actions
        self.actions = np.concatenate((self.actions, add_actions), axis=0)

    def load_file(self, filename, max_obs=None):
        loaded_data = np.load(filename, allow_pickle=True)
        states = np.zeros((0, 16))
        actions = np.array([])
        for ob in loaded_data:
            states = np.concatenate((states, np.array(ob[0])), axis=0)
            actions = np.concatenate((actions, np.array(ob[1])), axis=0)
            obs = np.shape(states)[0]
            if max_obs != None and obs > max_obs:
                break
        return states, actions.astype(int)
