from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv
from utils.batch import Batch
from utils.utils import normalize_states
import utils.utils as utils
import time
import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    states = []
    actions = np.load("demos/demo_actions.npy").tolist()
    states = np.load("demos/demo_states.npy").tolist()
    for state in states:
        normalize_states(state)

    probs=[1] * len(states)
    utils.transform_action(actions, states, probs)
    for state in states:
        normalize_states(state)
    expert_data = Batch(states=states, actions=actions, probs=[1] * len(states), pick_probs=[1] * len(states))

    action_size = 4
    state_size = 14
    agent_hidden_layer_size = 24
    agent_hidden_layers = 3
    cost_hidden_layer_size = 24
    cost_hidden_layers = 3
    agent = Agent(action_size, state_size, agent_hidden_layer_size, agent_hidden_layers)
    cost = Cost(action_size, state_size, cost_hidden_layer_size, cost_hidden_layers)
    env = ControlEnv(True)
    trainer = Trainer(env, agent, cost)
    iterations = 801
    trainer.train(iterations, expert_data)
