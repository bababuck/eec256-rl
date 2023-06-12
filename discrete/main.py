from discrete_agents.trainer import Trainer as DiscreteTrainer
from discrete_agents.agent import Agent as DiscreteAgent
from discrete_agents.cost import Cost as DiscreteCost
from env.control import ControlEnv
from utils.discrete_batch import Batch as DiscreteBatch
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
    expert_data = DiscreteBatch(states=states, actions=actions, probs=[1] * len(states), pick_probs=[1] * len(states))
 
    dir_action_size = 4
    seg_action_size = 2
    state_size = 14
    agent_hidden_layer_size = 24
    agent_hidden_layers = 3
    cost_hidden_layer_size = 24
    cost_hidden_layers = 3
    agent = DiscreteAgent(dir_action_size, seg_action_size, state_size, agent_hidden_layer_size, agent_hidden_layers)
    cost = DiscreteCost(dir_action_size, state_size, cost_hidden_layer_size, cost_hidden_layers)
    env = ControlEnv(True)
    trainer = DiscreteTrainer(env, agent, cost)
    iterations = 801
    trainer.train(iterations, expert_data)
