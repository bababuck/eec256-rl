from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv
from utils.batch import Batch
from utils.utils import normalize_states
import utils.utils as utils
import matplotlib.pyplot as plt
import time
import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    states = []
    actions = np.load("demos/demo_cont_actions.npy").tolist()
    states = np.load("demos/demo_cont_states.npy").tolist()
    for state in states:
        normalize_states(state)

    probs=[1] * len(states)
    utils.transform_action(actions, states, probs)
    for state in states:
        normalize_states(state)
    expert_data = Batch(states=states, actions=actions, probs=[1] * len(states))

    action_size = 3
    discrete_action_size = 2
    cont_action_size = 2
    state_size = 12
    hidden_layers = [24, 24, 24]
    agent_hidden_layer_size = 24
    agent_hidden_layers = 3
    cost_hidden_layer_size = 24
    cost_hidden_layers = 3
    agent = Agent(discrete_action_size, cont_action_size, state_size, hidden_layers)
    cost = Cost(action_size, state_size, cost_hidden_layer_size, cost_hidden_layers)
    env = ControlEnv(True)
    trainer = Trainer(env, agent, cost)
    iterations = 801
    trainer.train(iterations, expert_data)

    iteration = np.arange(0, iterations)
    plt.plot(iteration, cost.ioc_lik, label='IOC Likelihood')  # Plot cost vs iterations
    plt.xlabel('Iterations')  # Label x-axis
    plt.ylabel('Loss')  # Label y-axis
    plt.title('Loss of Cost Function Over Time')  # Title of the graph
    plt.legend()  # Show legend
    # Save the plot as a PNG file
    plt.savefig('cost vs iterations.png')

    plt.figure()
    plt.plot(iteration, agent.cum_loss, label='MSE Loss')  # Plot cost vs iterations
    plt.xlabel('Iterations')  # Label x-axis
    plt.ylabel('MSE Loss')  # Label y-axis
    plt.title('Loss of Policy Function Over Time')  # Title of the graph
    plt.legend()  # Show legend
    # Save the plot as a PNG file
    plt.savefig('MSE vs iterations.png')