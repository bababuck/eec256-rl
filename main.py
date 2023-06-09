from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv
import time
import gymnasium as gym
import numpy as np


if __name__ == '__main__':
#    env = ControlEnv('CartPole-v1')
#    env = ControlEnv('CartPole-v1')
#    gym.register(
#    id="roborope-v0",
#    entry_point=RopeEnv,
#    )
#    env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode = "human")
#    env = gym.make('RoboRope-v0', render_mode = "human")
    env = ControlEnv()
    env.reset()
    env.render()
    env.step([4, 0, 1])
    env.step([4, 0, 1])
    env.step([0, 0, -1])
    env.step([7, 0, -1])
    env.step([7, -0.5, 0])
    env.step([0, 0.5, 0])

    """
    action_size = env.action_space
    state_size = env.observation_space
    hidden_layer_size = 32
    hidden_layers = 1
    agent = Agent(action_size, state_size, hidden_layer_size, hidden_layers)
    cost = Cost(action_size, state_size, hidden_layer_size, hidden_layers)
    trainer = Trainer(env, agent, cost)
    iterations = 400
    trainer.train(iterations, "expert_data/expert_cartpole.npy")

    networks_folder = "netowrks"
    trainer.save_networks(networks_folder)
    agent.generate_samples(env)"""