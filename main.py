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
    env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode = "human")
#    env = gym.make('RoboRope-v0', render_mode = "human")
    env.reset()
    env.render()
    time.sleep(2)
    action = np.array([0, 1, 0, 0])
    env.step(action)
    time.sleep(1)
    action = np.array([1, 0, 0, 0])
    env.step(action)
    time.sleep(1)
#    action = np.array([0, 0, 0, -0.02])
#    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    action = np.array([0, 0, -1, 0])
    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, 0, 1])
    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, 0, -1])
    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, 0, -1])
    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, 0, -0.02])
    env.step(action)
    time.sleep(1)
    action = np.array([0, 0, 0, -0.02])
    env.step(action)
    action = np.array([0, -1, 0, 0])
    env.step(action)
    time.sleep(1)
    action = np.array([0, -1, 0, 0])
    env.step(action)
    time.sleep(1)
    action = np.array([0, -1, 0, 0])
    env.step(action)
    time.sleep(1)
    action = np.array([0, -1, 0, 0])
    env.step(action)
    time.sleep(1)
    time.sleep(50)
    env.close()
#    env.end_render()
"""
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
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