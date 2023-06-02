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
    states = []
    actions = []
    rewards = []
    env = ControlEnv()
    env.reset()
    env.render()
    states.append(env.get_rope_states())  # Initial

    # First Expert data

    #  print("\nInitial:\n", env.get_rope_states())
    action = [0, 0, -1]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0, -1]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0, -1]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0.5, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0.5, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0.5, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0.5, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [0, 0.5, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [7, 0, -3]
    actions.append(action)
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)

    action = [7, -3, 0]
    actions.append(action)
    observation, reward, done = env.step(action)
    # NO final state added
    rewards.append(reward)

    np.save("expert_data/expert_rope.npy", np.array([states, actions, rewards]))
    print("\nActions:\n", actions)
    print("\nRewards:\n", rewards)
    print("\nStates:\n", states)
    # Get Expertdata

    """
    import time
    
    t_end = time.time() + 8 * 1
    while time.time() < t_end:
        env.render()
    
    env.step([0, 0, -1])
    env.step([7, 0, -1])
    env.step([7, -0.5, 0])
    env.step([0, 0.5, 0])
    """
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
    agent.generate_samples(env)
    """
