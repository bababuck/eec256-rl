from agents.trainernew import Trainer
from agents.agentnew import Agent
from agents.cost import Cost
from env.control import ControlEnv
import time
import gymnasium as gym
import numpy as np


def two_to_1d(two_d):
    one_d = two_d[0]*4 + two_d[1]
    return one_d


#  Change 1D to 2D
def one_to_2d(one_d):
    two_d_back = [one_d // 4, one_d % 4]
    return two_d_back


def do_step(action):
    actions.append(two_to_1d(action))
    observation, reward, done = env.step(action)
    states.append(observation)
    rewards.append(reward)


def do_last_step(action):
    actions.append(two_to_1d(action))
    observation, reward, done = env.step(action)
    rewards.append(reward)


def load_file(filename, max_obs=None):
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
    """
    expert_data = []
    states = []
    actions = []
    rewards = []
    env = ControlEnv()
    env.reset()
    env.render()
    states.append(env.get_rope_states())  # Initial
    #  0:Up, 1:Down, 2:Left, 3:Right

    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)

    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)

    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)

    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_last_step(action)

    expert_curr = [states, actions, rewards]
    expert_data.append(expert_curr)

    #  Another
    states = []
    actions = []
    rewards = []
    env.reset()
    env.render()
    states.append(env.get_rope_states())  # Initial

    action = [4, 3]
    do_step(action)
    action = [4, 3]
    do_step(action)
    action = [4, 3]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [0, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 2]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [2, 0]
    do_step(action)
    action = [2, 0]
    do_step(action)
    action = [2, 0]
    do_step(action)
    action = [2, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [7, 0]
    do_step(action)
    action = [2, 0]
    do_step(action)
    action = [1, 0]
    do_step(action)
    action = [1, 0]
    do_step(action)
    action = [7, 3]
    do_step(action)
    action = [0, 1]
    do_step(action)
    action = [7, 3]
    do_step(action)
    action = [0, 1]
    do_last_step(action)

    expert_curr = [states, actions, rewards]
    expert_data.append(expert_curr)

    expert_data_np = np.array(expert_data, dtype=object)
    print("\n Expert Shape: ", expert_data_np.shape)
    np.save("expert_data/expert_rope.npy", expert_data_np)
    """

    """
    import time
    t_end = time.time() + 4 * 1
    while time.time() < t_end:
        env.render()

    print("\n Area :", env.get_area())
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
    
    """
    """
    np.save("expert_data/expert_rope.npy", np.array([states, actions, rewards]))
    print("\nActions:\n", actions)
    print("\nRewards:\n", rewards)
    print("\nStates:\n", states)
    # Get Expertdata

    import time
    
    t_end = time.time() + 8 * 1
    while time.time() < t_end:
        env.render()
    
    env.step([0, 0, -1])
    env.step([7, 0, -1])
    env.step([7, -0.5, 0])
    env.step([0, 0.5, 0])
    """    """
    states, actions = load_file("expert_data/expert_rope.npy")

    print("\n Shape: \n", states.shape)
    """
    action_size = 32   # Differ env.action_space
    state_size = env.observation_space
    hidden_layer_size = 32
    hidden_layers = [32, 32, 32]
    hidden_layers_cost = 2
    agent = Agent(action_size, state_size, hidden_layers)
    cost = Cost(action_size, state_size, hidden_layer_size, hidden_layers_cost)
    trainer = Trainer(env, agent, cost)
    iterations = 400
    trainer.train(iterations, "expert_data/expert_rope.npy")
    
    networks_folder = "networks"
    trainer.save_networks(networks_folder)
    agent.generate_samples(env)

