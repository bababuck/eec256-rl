from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv
from utils.batch import Batch
from utils.utils import normalize_states
import time
import gymnasium as gym
import numpy as np

   
def transform_action(actions):
    action_set = []
    action_set.append(actions)
    actions_cp = actions.copy()
    # Mirror segment picked
    for i in range(0,len(actions),2):
        actions_cp[i] = 7-actions_cp[i]
        if actions_cp[i+1] == 0:
            actions_cp[i+1] = 1
        elif actions_cp[i+1] == 1:
            actions_cp[i+1] = 0
    action_set.append(actions_cp)

    actions_cp = actions.copy()
    # Mirror both
    for i in range(0,len(actions),2):
        actions_cp[i] = 7-actions_cp[i]
        if actions_cp[i+1] == 0:
            actions_cp[i+1] = 1
        elif actions_cp[i+1] == 1:
            actions_cp[i+1] = 0
        if actions_cp[i+1] == 3:
            actions_cp[i+1] = 2
        elif actions_cp[i+1] == 2:
            actions_cp[i+1] = 3
    action_set.append(actions_cp)

    actions_cp = actions.copy()
    # Mirrow y direction
    for i in range(0,len(actions),2):
        if actions_cp[i+1] == 3:
            actions_cp[i+1] = 2
        elif actions_cp[i+1] == 2:
            actions_cp[i+1] = 3
    action_set.append(actions_cp)
    return action_set

if __name__ == '__main__':
    states = []
    env = ControlEnv()
    env.reset()
    env.render()
#    """
# 0 =[1,0]
# 1=[-1,0]
# 2=[0,1]
# 3=[0,-1]
    """
    actual_actions = [np.load("actions.csv.npy").tolist(), np.load("actions2.csv.npy").tolist()]
#    states = [np.load("state.csv.npy").tolist(), np.load("state2.csv.npy").tolist(), np.load("state3.csv.npy").tolist()]
    states = [[4,4,4,4,4,4,4,4,7,7,7,7,0,0,0,0, 0,0,0,0,0,0,0,0, 7, 0,0, 7,7,7,7,7,7,7,7,7,7, 4],
              [4,4,4,4,4,4,0,0,0,0,7,7,7,7,0,0,0,0,7,7,7,7, 0,0,0,0,7,7,7,7,    2,2,2,2, 5,5,5,5, 0, 0, 0, 0, 7,7,7,7,7,7],
              []]
    action_set = []
    for t in range(len(actual_actions)):
        action = actual_actions[t][:len(actual_actions[t])//4]
        actions = []
        print(action)
        state = states[t]
        print(state)
        for i in range(len(action)):
            print(i)
            actions.append(state[i])
            actions.append(action[i])
        print(actions)
        set = transform_action(actions)
        action_set += set

#    '''
    actual_actions = []
    actions = [2, 2] * 4 + \
               [5, 2] * 4+\
               [0, 3] * 4+\
               [7, 3] * 4+\
               [0, 0] * 4+\
               [7, 1] * 4+\
               [0, 0] * 4+\
               [7, 1] * 4+\
               [2, 1] * 4+\
               [5, 0] * 4+\
               [0, 0] * 4+\
               [7, 1] * 4+\
               [7, 1] * 2+\
               [4, 2] * 2+\
               [3, 2] * 2+\
               []
#    ''' 
    set = transform_action(actions)
    action_set += set
    states = []
    actual_actions = []
    for actions in action_set:
        env.reset()
        for i in range(0,len(actions),2):
            states.append(env.get_rope_states())
            states[-1][16+actions[i]] = 1
            env.step(actions[i], actions[i+1])
            actual_actions.append(actions[i+1])
            env.render()

    np.save("state", np.array(states))
    np.save("action", np.array(actual_actions))
    env.end_render()
#    """
    states = np.load("state.npy").tolist()
    for state in states:
        normalize_states(state)
    actual_actions = np.load("action.npy").tolist()
    expert_data = Batch(states=states, actions=actual_actions, probs=[1] * len(states))
#    print(states)
#    print(actual_actions)
    action_size = 4
    state_size = 24
    hidden_layer_size = 32
    hidden_layers = 2
    agent = Agent(action_size, state_size, hidden_layer_size, hidden_layers)
    cost = Cost(action_size, state_size, hidden_layer_size, hidden_layers)
    trainer = Trainer(env, agent, cost)
    iterations = 400
    trainer.train(iterations, expert_data)

    networks_folder = "networks"
    trainer.save_networks(networks_folder)
