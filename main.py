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
        actions_cp[i] = 3-actions_cp[i]
        if actions_cp[i+1] == 0:
            actions_cp[i+1] = 1
        elif actions_cp[i+1] == 1:
            actions_cp[i+1] = 0
    action_set.append(actions_cp)

    actions_cp = actions.copy()
    # Mirror both
    for i in range(0,len(actions),2):
        actions_cp[i] = 3-actions_cp[i]
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
    """
# 0 =[1,0]
# 1=[-1,0]
# 2=[0,1]
# 3=[0,-1]

    actual_actions = []
    action_set = []
    actions = [
               [3,2,4,2,3,2,4,2, 7,1, 0,0, 1,1,7,1, 0,0],
               [4,2, 3,2, 0,0, 7,1, 4,0, 3,1, 0,0, 7,1, 7,1],
               [7,3, 0,3, 7,3, 0,3, 2,2, 4,2, 7,1, 0,0, 7,1, 0,0,4,2, 3,2,],
               [0,3, 7,3, 4,2, 3,2, 7,1, 0,0, 7,1, 0,0, 4,0,2,1, 0,0], # okay :/
               ]
    for action in actions:
        for i in range(0,len(action), 2):
            if action[i] == 7:
                action[i] = 3
            elif action[i] == 3:
                action[i] = 1
            elif action[i] == 4:
                action[i] = 2

#    ''' 
    for action in actions:
        set = transform_action(action)
        action_set += set
    states = []
    actual_actions = []
    for actions in action_set:
        for _ in range(10):
            env.reset(jitter=True)
            for i in range(0,len(actions),2):
                states.append(env.get_rope_states())
                states[-1][8+actions[i]] = 1
                env.step(actions[i], actions[i+1])
                actual_actions.append(actions[i+1])

    np.save("b_state", np.array(states))
    np.save("b_action", np.array(actual_actions))
    env.end_render()
#    """
    states = np.load("b_state.npy").tolist()
    actual_actions = np.load("b_action.npy").tolist()

    rot = []
    for action in actual_actions:
        a = (action + 2) % 4
        rot.append(a)

    actual_actions += rot

    rot = []
    for state in states:
        r=state.copy()
        for i in range(0,8,2):
            r[i],r[i+1]=r[i+1],r[i]
        rot.append(r)
    states += rot
    for state in states:
        normalize_states(state)
    
    expert_data = Batch(states=states, actions=actual_actions, probs=[1] * len(states))
#    print(states)
#    print(actual_actions)
    action_size = 4
    state_size = 12
    hidden_layer_size = 8
    hidden_layers = 2
    agent = Agent(action_size, state_size, hidden_layer_size, hidden_layers)
    cost = Cost(action_size, state_size, hidden_layer_size, hidden_layers)
    trainer = Trainer(env, agent, cost)
    iterations = 400
    trainer.train(iterations, expert_data)

    networks_folder = "networks"
    trainer.save_networks(networks_folder)
