from agents.agent import Agent
from env.control import ControlEnv
from utils.utils import normalize_states
import utils.utils as utils
import time
import torch
import numpy as np

if __name__ == '__main__':
    # For demo of our policy
    # """
    dir_action_size = 4
    seg_action_size = 2
    state_size = 14
    agent_hidden_layer_size = 24
    agent_hidden_layers = 3
    agent = Agent(dir_action_size, seg_action_size, state_size, agent_hidden_layer_size, agent_hidden_layers)
    agent.load_pick("networks/pickagent_big_it_400.pt")
    agent.load("networks/agent_big_it_400.pt")
    env = ControlEnv(True, 0.02)
    np.random.seed(1000)
    ob = env.reset()
    env.render()
    time.sleep(1)
    for i in range(100):
        segment = argmax(agent.get_pick_probs(ob[:12], False).numpy())
        print(segment)
        ob[12+segment] = 1
        direction, probs = agent.get_policy_action(torch.tensor(ob, dtype=torch.float32))
        print(direction)
        print(probs)
        ob, _, _ = env.step(3*segment, direction)
        env.render()
    """
    # For demo of expert
    setup = [3, 3, 3, 1, 3, 1, 2, 0, 3, 1, 2, 0, 0, 0, 1, 2, 3, 3, 3, 0]
    demo = [3,1, 3,1, 0,3, 3,2, 3,2]

    env = ControlEnv(False, 0.1)

    env.reset()
    for i in range(0,len(setup),2):
        env.step(segment=setup[i], direction=setup[i+1])
    
    time.sleep(15)

    for i in range(0,len(demo),2):
        env.render()
        env.step(segment=demo[i], direction=demo[i+1])

    env.render()
    time.sleep(15)
    env.end_render()
    """