import torch
import torch.nn as nn

def generate_simple_network(input_size, output_size, hidden_size, hidden_layers):
    """ Generate a simple feed forward network of given dimensions. """
    # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    layers += [x for x in [nn.Linear(hidden_size, hidden_size), nn.ReLU()] for _ in range(hidden_layers)]
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

def normalize_states(states):
    min_x = 100
    min_y = 100
    for i in range(0, 12, 2):
        if (min_x > states[i]):
            min_x = states[i]
    for i in range(0, 12, 2):
        states[i] = states[i] - min_x

    for i in range(1, 12, 2):
        if (min_y > states[i]):
            min_y = states[i]
    for i in range(1, 12, 2):
        states[i] = states[i] - min_y

def transform_action(actions, states, probs):
    new_actions = []
    new_states = []
    # Mirror segment picked

    for i in range(len(actions)):
        probs.append(probs[i])
        if actions[i] == 0:
            new_actions.append(1)
        elif actions[i] == 1:
            new_actions.append(0)
        else:
            new_actions.append(actions[i])
        new_state = states[i].copy()
        for j in range(0,12,2):
            new_state[j] = -new_state[j]
        new_states.append(new_state)

    actions += new_actions
    states += new_states
    new_actions = []
    new_states = []
    for i in range(len(actions)):
        probs.append(probs[i])
        if actions[i] == 2:
            new_actions.append(3)
        elif actions[i] == 3:
            new_actions.append(2)
        else:
            new_actions.append(actions[i])
        new_state = states[i].copy()
        for j in range(1,12,2):
            new_state[j] = -new_state[j]
        new_states.append(new_state)
    actions += new_actions
    states += new_states

    c = len(actions)
    for i in range(c):
        probs.append(probs[i])
        if actions[i] == 0:
            actions.append(2)
        elif actions[i] == 1:
            actions.append(3)
        elif actions[i] == 2:
            actions.append(0)
        elif actions[i] == 3:
            actions.append(1)
        new_state = states[i].copy()
        for j in range(0,12,2):
            new_state[j], new_state[j+1] = new_state[j+1], new_state[j]
        states.append(new_state)

    c = len(actions)
    for i in range(c):
        probs.append(probs[i])
        if actions[i] == 0:
            actions.append(2)
        elif actions[i] == 1:
            actions.append(3)
        elif actions[i] == 2:
            actions.append(0)
        elif actions[i] == 3:
            actions.append(1)
        new_state = states[i].copy()
        new_state[12], new_state[13] = new_state[13], new_state[12]
        for j in range(0,6,1):
            new_state[11-j], new_state[j] = new_state[j], new_state[11-j]
        states.append(new_state)

    for state in states:
        normalize_states(state)