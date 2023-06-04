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
    for i in range(0, 15, 2):
        if (min_x > states[i]):
            min_x = states[i]
    for i in range(0, 15, 2):
        states[i] = states[i] - min_x

    for i in range(1, 16, 2):
        if (min_y > states[i]):
            min_y = states[i]
    for i in range(1, 16, 2):
        states[i] = states[i] - min_y