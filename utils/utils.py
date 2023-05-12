import pytorch as torch

def generate_simple_network(input_size, output_size, hidden_size, hidden_layers):
    """ Generate a simple feed forward network of given dimensions. """
    # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
    layers = [nn.Linear(state_size, layers_size), nn.ReLU()]
    layers += [*x for x in [nn.Linear(layers_size, layers_size), nn.ReLU()] for _ in range(self(hidden_layers))]
    layers.append(nn.Linear(layers_size, action_size))
    return nn.Sequential(*layers)
