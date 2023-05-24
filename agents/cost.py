import utils.utils as utils
import torch
import numpy as np

class Cost():
    """ Class for approximating the cost function. """

    K = 10 # Number of update steps to perform each iteration

    def __init__(self, action_size, state_size, hidden_layer_size, hidden_layers):
        """ Initialize the network and optimizer. """
        self.net = utils.generate_simple_network(state_size + action_size, 1, hidden_layer_size, hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def non_linear_ioc(self, d_demo, d_samp):
        """ Non-linear IOC with stochastic patterns.

        Algorithm 2 from paper.
        """
        for iter in range(self.K):
            # Sample demonstration batch Dˆdemo ⊂ Ddemo
            d_s_demo = d_demo.sample(50)

            # Sample background batch Dˆsamp ⊂ Dsamp
            d_s_samp = d_samp.sample(50)
            # Append demonstration batch to background batch:
            # Dˆsamp ← Dˆdemo ∪ Dˆsamp
            d_s_samp.extend(d_s_demo)
            # Estimate dLIOC dθ (θ) using Dˆdemo and Dˆsamp
            samp_probs = d_s_samp.probs
            samp_probs_t = torch.tensor(samp_probs, dtype=torch.float32)

            # z = [1/k * Sigma_k(qκ(τ))]^-1
            # L_ioc = 1/N * Sigma_demo(cost(τ)) + log( 1/M * Sigma_samp(z * exp(-cost(τ)) ) )
            samp_costs = self.get_cost(torch.tensor(d_s_samp.states, dtype=torch.float32))
            demo_costs = self.get_cost(torch.tensor(d_s_demo.states, dtype=torch.float32))

            ioc_lik = torch.mean( demo_costs ) + torch.log( torch.mean( torch.exp( -samp_costs ) / (samp_probs_t + 1e-7)) )

            # Update parameters θ using gradient dLIOC dθ (θ)
            self.optimizer.zero_grad()
            ioc_lik.backward()
            self.optimizer.step()

    def get_cost(self, x):
        """ Get the cost of a given state-action pair. """
        return torch.sigmoid(self.forward(x))

    def forward(self, x):
        """ Send data through the network. """
        return self.net.forward(x)

    def save(self, path):
        """ Save the model.

        Inputs:
        path - path to file to save network in
        """
        torch.save(self.net.state_dict(), path)