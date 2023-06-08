import os

import numpy as np
import torch

from utils.batch import Batch


class Trainer:
    """ Class for training the RL agent. """

    def __init__(self, env, agent, cost):
        """ Create the gym environment and agent. """
        self.env = env
        self.agent = agent
        self.cost = cost

    def train(self, iterations, expert_data_path):
        """ Main training loop per GCL.

        Algorithm 1 from the paper.

        Inputs:
        iterations - number of iterations to train for
        """
        # Initialize q_k(τ) as either a random initial controller or from demonstrations
        d_demo = Batch(expert_data_path)
        d_samp = Batch()

        # for iteration i = 1 to I:
        max_states = 20
        max_states_per_traj = 20
        for i in range(iterations):
            print(f"Iteration={i}")

            # Generate samples Dtraj from qk(τ)
            d_traj = self.agent.generate_samples(self.env, max_states, max_states_per_traj)
            one_hot_list = [[int(i == j) for i in range(32)] for j in range(32)]
            costs = [[self.cost.get_cost(
                torch.tensor(d_traj.states[s][:16].tolist(), dtype=torch.float32)).detach().item()]
                     for s in range(np.shape(d_traj.states)[0])]

            # Append samples: Dsamp ← Dsamp ∪ Dtraj
            d_samp.extend(d_traj)
            # Use Dsamp to update cost cθ using Algorithm 2
            self.cost.non_linear_ioc(d_demo, d_samp)
            print("\n Costs: \n", costs)
            # Update qk(τ) using Dtraj and the method from (Levine & Abbeel, 2014) to obtain qk+1(τ)
            states = torch.tensor(d_traj.states, dtype=torch.float32)
            n_states = states.shape[0]
            state_next, reward_next, done_next = self.env.step(int(d_traj.actions[n_states-1][0]))
            target_next = [state_next, reward_next]
            self.agent.update(torch.tensor(d_traj.states, dtype=torch.float32), d_traj.actions, target_next)
            self.agent.test(env=self.env, num_test=3)

        # return optimized cost parameters θ and trajectory distribution q(τ)
        self.save_networks('./model')

    def save_networks(self, save_folder, cost_net_name="cost.pt", agent_net_name="agent.pt"):
        """ Save the networks for cost and agent to specified path.

        Inputs:
        save_folder - path to directory in which to store the network files
        cost_net_name - name of file to save cost network to
        agent_net_name - name of file to save agent=t network to
        """
        self.cost.save(os.path.join(save_folder, cost_net_name))
        self.agent.save(os.path.join(save_folder, agent_net_name))