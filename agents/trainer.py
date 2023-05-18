from utils.batch import Batch
import numpy as np
import torch

class Trainer():
    """ Class for training the RL agent. """

    def __init__(self, env, agent, cost):
        """ Create the gym enviroment and agent. """
        self.env = env
        self.agent = agent
        self.cost = cost

    def train(self, iterations):
        """ Main training loop per GCL.

        Algorithm 1 from the paper.

        Inputs:
        iterations - number of iterations to train for
        """
        # Initialize q_k(τ) as either a random initial controller or from demonstrations
        d_demo = Batch("expert_data/expert_cartpole.npy")
        d_samp = Batch()
#        for i in range(50):
#            self.agent.pretrain(d_demo.states[:,:4], d_demo.states[:,4:5])
        # for iteration i = 1 to I:
        max_states = 1000
        max_states_per_traj = 1000
        for i in range(iterations):
            print(f"Iteration={i}")

            # Generate samples Dtraj from qk(τ )
            _, _ = self.agent.generate_samples(self.env, 10000, max_states_per_traj, self.cost, True)
            d_traj, costs = self.agent.generate_samples(self.env, max_states, max_states_per_traj, self.cost)

            # Append samples: Dsamp ← Dsamp ∪ Dtraj
            d_samp.extend(d_traj)

            # Use Dsamp to update cost cθ using Algorithm 2
            self.cost.non_linear_ioc(d_demo, d_samp)

            # Update qk(τ ) using Dtraj and the method from (Levine & Abbeel, 2014) to obtain qk+1(τ )
            states = torch.tensor(d_traj.states, dtype=torch.float32)
            self.agent.update(torch.tensor(d_traj.states, dtype=torch.float32), costs, None)

        # return optimized cost parameters θ and trajectory distribution q(τ)

    def save_networks(self, save_folder):
        self.cost.save(save_folder + "/cost.pt")
        self.agent.save(save_folder + "/agent.pt")