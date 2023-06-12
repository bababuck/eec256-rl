from utils.discrete_batch import Batch
import numpy as np
import torch

class Trainer():
    """ Class for training the RL agent. """

    max_states = 35
    max_states_per_traj = 35

    def __init__(self, env, agent, cost):
        """ Create the gym enviroment and agent. """
        self.env = env
        self.agent = agent
        self.cost = cost

    def train(self, iterations, expert_data):
        """ Main training loop per GCL.

        Algorithm 1 from the paper.

        Inputs:
        iterations - number of iterations to train for
        """
        # Initialize q_k(τ) as either a random initial controller or from demonstrations
        d_demo = expert_data
        d_samp = Batch()

        # Initialize from demonstrations
        for i in range(200):
            d_s_demo = d_demo.sample(20)
            self.agent.pretrain_pick(d_s_demo.states[:,:14], d_s_demo.states[:,14:18])
            self.agent.pretrain(d_s_demo.states[:,:14], d_s_demo.states[:,14:18])

        # for iteration i = 1 to I:
        for i in range(iterations):
 
            print(f"Iteration={i}")

            # Generate samples Dtraj from qk(τ )
            d_traj = self.agent.generate_samples(self.env, self.max_states, self.max_states_per_traj)

            # Append samples: Dsamp ← Dsamp ∪ Dtraj
            d_samp.extend(d_traj)

            # Use Dsamp to update cost cθ using Algorithm 2
            self.cost.non_linear_ioc(d_demo, d_samp, i, self.agent)

            # Find the cost for updating the networks
            seg_costs = [[self.cost.get_pick_cost(torch.tensor(d_traj.states[s][:12].tolist()+a, dtype=torch.float32)).detach().item() for a in [[1,0], [0,1]]] for s in range(np.shape(d_traj.states)[0])]
            costs = [[self.cost.get_cost(torch.tensor(d_traj.states[s][:14].tolist()+a, dtype=torch.float32)).detach().item() for a in [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]] for s in range(np.shape(d_traj.states)[0])]

            # Update qk(τ ) using Dtraj and the method from (Levine & Abbeel, 2014) to obtain qk+1(τ ) 
            states = torch.tensor(d_traj.states, dtype=torch.float32)
            self.agent.update_pick(d_traj.states, seg_costs)
            self.agent.update(torch.tensor(d_traj.states, dtype=torch.float32), costs, None)

            # Logging
#            if i % 20 == 0:
#                self.save_networks('networks', agent_net_name=f"agent_big_it_{i}.pt")

        # return optimized cost parameters θ and trajectory distribution q(τ)

    def save_networks(self, save_folder, cost_net_name="cost.pt", agent_net_name="agent.pt"):
        """ Save the networks for cost and agent to specified path.

        Inputs:
        save_folder - path to directory in which to store the network files
        cost_net_name - name of file to save cost network to
        agent_net_name - name of file to save agent=t network to
        """
        self.cost.save(save_folder + "/" + cost_net_name)
        self.agent.save(save_folder + "/" + agent_net_name)
        self.agent.save_pick(save_folder + "/pick" + agent_net_name)

    def load_networks(self, save_folder, cost_net_name="cost.pt", agent_net_name="agent.pt"):
        """ Save the networks for cost and agent to specified path.

        Inputs:
        save_folder - path to directory in which to store the network files
        cost_net_name - name of file to save cost network to
        agent_net_name - name of file to save agent=t network to
        """
        self.agent.load(save_folder + "/" + agent_net_name)
        self.agent.load_pick(save_folder + "/pick" + agent_net_name)