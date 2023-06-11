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

    def train(self, iterations, expert_data):
        """ Main training loop per GCL.

        Algorithm 1 from the paper.

        Inputs:
        iterations - number of iterations to train for
        """
        # Initialize q_k(τ) as either a random initial controller or from demonstrations
        d_demo = expert_data
        d_samp = Batch()
        new_d_demo_states = np.zeros((len(d_demo.states), 14))
        new_d_demo_states[:, :12] = d_demo.states
        for j in range(len(d_demo.states)):
            # action_state = np.array([0, 0])
            if d_demo.actions[j][0] == 0:
                new_d_demo_states[j][12] = 1
                new_d_demo_states[j][13] = 0
            else:
                new_d_demo_states[j][12] = 0
                new_d_demo_states[j][13] = 1
        d_demo.states = new_d_demo_states
        # for iteration i = 1 to I:
        for i in range(iterations):
            max_states = 20  # 35
            max_states_per_traj = 20

            print(f"Iteration={i}")

            # Generate samples Dtraj from qk(τ )
            d_traj = self.agent.generate_samples(self.env, max_states, max_states_per_traj)

            # Add action to state
            new_d_traj_states = np.zeros((len(d_traj.states), 14))
            new_d_traj_states[:, :12] = d_traj.states
            for j in range(len(d_traj.states)):
                # action_state = np.array([0, 0])
                if d_traj.actions[j][0] == 0:
                    new_d_traj_states[j][12] = 1
                    new_d_traj_states[j][13] = 0
                else:
                    new_d_traj_states[j][12] = 0
                    new_d_traj_states[j][13] = 1
            d_traj.states = new_d_traj_states

            # Append samples: Dsamp ← Dsamp ∪ Dtraj
            d_samp.extend(d_traj)

            # Use Dsamp to update cost cθ using Algorithm 2
            self.cost.non_linear_ioc(d_demo, d_samp, i, self.agent)
            # Only add discrete, treat the continuous as we always get mean
            # Not sure about add a [0,1], [1,0]
            costs = [[self.cost.get_cost(torch.cat((torch.tensor(d_traj.states[s][:14], dtype=torch.float32), torch.tensor(d_traj.actions[s][1:3], dtype=torch.float32)), 0)).detach().item()] for s in range(np.shape(d_traj.states)[0])]
            print("\n Costs: ", costs)
            # Update qk(τ ) using Dtraj and the method from (Levine & Abbeel, 2014) to obtain qk+1(τ )
            states = torch.tensor(d_traj.states[:, :12], dtype=torch.float32)
            self.agent.update(states, d_traj.actions, costs, self.env)
            if i % 20 == 0:
                self.save_networks('networks', agent_net_name=f"agent_last_{i}.pt")

        # return optimized cost parameters θ and trajectory distribution q(τ)

    def save_networks(self, save_folder, cost_net_name="cost_new.pt", agent_net_name="agent.pt"):
        """ Save the networks for cost and agent to specified path.

        Inputs:
        save_folder - path to directory in which to store the network files
        cost_net_name - name of file to save cost network to
        agent_net_name - name of file to save agent=t network to
        """
        self.cost.save(save_folder + "/" + cost_net_name)
        self.agent.save(save_folder + "/" + agent_net_name)

    def load_networks(self, save_folder, cost_net_name="cost.pt", agent_net_name="agent.pt"):
        """ Save the networks for cost and agent to specified path.

        Inputs:
        save_folder - path to directory in which to store the network files
        cost_net_name - name of file to save cost network to
        agent_net_name - name of file to save agent=t network to
        """
        self.agent.load(save_folder + "/" + agent_net_name)