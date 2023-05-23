from utils.batch import Batch
import utils.utils as utils
import torch
import numpy as np

class Agent():
    """ Agent class that gives actions based on current state. """

    def __init__(self, action_size, state_size, hidden_layer_size, hidden_layers):
        """ Init network and optimizer. """
        self.net = utils.generate_simple_network(state_size, action_size, hidden_layer_size, hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def get_action(self, state, training):
        """ Return an action based on the current state.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - action to perform
        probs - probability of each_action
        """
        if training:
            logits = self.net.forward(state)
        else:
            logits = self.net.forward(state).detach()

        probs = torch.softmax(logits,-1)
        action = -1
        if not training:
            action = np.random.choice(2,  p = probs.detach().numpy())
        else:
            action = np.argmax(probs.detach().numpy())
        return action, probs

    def update(self, states, cost, probs):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        rewards - sequence of rewards from the performed actions
        """
        actions, probs = self.get_action(states[:,:4], True)
        actions = states[:,4:5]
        log_probs = torch.log(probs+1e-7)
        cost = torch.tensor(cost,dtype=torch.float32)
        cost = torch.mean(probs * cost, dim=-1)
        entropy = torch.mean(torch.mul(probs, log_probs), dim=-1)
        loss = torch.mean(cost - entropy*1e-3)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pretrain(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Supervised training will not
        """
        states = torch.tensor(states, dtype=torch.float32)
        truth = torch.tensor(truth, dtype=torch.float32)
        mse_loss = torch.nn.MSELoss()
        _, probs = self.get_action(states, True)
        loss = mse_loss(probs, truth)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_samples(self, env, max_states, max_states_per_trajectory, cost_net):
        """ Generate a set of sample trajectories from the enviroment.

        Inputs:
        env - ControlEnv type
        max_states - number of total states to visit
        max_states_per_trajectory - max timesteps for a single rollout

        Outputs:
        batch - batch containing the rewards, states, and actions
        """
        rewards, states, actions, probs = [], [], [], []
        states_visited = 0
        while states_visited < max_states:
            state = env.reset()
            c_rewards, c_states, c_actions, c_probs = [], [], [], []
            action, prob = self.get_action(torch.tensor(state), False)
            for i in range(max_states_per_trajectory):
                states_visited += 1
                state, reward, done = env.step(action)
                action, prob = self.get_action(torch.tensor(state), False)
                c_states.append(state)
                c_actions.append(action)
                c_probs.append(prob.numpy()[action])
                    
                cost = [cost_net.get_cost(torch.tensor(state.tolist()+[0], dtype=torch.float32)).detach().item(), cost_net.get_cost(torch.tensor(state.tolist()+[1], dtype=torch.float32)).detach().item()]
                c_rewards.append(cost)
                if done:
                    break

            states += c_states
            actions += c_actions
            probs += c_probs
            rewards += c_rewards

        return Batch(states=states, actions=actions, probs=probs), rewards

    def forward(self, x):
         return self.net.forward(x)

    def save(self, path):
        """ Save the model"""
        torch.save(self.state_dict(), path)
