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

        segs = 2
        self.pick_net = utils.generate_simple_network(state_size-segs, segs, hidden_layer_size, hidden_layers)
        self.pick_optimizer = torch.optim.Adam(self.pick_net.parameters(), lr=0.001)

        self.action_size = action_size

        self.pick_loss = []
        self.loss = []
        self.entropy = []
        self.cost = []
        self.pick_entropy = []
        self.pick_cost = []

    def get_random_action(self, state):
        """ Return an action based on the current state randomly chosed from model distribution.

        Inputs:
        state - observable state

        Outputs:
        action - action to perform
        probs - probability of each_action
        """
        probs = self.get_probs(state, False)
        action = np.random.choice(self.action_size, p = probs.numpy())
        return action, probs

    def get_policy_action(self, state):
        """ Return an action based on the current state chosen as most probable action from distribution.

        Inputs:
        state - observable state

        Outputs:
        action - action to perform
        probs - probability of each_action
        """
        probs = self.get_probs(state, False)
        action = np.argmax(probs.numpy())
        return action, probs

    def get_probs(self, state, training):
        """ Return probability distribution based on current state.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - action to perform
        probs - probability of each_action
        """
        logits = self.net.forward(state)
        probs = torch.softmax(logits,-1)
        if not training:
            probs = probs.detach()
        return probs

    def get_pick_probs(self, state, training):
        """ Return probability distribution based on current state.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - action to perform
        probs - probability of each_action
        """
        logits = self.pick_net.forward(torch.tensor(state, dtype=torch.float32))
        probs = torch.softmax(logits,-1)
        if not training:
            probs = probs.detach()
        return probs

    def update(self, states, cost, probs):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        rewards - sequence of rewards from the performed actions
        """
        cost = np.array(cost)
        mean = cost.mean(axis=1)
        mean = mean.reshape((-1, 1))
        cost = cost - mean
        print(cost.mean(axis=0))
        for i in range(1):
            probs = self.get_probs(states[:,:14], True)
            log_probs = torch.log(probs+1e-7)
            print("dir")
            print(cost)
            print(probs)
            costs = torch.tensor(cost,dtype=torch.float32)        
            costs = torch.mean(probs * costs, dim=-1)
            entropy = -torch.mean(torch.mul(probs, log_probs), dim=-1)
            self.cost.append(torch.mean(costs).item())
            self.entropy.append(-torch.mean(entropy).item())
            loss = torch.mean(costs - entropy)
            self.loss.append(loss.item())            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_pick(self, states, cost):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        rewards - sequence of rewards from the performed actions
        """

        cost = np.array(cost)
        mean = cost.mean(axis=1)
        mean = mean.reshape((-1, 1))
        cost = cost - mean
        for i in range(1):
            probs = self.get_pick_probs(states[:,:12], True)
            log_probs = torch.log(probs+1e-7)
            print("seg")
            print(cost)
            print(probs)
            costs = torch.tensor(cost,dtype=torch.float32)        
            costs = torch.mean(probs * costs, dim=-1)
            entropy = -torch.mean(torch.mul(probs, log_probs), dim=-1)
            self.pick_cost.append(torch.mean(costs).item())
            self.pick_entropy.append(-torch.mean(entropy).item())
            loss = torch.mean(costs - entropy)
            self.pick_loss.append(loss.item())
            self.pick_optimizer.zero_grad()
            loss.backward()
            self.pick_optimizer.step()

    def pretrain(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Supervised training will not
        """
        tstates = torch.tensor(states, dtype=torch.float32)
        truth = torch.tensor(truth, dtype=torch.float32)
        mse_loss = torch.nn.MSELoss()
        probs = self.get_probs(tstates, True)
        loss = mse_loss(probs, truth)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#        print(loss)

    def pretrain_pick(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Supervised training will not
        """
        tstates = torch.tensor(states[:,:12], dtype=torch.float32)
        truth = torch.tensor(states[:,12:14], dtype=torch.float32)
        mse_loss = torch.nn.MSELoss()
        probs = self.get_pick_probs(tstates, True)
        loss = mse_loss(probs, truth)
        
        self.pick_optimizer.zero_grad()
        loss.backward()
        self.pick_optimizer.step()
#        print(loss)

    def generate_samples(self, env, max_states, max_states_per_trajectory, cost):
        """ Generate a set of sample trajectories from the enviroment.

        Inputs:
        env - ControlEnv type
        max_states - number of total states to visit
        max_states_per_trajectory - max timesteps for a single rollout

        Outputs:
        batch - batch containing the rewards, states, and actions
        """
        states, actions, probs, pick_probs = [], [], [], []
        states_visited = 0
        seg_costs = []
        while states_visited < max_states:
            state = env.reset()
            # seg = np.random.randint(16,24)
            seg, scost, p = self.get_e_greedy_seg(cost, state)
            seg += 12
            state[seg] = 1
            cum_prob = 1
            action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))
#            print(f"seg{seg} action{action}")
            for i in range(max_states_per_trajectory):
                states.append(state)
                actions.append(action)
                cum_prob = prob.numpy()[action]
                probs.append(cum_prob)
                pick_probs.append(p)
                seg_costs.append(scost)
                states_visited += 1
                state, reward, done = env.step((seg - 12)*3, action)
                # seg = np.random.randint(16,24)
                seg, scost, p = self.get_e_greedy_seg(cost, state)
                seg += 12
                state[seg] = 1
                env.render()
                action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))
#                print(f"seg{seg} action{action}")
                if done:
                    break

        utils.transform_action(actions, states, probs)
        return Batch(states=states, actions=actions, probs=probs, pick_probs=pick_probs * 16), seg_costs * 16

    def get_random_pick(self,state):
        probs = self.get_pick_probs(state[:12], False).numpy()
        seg = np.random.choice(2, p = probs)
        return seg, probs[seg]

    def get_e_greedy_seg(self, cost, state):
#        if np.random.randint(0, 10) > 9:
#            return np.random.randint(0,4)
        seg_costs = []
        for i in range(2):
            state_list = state.tolist()
            state_list[i+12] = 1
            costs = cost.get_pick_cost(torch.tensor(state_list, dtype=torch.float32)).detach()
            seg_costs.append(costs.item())
        seg, probs = self.get_random_pick(state)
        return seg, seg_costs, probs

    def forward(self, x):
        """ Run data through the network. """
        return self.net.forward(x)

    def save(self, path):
        """ Save the model. """
        torch.save(self.net.state_dict(), path)

    def save_pick(self,path):
        torch.save(self.pick_net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def load_pick(self, path):
        self.pick_net.load_state_dict(torch.load(path))