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
        self.action_size = action_size

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

    def update(self, states, cost, probs):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        rewards - sequence of rewards from the performed actions
        """
        probs = self.get_probs(states[:,:24], True)
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
        probs = self.get_probs(states, True)
        loss = mse_loss(probs, truth)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_samples(self, env, max_states, max_states_per_trajectory, cost):
        """ Generate a set of sample trajectories from the enviroment.

        Inputs:
        env - ControlEnv type
        max_states - number of total states to visit
        max_states_per_trajectory - max timesteps for a single rollout

        Outputs:
        batch - batch containing the rewards, states, and actions
        """
        states, actions, probs = [], [], []
        states_visited = 0
        while states_visited < max_states:
            state = env.reset()
            # seg = np.random.randint(16,24)
            seg = self.get_e_greedy_seg(cost, state) + 16
            state[seg] = 1
            action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))
#            print(f"seg{seg} action{action}")
            for i in range(max_states_per_trajectory):
                states.append(state)
                actions.append(action)
                probs.append(prob.numpy()[action])

                states_visited += 1
                state, reward, done = env.step(seg - 16, action)
                # seg = np.random.randint(16,24)
                seg = self.get_e_greedy_seg(cost, state) + 16
                state[seg] = 1
                action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))
#                print(f"seg{seg} action{action}")

                if done:
                    break

        return Batch(states=states, actions=actions, probs=probs)

    def get_e_greedy_seg(self, cost, state):
        min_cost = 100000
        seg = -1
        for i in range(8):
            state_list = state.tolist()
            state_list[i+16] = 1
            action, _ = self.get_policy_action(torch.tensor(state_list))
            acts = 4*[0]
            acts[action] = 1
            costs = cost.get_cost(torch.tensor(state_list + acts, dtype=torch.float32)).detach()
            if costs < min_cost:
                min_cost = costs
                seg = i
        return seg


    def test(self, env, num_test, cost):
        """ Run a test of the model on the enviroment.

        Use the on-policy actions.
        """
        for t in range(num_test):
            done = False
            state = env.reset()
            while not done:
                env.render()
                min_cost = 100000
                seg = -1
                for i in range(8):
                    state_list = state.tolist()
                    state_list[i+16] = 1
                    action, _ = self.get_policy_action(torch.tensor(state_list))
                    acts = 4*[0]
                    acts[action] = 1
                    costs = cost.get_cost(torch.tensor(state_list + acts, dtype=torch.float32)).detach()
                    if costs < min_cost:
                        min_cost = costs
                        seg = i
                    print(f"seg{seg} cost{costs}")
                state = state.tolist()
                state[seg+16] = 1
#                print(state)
                action, _ = self.get_policy_action(torch.tensor(state))
#                print(action)
                state, _, done = env.step(seg, action)
#            print(state)

    def forward(self, x):
        """ Run data through the network. """
        return self.net.forward(x)

    def save(self, path):
        """ Save the model. """
        torch.save(self.net.state_dict(), path)
