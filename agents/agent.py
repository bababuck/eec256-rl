from utils.batch import Batch
import utils.utils as utils
import torch
import numpy as np

class Agent():
    """ Agent class that gives actions based on current state. """

    def __init__(self, dir_action_size, seg_action_size, state_size, hidden_layer_size, hidden_layers):
        """ Init network and optimizer. """ 
        self.net = utils.generate_simple_network(state_size, dir_action_size, hidden_layer_size, hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.pick_net = utils.generate_simple_network(state_size-seg_action_size, seg_action_size, hidden_layer_size, hidden_layers)
        self.pick_optimizer = torch.optim.Adam(self.pick_net.parameters(), lr=0.001)

        self.dir_action_size = dir_action_size
        self.seg_action_size = seg_action_size

        # For logging if desired
        self.pick_loss = []
        self.loss = []
        self.entropy = []
        self.cost = []
        self.pick_entropy = []
        self.pick_cost = []

    def get_random_action(self, state):
        """ Return a direction based on the current state randomly chosed from model distribution.

        Inputs:
        state - observable state

        Outputs:
        action - direction to move
        probs - probability of each_action
        """
        probs = self.get_probs(state, False)
        action = np.random.choice(self.dir_action_size, p = probs.numpy())
        return action, probs

    def get_policy_action(self, state):
        """ Return a direction based on the current state chosen as most probable action from distribution.

        Inputs:
        state - observable state

        Outputs:
        action - direction to move
        probs - probability of each_action
        """
        probs = self.get_probs(state, False)
        action = np.argmax(probs.numpy())
        return action, probs

    def get_probs(self, state, training):
        """ Return probability distribution based on current state for direction to move.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        probs - probability of each_action
        """
        logits = self.net.forward(state)
        probs = torch.softmax(logits,-1)
        if not training:
            probs = probs.detach()
        return probs

    def get_pick_probs(self, state, training):
        """ Return probability distribution based on current state for segment to pick.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - segment to pick
        probs - probability of each_action
        """
        logits = self.pick_net.forward(state)
        probs = torch.softmax(logits,-1)
        if not training:
            probs = probs.detach()
        return probs


    def get_random_pick(self,state):
        """ Return a segment based on the current state randomly chosed from model distribution.

        Inputs:
        state - observable state

        Outputs:
        action - segment to grab
        probs - probability of each_action
        """
        state = torch.tensor(state[:12], dtype=torch.float32)
        probs = self.get_pick_probs(state, False).numpy()
        seg = np.random.choice(self.seg_action_size, p = probs)
        return seg, probs[seg]

    def update(self, states, cost, probs):
        """ Update policy function for direction based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        cost - sequence of costs of the performed actions
        """
        cost = np.array(cost)

        # Doesn't change the gradient but easier to see issues
        mean = cost.mean(axis=1)
        mean = mean.reshape((-1, 1))
        cost = cost - mean

        probs = self.get_probs(states[:,:14], True)
        log_probs = torch.log(probs+1e-7)
        costs = torch.tensor(cost,dtype=torch.float32)        
        costs = torch.mean(probs * costs, dim=-1)
        entropy = -torch.mean(torch.mul(probs, log_probs), dim=-1)
        loss = torch.mean(costs - entropy)

        # Logging
        self.cost.append(torch.mean(costs).item())
        self.entropy.append(-torch.mean(entropy).item())
        self.loss.append(loss.item())            

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_pick(self, states, cost):
        """ Update policy function for segment picking based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        cost - sequence of costs from the performed actions
        """
        cost = np.array(cost)

        # Doesn't change the gradient but easier to see issues
        mean = cost.mean(axis=1)
        mean = mean.reshape((-1, 1))
        cost = cost - mean

        states = torch.tensor(states[:, :12], dtype=torch.float32)
        probs = self.get_pick_probs(states, True)
        log_probs = torch.log(probs+1e-7)

        costs = torch.tensor(cost,dtype=torch.float32)        
        costs = torch.mean(probs * costs, dim=-1)
        entropy = -torch.mean(torch.mul(probs, log_probs), dim=-1)
        loss = torch.mean(costs - entropy)

        # Logging
        self.pick_cost.append(torch.mean(costs).item())
        self.pick_entropy.append(-torch.mean(entropy).item())
        self.pick_loss.append(loss.item())

        self.pick_optimizer.zero_grad()
        loss.backward()
        self.pick_optimizer.step()

    def pretrain(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Doesn't have much effect on quality of results.
        """
        tstates = torch.tensor(states, dtype=torch.float32)
        truth = torch.tensor(truth, dtype=torch.float32)
        mse_loss = torch.nn.MSELoss()
        probs = self.get_probs(tstates, True)
        loss = mse_loss(probs, truth)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pretrain_pick(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Doesn't have much effect on quality of results.
        """
        tstates = torch.tensor(states[:,:12], dtype=torch.float32)
        truth = torch.tensor(states[:,12:14], dtype=torch.float32)
        mse_loss = torch.nn.MSELoss()
        probs = self.get_pick_probs(tstates, True)
        loss = mse_loss(probs, truth)
        
        self.pick_optimizer.zero_grad()
        loss.backward()
        self.pick_optimizer.step()

    def generate_samples(self, env, max_states, max_states_per_trajectory):
        """ Generate a set of sample trajectories from the enviroment.

        Inputs:
        env - ControlEnv type
        max_states - number of total states to visit
        max_states_per_trajectory - max timesteps for a single rollout

        Outputs:
        batch - batch containing the probabilities, states, and actions
        """
        states, actions, probs, pick_probs = [], [], [], []
        states_visited = 0

        while states_visited < max_states:
            state = env.reset()

            seg, seg_prob = self.get_random_pick(state)
            state[seg+12] = 1
            action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))
            for i in range(max_states_per_trajectory):
                states.append(state)
                actions.append(action)
                probs.append(prob.numpy()[action])
                pick_probs.append(seg_prob)
                states_visited += 1

                state, reward, done = env.step((seg*3), action) # Segment network outputs 0 and 1, but need to pick either 0 or 3

                seg, seg_probs = self.get_random_pick(state)
                state[seg+12] = 1
                env.render()
                action, prob = self.get_random_action(torch.tensor(state,dtype=torch.float32))

                if done or states_visited > max_states:
                    break

        utils.transform_action(actions, states, probs)
        pick_probs=pick_probs * 16 # Unchanged regardless of transformation
        return Batch(states=states, actions=actions, probs=probs, pick_probs=pick_probs)

    def forward(self, x):
        """ Run data through the network. """
        return self.net.forward(x)

    def save(self, path):
        """ Save the direction model. """
        torch.save(self.net.state_dict(), path)

    def save_pick(self,path):
        """ Save the segment model. """
        torch.save(self.pick_net.state_dict(), path)

    def load(self, path):
        """ Load the direction model. """
        self.net.load_state_dict(torch.load(path))

    def load_pick(self, path):
        """ Load the segment model. """
        self.pick_net.load_state_dict(torch.load(path))