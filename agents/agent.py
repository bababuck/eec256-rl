from utils.batch import Batch
import utils.utils as utils
import torch
import torch.nn as nn
import numpy as np


class Agent:
    """ Agent class that gives actions based on current state. """

    def __init__(self, discrete_action_size, cont_action_size, state_size, hidden_layers,
                 cont_action_min=-1, cont_action_max=1, lr=0.001):
        """
        Init network and optimizer.
        Parameters
        ----------
        discrete_action_size : int
            Dimension of the continuous part of the action space.
        cont_action_size : int
            Dimension of the continuous part of the action space.
        cont_action_min, cont_action_max : float or 1-D array-like
            The bound of the continuous part of the action. If is array，
            size should be (cont_action_size,)l
            Used to generate the random actions.
        state_size : int
            Dimension of the state space.
        hidden_layers : 1-D array-like
            The size of each hidden layer.
        lr : float
            The learning rate.
        """
        self.discrete_action_size = discrete_action_size
        self.cont_action_size = cont_action_size
        self.cont_action_min = cont_action_min
        self.cont_action_max = cont_action_max
        self.cum_loss = []
        # Continuous action based on NAF in https://arxiv.org/pdf/1603.00748.pdf
        # For each discrete action, the output is a lower triangular matrix L of size m
        # and a vector μ of size m. There is also an additional value estimator.
        n = discrete_action_size
        m = cont_action_size
        n_outputs_per_bin = (m + 1) * m // 2 + m + 1
        n_outputs = n * n_outputs_per_bin

        layers = []
        prev_size = state_size
        for size in hidden_layers:
            layers += [nn.Linear(prev_size, size), nn.ReLU(inplace=True)]
            prev_size = size
        layers.append(nn.Linear(prev_size, n_outputs))
        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_random_action(self, _state):
        """
        Return a random action.
        The discrete action is chosen uniformly, and the continuous action is sampled
        uniformly between `cont_action_min` and `cont_action_max`.
        Parameters
        ----------
        _state : 1-D array-like
            The observed state. ignored.

        Returns
        -------
        action : 1-D ndarray
            The random action. The first dimension is the discrete action index, and the
            rest are continuous action
        """

        output = self.net(_state)
        _, mus, qs = self.reformat_output(output)
        mus = mus.detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        if qs[0] < 0 and qs[1] < 0:
            qs[0] = -1/qs[0]
            qs[1] = -1 / qs[1]
        elif qs[0] < 0:
            qs[0] = qs[1]/100  # Like greedy
        elif qs[1] < 0:
            qs[1] = qs[0]/100
        prob = qs / qs.sum()
        discrete_action_idx = np.random.choice(2, p=prob)
        cont_action_mean = mus[discrete_action_idx]
        # Assumes Random continuous action is normal distributed
        cont_action = [np.random.normal(cont_action_mean[0], 0.01), np.random.normal(cont_action_mean[1], 0.01)]
        action = np.concatenate(([discrete_action_idx], cont_action))
        return action, prob

    def reformat_output(self, output):
        """
        Reformat the network output. Gradients are kept.
        Parameters
        ----------
        output: 1-D Tensor
            Network output. Size (n * ((m + 1) * m // 2 + m + 1),)
        Returns
        -------
        Ls: 3-D Tensor
            Lower triangular matrices. Size (n, m, m). Each (m, m) matrix is lower triangular.
        mus: 2-D Tensor
            Action mean. Size (n, m).
        qs: 1-D Tensor
            Q-value estimator for each discrete action. Size (n,).
        """
        n = self.discrete_action_size
        m = self.cont_action_size
        n_triangular_entries = (m + 1) * m // 2

        Ls = torch.zeros((n, m, m))
        tril_indices = torch.tril_indices(m, m)
        Ls[:, tril_indices[0], tril_indices[1]] = torch.reshape(output[:n * n_triangular_entries],
                                                                (n, n_triangular_entries))
        mus = torch.reshape(output[n * n_triangular_entries: n * (n_triangular_entries + m)],
                            (n, m))
        qs = output[n * (n_triangular_entries + m):]
        assert qs.shape == (n,)
        return Ls, mus, qs

    def get_policy_action(self, state):
        """
        Return the best action based on the input state based on the agent's estimation.
        Parameters
        ----------
        state : 1-D array-like
        The observed state.

        Returns
        -------
        action : 1-D ndarray
            The policy's best action. The first dimension is the discrete action index,
            and the rest are continuous action.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        if state.device != self.device:
            state = state.to(self.device)
        output = self.net(state)
        _, mus, qs = self.reformat_output(output)
        mus = mus.detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        prob = qs / qs.sum()
        discrete_action_idx = np.argmax(qs)
        cont_action = mus[discrete_action_idx]
        action = np.concatenate(([discrete_action_idx], cont_action))
        return action, prob

    def get_q_values(self, state, discrete_action, cont_action):
        """
        Return the Q-value for each discrete action.
        Parameters
        ----------
        state : 1-D or 2-D array-like
            The observed state.
        discrete_action : int or 1-D array-like
            The discrete part of the action.
        cont_action : 1-D or 2-D array-like
            The continuous part of the action.
        Returns
        -------
        """
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        if state.device != self.device:
            state = state.to(self.device)

        # if not isinstance(discrete_action, torch.Tensor):
            # discrete_action = torch.Tensor(discrete_action)
        # if discrete_action.device != self.device:
            # discrete_action = discrete_action.to(self.device)

        if not isinstance(cont_action, torch.Tensor):
            cont_action = torch.Tensor(cont_action)
        if cont_action.device != self.device:
            cont_action = cont_action.to(self.device)

        output = self.net.forward(state)
        Ls, mus, qs = self.reformat_output(output)
        L = Ls[discrete_action]
        mu = mus[discrete_action]
        q = qs[discrete_action]

        return q - 0.5 * (cont_action - mu) @ L @ L.T @ (cont_action - mu)

    def update(self, states, actions, costs, env):
        """
        Update the Q-value estimation from on one step.
        Parameters
        ----------
        states : 2-D array-like
            The observed states.
        actions : 2-D array-like
            The actions. The first column are the discrete action indices,
            and the rest are continuous action.
        costs : 1-D array-like
            As reward
        """

        discrete_actions = actions[:, 0]
        cont_actions = actions[:, 1:]
        n_states = states.shape[0]
        targets = torch.zeros(n_states)
        q = torch.zeros(n_states)

        # One more step
        action_next, prob_next = self.get_random_action(states[-1])
        states_next, reward_next, done_next = env.step(int(action_next[0]), action_next[1:3])
        q_next = self.get_q_values(states_next, int(action_next[0]), action_next[1:3]) * 100
        c = torch.zeros(n_states)

        costs = torch.tensor(costs, dtype=torch.float32)
        for i in range(n_states):
            q[i] = self.get_q_values(states[i], int(discrete_actions[i]), cont_actions[i]) * 100
            c[i] = -costs[i]  # Cost is negative reward
        for i in range(n_states - 1):
            targets[i] = q[i + 1]
        targets[n_states - 1] = q_next
        targets = targets + c  # Gamma = 1
        mse = nn.MSELoss()
        loss = mse(q, targets)
        print("\n q: ", q)
        print("\n targets: ", targets)
        print("\n loss: ", loss)
        self.cum_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pretrain(self, states, truth):
        """ Pre-train the agent in a supervised learning fashion.

        Supervised training will not
        """
        raise NotImplemented
        # states = torch.tensor(states, dtype=torch.float32)
        # truth = torch.tensor(truth, dtype=torch.float32)
        # mse_loss = torch.nn.MSELoss()
        # probs = self.get_probs(states, True)
        # loss = mse_loss(probs, truth)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def generate_samples(self, env, max_states, max_states_per_trajectory):
        """ Generate a set of sample trajectories from the environment.

        Inputs:
        env - ControlEnv type
        max_states - number of total states to visit
        max_states_per_trajectory - max time steps for a single rollout

        Outputs:
        batch - batch containing the rewards, states, and actions
        """
        states, actions, probs = [], [], []
        states_visited = 0
        while states_visited < max_states:
            env.reset()
            state = env.get_rope_states()
            cum_prob = 1
            action, prob = self.get_random_action(torch.tensor(state, dtype=torch.float32))
            # action = int(action[0])
            for i in range(max_states_per_trajectory):
                states.append(state)
                actions.append(action)
                cum_prob = prob[int(action[0])]
                probs.append(cum_prob)
                states_visited += 1
                state, reward, done = env.step(int(action[0]), action[1:3])
                action, prob = self.get_random_action(torch.tensor(state, dtype=torch.float32))
                # action = int(action[0])
                if done:
                    break
        return Batch(states=states, actions=actions, probs=probs)

    def test(self, env, num_test):
        """ Run a test of the model on the environment.

        Use the on-policy actions.
        """
        for t in range(num_test):
            steps = 0
            done = False
            env.reset()
            state = env.get_rope_states()
            while not done:
                steps += 1
                action, prob = self.get_policy_action(torch.tensor(state, dtype=torch.float32))
                state, _, done = env.step(int(action[0]), action[1:3])
            print(f"Test number {t}: {steps} steps reached")

    def forward(self, x):
        """ Run data through the network. """
        return self.net.forward(x)

    def save(self, path):
        """ Save the model. """
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
