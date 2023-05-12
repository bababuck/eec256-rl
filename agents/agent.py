from utils.batch import Batch
import utils.utils as utils

class Agent(nn.module):
    """ Agent class that gives actions based on current state. """

    def __init__(self, action_size, state_size, hidden_layer_size, hidden_layers):
        """ Init network and optimizer. """
        super(Agent, self).__init__()
        self.net = utils.generate_simple_network(state_size, action_size, hidden_layer_size, hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters())

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
            logits = self.net.forward(states)
        else:
            logits = self.net.forward(states).detach()

        probs = torch.softmax(logits)
        action = torch.argmax(logits)
        return action, probs

    def update(self, states, actions, rewards, probs):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        actions - sequence of performed actions
        rewards - sequence of rewards from the performed actions
        probs - the probability of each action
        """
        
        entropy = probs * torch.log(probs)
        loss = entropy - rewards
        optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def generate_samples(self, env, max_states, max_states_per_trajectory):
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
            for _ in range(max_t):
                states_visited += 1
                action, prob = self.get_action(state, false)
                states.append(states)
                actions.append(action)
                probs.append(prob)

                state, reward, done = env.step(action)
                rewards.append(reward) # These will be meaningless since we don't model the reward directly, instead estimate later with cost function
                if done:
                    break

        return Batch(states=states, actions=actions, probs=probs)

    def forward(self, x):
         return self.net.forward(x)
