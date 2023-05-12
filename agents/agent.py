from utils.batch import Batch
import utils.utils as utils

class Agent(nn.module):
    """ Agent class that gives actions based on current state. """

    def __init__(self, action_size, state_size, hidden_layer_size, hidden_layers):
        """ Init network and optimizer. """
        super(Agent, self).__init__()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.net = utils.generate_simple_network(state_size, action_size, hidden_layer_size, hidden_layers)

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
            self.net.forward(states)
            probs = torch.softmax(states)
            action = torch.argmax(logits)
        else:
            logits = self.net.forward(states).detach().numpy()
            probs = nn.functional.softmax(logits)
            action = np.argmax(logits)

        return action, probs

    def update(self, states, actions, rewards):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ)] − H(τ)

        Inputs:
        states - sequence of observed states
        actions - sequence of performed actions
        rewards - sequence of rewards from the performed actions
        """
        entropy = forward(states) * log(forward(states))
        minimize(rewards - entropy)

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
