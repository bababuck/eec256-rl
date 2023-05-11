from utils.batch import Batch

class Agent():
    """ Agent class that gives actions based on current state. """

    def __init__(self):
        """ Init policy and value functions. """

    def get_action(self, state, training):
        """ Return an action based on the current state.

        Inputs:
        state - observable state
        training - boolean if training or evaluating

        Outputs:
        action - action to perform
        """

    def update(self, states, actions, rewards):
        """ Update policy and value functions based on a set of observations.

        minq Eq[cθ(τ )] − H(τ ) 

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
        rewards, states, actions = [], [], []
        states_visited = 0
        while states_visited < max_states:
            state = env.reset()
            for _ in range(max_t):
                states_visited += 1
                action = self.get_action(state, false)
                states.append(states)
                actions.append(action)

                state, reward, done = env.step(action)
                rewards.append(reward) # These will be meaningless since we don't model the reward directly, instead estimate later with cost function
                if done:
                    break

        return Batch(states=states, actions=actions)