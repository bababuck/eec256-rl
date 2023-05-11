from env.control import Env
from agent import Agent
from cost import Cost

class Trainer():
    """ Class for training the RL agent. """

    def __init__(self, params):
        """ Create the gym enviroment and agent. """
        self.env = ControlEnv()
        self.agent = Agent()
        self.cost = Cost()

    def training_loop(self, rollouts):
        """ Train the agent.

        Inputs:
        rollouts - number of iterations to train for
        """
        for itr in range(iterations):
            self.simulate()
            
    def simulate():
        """ Run one rollout. """

    def training_loop():
        """ Main training loop per GCL.

        Algorithm 1 from the paper.
        """
        # Initialize q_k(τ) as either a random initial controller or from demonstrations
        d_demo = Batch("demo_path.npy")
        d_samp = Batch()
        # for iteration i = 1 to I:
        for i in range(iterations):
            # Generate samples Dtraj from qk(τ )
            d_traj = self.agent.generate_samples(self.env)
            # Append samples: Dsamp ← Dsamp ∪ Dtraj
            d_samp.extend(d_traj)
            # Use Dsamp to update cost cθ using Algorithm 2
            self.cost.non_linear_ioc(d_demo, d_samp)
            # Update qk(τ ) using Dtraj and the method from (Levine & Abbeel, 2014) to obtain qk+1(τ )
            self.agent.update()
        # return optimized cost parameters θ and trajectory distribution q(τ )
