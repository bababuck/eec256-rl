class Cost(nn.module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()

    def non_linear_ioc(self, d_demo, d_samp):
        """ Non-linear IOC with stochastic patterns.

        Algorithm 2 from paper.
        """
        for iter in range(K):
            # Sample demonstration batch Dˆdemo ⊂ Ddemo
            demo_sample = d_demo.sample()
            # Sample background batch Dˆsamp ⊂ Dsamp
            samp_sample = d_samp.sample()
            # Append demonstration batch to background batch:
            # Dˆsamp ← Dˆdemo ∪ Dˆsamp
            data = demo_sample + samp_sample
            # Estimate dLIOC dθ (θ) using Dˆdemo and Dˆsamp
            ioc_like = avg(forward(demo_sample)) + log (avg(exp(forward(data))/probs))
            # Update parameters θ using gradient dLIOC dθ (θ)
            ioc_like.backwards()
        # return optimized cost parameters θ