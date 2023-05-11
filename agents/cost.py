class Cost(nn.module):

    def __init__(self, action_size, state_size, n_layers):
        super(PolicyNetwork, self).__init__()
        self.optimizer = torch.optim.Adam(self.parameters())

    def non_linear_ioc(self, d_demo, d_samp):
        """ Non-linear IOC with stochastic patterns.

        Algorithm 2 from paper.
        """
        for iter in range(K):
            # Sample demonstration batch Dˆdemo ⊂ Ddemo
            d_s_demo = d_demo.sample()
            # Sample background batch Dˆsamp ⊂ Dsamp
            d_s_samp = d_samp.sample()
            # Append demonstration batch to background batch:
            # Dˆsamp ← Dˆdemo ∪ Dˆsamp
            d_s_samp.extend(d_s_demo)
            # Estimate dLIOC dθ (θ) using Dˆdemo and Dˆsamp
            samp_states = torch.tensor(d_s_samp.states)
            samp_probs = torch.tensor(d_s_samp.probs)
            demo_states = torch.tensor(d_s_demo.states)

            # z = [1/k * Sigma_k(qκ(τ))]^-1 -> samp_probs
            # L_ioc = 1/N * Sigma_demo(cost(τ)) + log( 1/M * Sigma_samp(z * exp(-cost(τ)) ) )
            samp_costs = self.get_cost(samp_states)
            demo_costs = self.get_cost(demo_states)
            ioc_lik = torch.mean( demo_costs ) + torch.log( torch.mean( torch.exp( samp_cost ) / samp_probs ) )
            # Update parameters θ using gradient dLIOC dθ (θ)
            self.optimizer.zero_grad()
            ioc_lik.backwards()
            self.optimizer.step()

    def get_cost(x):
        return forward(x)

    def forward(x):
        