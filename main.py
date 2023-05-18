from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv

if __name__ == '__main__':
    env = ControlEnv('CartPole-v1')
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    hidden_layer_size = 32
    hidden_layers = 1
    agent = Agent(action_size, state_size, hidden_layer_size, hidden_layers)
    cost = Cost(action_size, state_size, hidden_layer_size, hidden_layers)
    trainer = Trainer(env, agent, cost)
    iterations = 400
    trainer.train(iterations)

    networks_folder = "netowrks"
    trainer.save_networks(networks_folder)
    agent.generate_samples(env)