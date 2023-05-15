from agents.trainer import Trainer
from agents.agent import Agent
from agents.cost import Cost
from env.control import ControlEnv

if __name__ == '__main__':
    env = ControlEnv()
    agent = Agent()
    cost = Cost()
    trainer = Trainer(env, agent, cost)
    trainer.train()
    trainer.save_networks(networks_folder)
    agent.generate_samples(env)