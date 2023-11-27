import os
import gym
import numpy as np
import argparse
import malware_rl
import pickle
import torch

from tensorboardX import SummaryWriter
from utils import Memory
from ICMPPO import ICMPPO

class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

render = False
test_episodes = 400      # max training episodes
max_timesteps = 60    # max timesteps in one episode
ppo_dir= './saved_model/malconv/icm_ppo_999.pt'    # Where to store ppo model
icm_dir= './saved_model/malconv/icm_icm_999.pt'    # Where to store ppo model
log_dir= './logs/'           # Where to store tensorboard logs
# Initialize icmppo

writer = SummaryWriter(log_dir)
memory = Memory()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self,):
        return self.action_space.sample()

def main():

    print('---------- start testing ------------')

    env_test = gym.make('ember-test-v0')

    agent = ICMPPO(writer=writer, state_dim=env_test.observation_space.shape[0], action_dim=env_test.action_space.n, device=device)

    agent.policy.load_state_dict(torch.load(ppo_dir))
    agent.icm.load_state_dict(torch.load(icm_dir))

    # agent = RandomAgent(env_test.action_space) # random agent

    max_turn = env_test.maxturns

    evasions = 0
    evasion_history = {}

    for i in range(test_episodes):
        total_reward = 0
        done = False
        state = env_test.reset()
        sha256 = env_test.sha256
        num_turn = 0

        while num_turn < max_turn:

            # action = agent.choose_action() # random agent

            action = agent.policy_old.act(np.array(state), memory)
            state_, reward, done, info = env_test.step(action)
            total_reward += reward


            num_turn = env_test.turns
            if done and reward >= 10.0:
                evasions += 1
                evasion_history[sha256] = info
                break

            elif done:
                break


    # Output metrics/evaluation stuff
    evasion_rate = (evasions / test_episodes) * 100
    print(f"{evasion_rate}% samples evaded model.")

    # write evasion_history to txt file
    # file = open('history_malconv.txt', 'w')
    # for k, v in evasion_history.items():
    #     file.write(str(k) + ' ' + str(v) + '\n')
    # file.close()



if __name__ == '__main__':
    main()
