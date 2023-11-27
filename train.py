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

render = False
solved_reward = 10     # stop training if avg_reward > solved_reward
log_interval = 1000     # print avg reward in the interval
max_episodes = 1000      # max training episodes
max_timesteps = 60    # max timesteps in one episode
update_timestep = 64
# Replay buffer size, update policy every n timesteps
log_dir= './logs/'           # Where to store tensorboard logs

# Initialize log_writer, memory buffer, icmppo
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
    env = gym.make('malconv-train-v0')

    agent = ICMPPO(writer=writer, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)

    # agent = RandomAgent

    # agent.load_models(500)

    timestep = 0

    log_episode_reward = []
    log_episode_mreward = []
    total_rewards, avg_rewards, epsilon_history = [], [], []
    for episode in range(max_episodes):
        episode_rewards = 0
        done = False
        state = env.reset()
        while not done:
            timestep += 1

            action = agent.policy_old.act(np.array(state), memory)
            state, reward, done, info = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                agent.update(memory, timestep)
                memory.clear_memory()

            episode_rewards += reward

        log_episode_reward.append(episode_rewards)
        log_episode_mreward.append(sum(log_episode_reward[-20:]) / 20)
        if (episode + 1) % 100 == 0:
            print("########## Saved! ##########")
            torch.save(agent.policy.state_dict(), './saved_model/ember/noicm_ppo_{}.pt'.format(episode))
            torch.save(agent.icm.state_dict(), './saved_model/ember/noicm_icm_{}.pt'.format(episode))

        # logging

        if (episode + 1) % 20 == 0:
            print(
                'Episode {} \t average episode reward: {} \t'.format(episode, sum(log_episode_reward[-20:]) / 20))
            writer.add_scalar('Mean_extr_reward_per_20_episodes',
                              sum(log_episode_reward[-20:]) / 20,
                              timestep
                              )


    print('---------- train over ------------')



if __name__ == '__main__':
    main()
