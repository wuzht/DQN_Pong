import time
import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import ExperienceReplay, Agent, Logger, choose_gpu
from gym_wrappers import make_env
from model import DQN

# Environments
cur_dir = os.path.join('./exp', datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S'))
if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)
log = Logger(os.path.join(cur_dir, 'train.log')).logger
device = (torch.device('cuda:{}'.format(choose_gpu()[0])) if 'linux' in sys.platform else torch.device('cuda:0')) \
    if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameters
DEFAULT_ENV_NAME   = "PongNoFrameskip-v4"   # identify the Environment to train on
MEAN_REWARD_BOUND  = 19.0                   # reward boundary to stop training
gamma              = 0.99                   # the discount factor
batch_size         = 32                     # the minibatch size
replay_size        = 10000                  # the replay buffer size (maximum number of experiences stored in replay memory)
learning_rate      = 1e-4                   # the learning rate
sync_target_frames = 1000                   # indicates how frequently we sync model weights from the main DQN network to the target DQN network (how many frames in between syncing)
replay_start_size  = 10000                  # the count of frames (experiences) to add to replay buffer before starting training

eps_start = 1.0
eps_decay = 0.999985
eps_min   = 0.02

for key, val in {k: v for k, v in globals().items() if (type(v) == int or type(v) == float or type(v) == str) and not k.startswith('_')}.items():
    log.critical('{}: {}'.format(key, val))

# Training
env = make_env(DEFAULT_ENV_NAME)
net = DQN(env.observation_space.shape, env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
agent = Agent(env, ExperienceReplay(replay_size))
writer = SummaryWriter(log_dir=cur_dir)

epsilon = eps_start
total_rewards = []
frame_idx = 0  
best_mean_reward = None

log.critical(">>>Training starts at {}".format(datetime.datetime.now()))
while True:
    frame_idx += 1
    epsilon = max(epsilon * eps_decay, eps_min)

    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-100:])

        log.info("{}: {} games, mean reward {:.3f}, (epsilon {:.3f})".format(frame_idx, len(total_rewards), mean_reward, epsilon))
        
        writer.add_scalar("epsilon", epsilon, len(total_rewards))
        writer.add_scalar("reward_100", mean_reward, len(total_rewards))
        writer.add_scalar("reward", reward, len(total_rewards))
        writer.flush()

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), os.path.join(cur_dir, "best.pt"))
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                log.info("Best mean reward updated {:.3f}".format(best_mean_reward))

        if mean_reward > MEAN_REWARD_BOUND:
            log.critical("Solved in {} frames!".format(frame_idx))
            break

    if len(agent.exp_buffer) < replay_start_size:
        continue

    batch = agent.exp_buffer.sample(batch_size)
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * gamma + rewards_v

    loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()

    if frame_idx % sync_target_frames == 0:
        target_net.load_state_dict(net.state_dict())
    
writer.close()
log.critical(">>>Training ends at {}".format(datetime.datetime.now()))
