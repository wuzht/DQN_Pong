# The DQN model
# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/dqn_model.py
import torch
import torch.nn as nn
import numpy as np
from gym_wrappers import make_env


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


if __name__ == "__main__":
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
    device = torch.device("cuda")
    
    test_env = make_env(DEFAULT_ENV_NAME)
    test_net = DQN(test_env.observation_space.shape, test_env.action_space.n).to(device)
    print(test_net)
    # DQN(
    # (conv): Sequential(
    #     (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    #     (1): ReLU(inplace=True)
    #     (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    #     (3): ReLU(inplace=True)
    #     (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    #     (5): ReLU(inplace=True)
    # )
    # (fc): Sequential(
    #     (0): Linear(in_features=3136, out_features=512, bias=True)
    #     (1): ReLU(inplace=True)
    #     (2): Linear(in_features=512, out_features=6, bias=True)
    # )
    # )
