# DQN_Pong
Play OpenAI Gym game of [Pong](https://gym.openai.com/envs/Pong-v0/) using Deep Q-Learning

[TOC]

## 0 文件结构

```sh
DQN_Pong
│  train.py				# 训练代码
│  utils.py				# ExperienceReplay类, Agent类等
│  gym_wrappers.py		# OpenAI Gym Wrappers
│  model.py				# DQN模型代码
│  test.py				# 测试代码，加载模型并对其测试，并录制的游戏测试视频
|
│  report.pdf			# 实验报告
│  video.mp4			# 录制的游戏测试视频
│
└─exp					# 各次实验的tensorboard log, train log, 以及模型
   ├─1209-215314				# 最好的实验
   │      events.out.tfevents	# tensorboard log
   │      train.log				# 训练log
   │      best.pt				# 模型
   │
   ├─1209-215341
   ├─1209-215353
   ├─1209-215432
   └─1209-215446
```



## 1 Environments

* Windows 10 or Ubuntu
* GPU: Nvidia GeForce GTX 1080
* CUDA Version: 10.1
* Python 3.8.3
* Pytorch 1.7.0
* tensorboard 2.4.0
* gym 0.17.3
* atari-py 0.2.6

**Windows**

安装 `gym` 和 `atari`：

```bash
pip install gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

安装 `ffmpeg` (用于在 `gym` 中录制视频)：

* 从 https://github.com/BtbN/FFmpeg-Builds/releases 下载 [ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2020-12-07-12-50/ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3.zip)
* 解压缩，并添加该路径至环境变量：`ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3/bin`
* 打开命令行，输入 `ffmpeg -version` 看是否成功安装 `ffmpeg`

**Ubuntu**

安装 atari 之前，必须安装 `cmake` 和 `zlib`：

> [Problem installing Atari](https://github.com/openai/gym/issues/779)

```bash
sudo apt install cmake
sudo apt install zlib1g-dev
```

然后安装 `gym` 和 `atari`：

```bash
pip insatll gym
pip install gym[atari]
```

安装 `ffmpeg`

```sh
sudo apt-get install ffmpeg
```



## 2 Pong

[Pong](https://gym.openai.com/envs/Pong-v0/) 是 Atari 的一款乒乓球游戏。Pong 的界面由简单的二维图形组成，当玩家将球打过去，对手没有把球接住并打回来时，玩家得 1 分。当一个玩家达到 21 分时，一集 (episode) 结束。在 OpenAI Gym 框架版本的 Pong 中，Agent 显示在右侧，对手显示在左侧。

![pong](assets/pong.gif)

（图片来自 [[2]](#2)）

在 Pong 环境中，一个 Agent（玩家）可以采取三个动作：保持静止、垂直向上平移和垂直向下平移。但是，如果我们使用`action_space.n` 方法，我们可以发现环境有6个动作：

```python
import gym
import gym.spaces
DEFAULT_ENV_NAME = “PongNoFrameskip-v4”
test_env = gym.make(DEFAULT_ENV_NAME)
print(test_env.action_space.n)
6
print(test_env.unwrapped.get_action_meanings())
[‘NOOP’, ‘FIRE’, ‘RIGHT’, ‘LEFT’, ‘RIGHTFIRE’, ‘LEFTFIRE’]
```

其中三个是多余的（FIRE 等于 NOOP，LEFT 等于 LEFTFIRE，RIGHT 等于 RIGHTFIRE）。



## 3 Deep Q-network (DQN)

### 3.1 Input

Atari 游戏以 210 × 160 像素的分辨率显示，每个像素有 128 种可能的颜色：

```python
print(test_env.observation_space.shape)
(210, 160, 3)
```

游戏中的每一帧图像是 210 × 160 × 3 的 RGB 图像。为了降低复杂性，我们可以将帧转换为灰度，并将它们缩小到一个 84 × 84 像素的正方形块。但是，仅仅根据一帧图像，我们无法知道球朝哪个方向移动。解决办法是保留一些过去的观察结果，并将它们作为一个状态使用。在Atari 游戏中，在 [[1]](#1) 中作者建议将4个后续帧叠加在一起，作为每个状态下的观察值。因此，预处理将四个帧堆叠在一起，最终状态空间大小为 84 × 84 × 4：

![input](assets/input.png)

（图片来自 [[2]](#2)）

### 3.2 Output

与传统强化学习设置（一次只产生一个 Q 值）不同DQN 在一次前向传播中为环境中的每个可能的动作 (action) 产生一个Q值：

![output](assets/output.png)

（图片来自 [[2]](#2)）

这种方法通过一次网络前向计算就得到了所有 action 的 Q 值，避免了每个 action 都必须单独运行网络，显著提高了速度。这样，我们可以通过选择这个输出的向量的最大值来执行一个动作。

### 3.3 Network Architecture

网络结构如下所示，网络由 3 个卷积层以及 2 个全连接层组成，每层之间的激活函数为 ReLU。

```python
DQN(
  (conv): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU(inplace=True)
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU(inplace=True)
  )
  (fc): Sequential(
    (0): Linear(in_features=3136, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=512, out_features=6, bias=True)
  )
)
```



## 4 OpenAI Gym Wrappers

在 DeepMind 的论文 [[1]](#1) 中，为了提高方法的速度和收敛性，在 Atari 平台交互中应用了几种 transformations (如将帧转换为灰度，并将其缩小到 84 × 84 像素的正方形块)。本实验中使用了 OpenAI-Gym 模拟器，其中的 transformations 使用 OpenAI Gym wrappers 来实现，共使用了下面几个 wrappers：

* `FireResetEnv`
* `MaxAndSkipEnv`
* `ProcessFrame84`
* `BufferWrapper`
* `ImageToPyTorch`
* `ScaledFloatFrame`

Pong 游戏需要用户按 FIRE 按钮来启动游戏。`FireResetEnv` 会按下 `FIRE` 按钮让游戏开始：

```python
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
```

我们需要的下一个 wrapper 是 `MaxAndSkipEnv`，它为 Pong 编写了两个重要的转换代码如下所示。`MaxAndSkipEnv` 负责完成下面两项工作：

* 使用每 N 个 (默认为 4 个) observations 作为一个 observation 来显著加快训练速度，并将其作为一个 step 中的 observation 返回。这是因为在中间帧上，所选择的动作是简单重复的，而且我们可以每 N 步做出一个动作决策。因为如果让网络对每一帧都处理，这样计算量非常高，而且相邻帧之间的差别通常很小。

* 取最后两帧中每个像素的最大值并将其作为观察值。

```python
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
```

下一个 wrapper 是 `ProcessFrame84`，代码如下所示。它对图片进行一些处理，最后得到 84 × 84 大小的图像。步骤如下：

1. 将 RGB 图片转化为单通道灰度图
2. resize 到 110 × 84 大小
3. 裁剪图像中不相关的部分

```python
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)
```

`BufferWrapper` 将几个（通常是四个）连续帧堆叠 (stack) 在一起并返回：

```python
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
```

最后两个 wrappers 是 `ImageToPyTorch` 和 `ScaledFloatFrame`。

* `ImageToPyTorch` 将 observation 的形状从 HWC (height, width, channel) 更改为 PyTorch 所需的CHW (channel, height, width) 格式
* `ScaledFloatFrame` 将图像数据类型转换为浮点类型，并将值缩放到范围 [0.0, 1.0]

```python
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
```

最后，使用下面 `make_env` 函数通过环境的名称创建一个环境，并对其应用所有必需的 wrappers：

```python
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
```



## 5 Challenges in Deep Reinforcement Learning

不幸的是，强化学习在使用神经网络表示 action-values 时更不稳定，尽管应用了上面的 wrappers。训练这样一个网络需要大量的数据，但即使这样，也不能保证网络收敛。事实上，由于 actions 和 states 之间的高度相关性，网络权值会出现振荡或发散的情况。

为了解决这个问题，在本实验中使用了 DQN 中的两种技术：

* Experience Replay
* Target Network

### 5.1 Experience Replay

我们试图用神经网络逼近一个复杂的非线性函数 Q(s, a)。为此，我们必须使用 Bellman 方程计算目标，然后当作有监督学习来处理。然而，SGD 优化的一个基本要求是训练数据是独立同分布 (independent and identically distributed) 的，当 Agent 与环境交互时，经验元组的序列通常会高度相关。从这些经验元组中按顺序学习的朴素 Q-learning 算法有可能受到这种相关性影响的影响。

一种解决办法就是 **experience replay**。它创建一个固定大小的 **experience buffer** (aka **replay buffer**)，这个 buffer 包含一系列的 experience tuples (S, A, R, S′)。当 Agent 与环境交互时，元组会逐渐添加到 buffer 中。buffer 是一个固定长度的 queue，在 buffer 的末尾添加新的数据，若 buffer 已满，则移除最旧的 experience tuple。

总而言之，experience replay 的基本思想是存储过去的 experiences，然后使用这些 experiences 的随机子集 (即一个 batch) 来更新 Q-network，而不是仅仅使用最近的一次 experience。

在训练中，experience buffer 的大小设为 10000, batch size 设为 32。也就是说，从最近的 10000 条 experience 中随机选 32 条来进行一次 training iteration。

### 5.2 Target Network

在 Q-learning 中，我们用猜测更新猜测 (update a guess with a guess)，这可能导致有害的关联。Bellman 方程通过 Q(s’, a’) 为我们提供了 Q(s, a) 的值。然而，这两个 states (s 和 s’) 之间只有一步。这使得它们非常相似，网络很难区分它们。

为了使训练更稳定，有一个技巧，叫做 **target network**，我们保留一个网络的副本，并将其用于计算 Bellman 方程中的 Q(s’, a’) 值。

也就是说，这个被称为 target network 的第二个 Q-network 的预测 Q 值被用来反向传播和训练 main Q-network。需要强调的是，target network 的参数是不需要训练的，而是与 main Q-network 的参数周期性同步的。其思想是利用 target network 的 Q 值对 main Q-network 进行训练，可以提高训练的稳定性。



## 6 Deep Q-Learning Algorithm

Deep Q-Learning 算法的伪代码如下所示：

![algorithm](assets/algorithm.png)

（图片来自 [[2]](#2)）

在 Deep Q-Learning 算法中，training loop 有两个主要的阶段。

* 阶段 1：
  * 将当前 state `s` 输入网络 `Q`，得到所有动作对应的 Q 值输出
  * 用 ϵ−greedy 方法在当前 Q 值输出中选择 action `a`
  * 在 state `s` 中执行 action `a`，得到 reward `r` 、next state `s'` 和是否终止状态 `done`
  * 将五元组 `(s, a, r, s', done)` 存入 experience replay buffer `D` 中
* 若 `D` 中有足够的 experience tuples，则执行阶段 2：
  * 从 `D` 中随机采样 N 个样本 (a mini-batch) 对网络进行训练，其中损失函数为均方误差
  * 每运行 C 步，将 main network 中的参数复制到 target network



## 7 Hyperparameters

超参数设置及超参数的意义如下所示。以下是实验的默认超参数。

```python
MEAN_REWARD_BOUND = 19.0		# 近100次episode的mean reward达到19.0即停止训练
gamma = 0.99					# discount factor
batch_size = 32					# minibatch size
replay_size = 10000				# replay buffer size
learning_rate = 1e-4			# learning rate
sync_target_frames = 1000		# 每运行多少帧将main network中的参数复制到target network
replay_start_size = 10000		# replay buffer中有多少帧才开始训练

eps_start = 1.0					# epsilon的初始值为eps_start
eps_decay = 0.999985			# 在开始一轮迭代前，将当前epsilon乘以eps_decay
eps_min = 0.02					# epsilon最小值为eps_min

optimizer = optim.Adam(net.parameters(), lr=learning_rate)	# 优化器为Adam
```



## 8 实验结果

### 8.1 默认超参数的实验结果

使用上一节 [7 Hyperparameters](#7 Hyperparameters) 中的默认超参数进行训练，实验结果如下：

<table>
    <td width=33.3%><center><img src="assets/epsilon.svg" ></center><center>(a) epsilon</center></td>
    <td width=33.3%><center><img src="assets/reward.svg" ></center><center>(b) reward</center></td>
    <td width=33.3%><center><img src="assets/reward_100.svg" ></center><center>(c) reward_100</center></td> 
</table>
<center>Fig 1. 默认超参数的训练过程。其中，各图横坐标为 games (episodes)。(a) epsilon 的变化图。(b) reward 的变化图。(c) 近 100 个 games (episodes) 的平均 reward 变化图。</center>

运行 `test.py` 代码对该模型进行测试，**测试结果的 reward 为 21**，视频见文件 `video.mp4`。

### 8.2 不同超参数对实验结果的影响

使用控制变量法，对超参数进行了调整，观察这些超参数的设置对 Agent 性能的影响。选择对比的超参数有 `learning_rate` 和 `sync_target_frames`。

在本实验中，training loop 会一直循环，直到近 100 个 episode 的 mean reward 达到某个设定的数值 (在本实验中设为 19.0) 才会停止。所以，当训练完成 (若模型收敛) 时，Agent 的 mean reward 都是 19.0，故下面要比较的是不同超参数设置对收敛速度的影响，而这个收敛速度可根据模型达到收敛需要多少个 games 和 frames 得到。

#### 8.2.1 不同 learning_rate 对实验结果的影响

在默认参数的基础上，控制其他超参数不变，改变 `learning_rate`，实验结果如下：

| exp         | learning_rate | games | frames  | mean reward | 是否收敛 | 测试 reward |
| ----------- | ------------- | ----- | ------- | ----------- | -------- | ----------- |
| 1209-215314 | 1e-4          | 406   | 771807  | 19.03       | 是       | 21          |
| 1209-215341 | 1e-3          | 2494  | 1998641 | -20.95      | 否       | -21         |
| 1209-215353 | 1e-5          | 1377  | 3036438 | 19.01       | 是       | 21          |

其中表头的意义如下：

* exp: 实验序号
* games: 模型共训练了多少个 games
* frames: 模型共训练了多少个 frames
* mean reward: 训练结束时近 100 个 episode 的 mean reward
* 测试 reward: 运行 `test.py` 加载模型并对其进行测试

训练过程中的 epsilon, reward, mean reward 变化曲线如下：

<table>
    <td width=33.3%><center><img src="assets/lr_epsilon.svg" ></center><center>(a) epsilon</center></td>
    <td width=33.3%><center><img src="assets/lr_reward.svg" ></center><center>(b) reward</center></td>
    <td width=33.3%><center><img src="assets/lr_reward_100.svg" ></center><center>(c) reward_100</center></td> 
</table>
<left>Fig 2. 在默认参数的基础上，控制其他超参数不变，改变 learning_rate 对实验的影响。其中，各图横坐标为 games (episodes)，橙色，蓝色，红色线的 learning_rate 分别为 1e-4, 1e-3, 1e-5。(a) epsilon 的变化图。(b) reward 的变化图。(c) 近 100 个 games (episodes) 的平均 reward 变化图。</left>

从结果中可知，学习率太大会导致模型不收敛，学习率太小又会导致训练太慢。当 `learning_rate = 1e-3` ，模型不能收敛，手动中断了其运行。在本实验中， `learning_rate = 1e-4` 时能获得相对较好的结果。

#### 8.2.2 不同 sync_target_frames 对实验结果的影响

在默认参数的基础上，控制其他超参数不变，改变 `sync_target_frames`，实验结果如下：

| exp         | sync_target_frames | games | frames  | mean reward | 是否收敛 | 测试 reward |
| ----------- | ------------------ | ----- | ------- | ----------- | -------- | ----------- |
| 1209-215314 | 1000               | 406   | 771807  | 19.03       | 是       | 21          |
| 1209-215432 | 100                | 741   | 1382139 | 19.03       | 是       | 21          |
| 1209-215446 | 10000              | 1375  | 2730985 | 19.03       | 是       | 21          |

训练过程中的 epsilon, reward, mean reward 变化曲线如下：

<table>
    <td width=33.3%><center><img src="assets/stf_epsilon.svg" ></center><center>(a) epsilon</center></td>
    <td width=33.3%><center><img src="assets/stf_reward.svg" ></center><center>(b) reward</center></td>
    <td width=33.3%><center><img src="assets/stf_reward_100.svg" ></center><center>(c) reward_100</center></td> 
</table>
<left>Fig 3. 在默认参数的基础上，控制其他超参数不变，改变 sync_target_frames 对实验的影响。其中，各图横坐标为 games (episodes)，橙色，浅蓝色，粉色线的 sync_target_frames 分别为 1000, 100, 10000。(a) epsilon 的变化图。(b) reward 的变化图。(c) 近 100 个 games (episodes) 的平均 reward 变化图。</left>

超参数 `sync_target_frames` 指的是每运行多少帧将 main network 中的参数复制到 target network，选好这一参数对于模型训练非常重要。从结果中可知，在本实验中 `sync_target_frames = 1000` 时能获得相对较好的结果。



## References

* <span id="1">[1]</span> [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* <span id="2">[2]</span> https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af

