# DQN_Pong
Play OpenAI Gym game of Pong using Deep Q-Learning



## Environments

* Windows 10
* GPU: Nvidia GeForce 1070
* CUDA Version: 11.1
* Python 3.7.7
* Pytorch 1.7.0
* tensorboard 2.3.0
* gym 0.17.3
* atari-py 1.2.2

Environments set up (on Windows):

```bash
# install gym
pip insatll gym

# install atari-py (on Windows)
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

Install `ffmpeg` (used for recording video in `gym`) on Windows:

* download [ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2020-12-07-12-50/ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3.zip) form https://github.com/BtbN/FFmpeg-Builds/releases
* unzip it and add a environment path of `ffmpeg-n4.3.1-26-gca55240b8c-win64-gpl-4.3/bin`
* open a terminal, type `ffmpeg -version` to check if you've successfully installed `ffmpeg`

