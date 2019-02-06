# This Repository is Reinforcement Learning Agent FramWork

This repository is designed to provide an easy deeo reinforcement learning framework for those studying deep reinforcement learning.

This framework is based on a tensor flow, and you can use it if you want to import the model you want into model.py.

We provide a tutorial to train the agent for the environment, and tutorials by action and input shape are provided as follows.

```
Continuous Action MLP - bipedalwalker, pendulum
Discrete Action MLP - LunarLander
Discrete Action CNN - Breakout
```

Our tutorial is being done in the gym environment provided by openai and you need to install the openai gym and box2d to run the tutorial code.

## Installation

cpu version

```
pip install tensorflow-rl[tf-cpu]
```

gpu version
```
pip install tensorflow-rl[tf-gpu]
```

## Requirements

```
tensorflow
box2d
gym
numpy
tensorboardX
```

## Implemented

- [x] Vanilla Policy Gradient
- [x] Advantage Actor Critic
- [x] Proximal Policy Optimization
- [x] Deep Deterministic Policy Gradient
- [ ] Trust Region Policy Optimization
- [ ] Actor Critic Experience Replay
- [ ] Soft Actor Critic

## Demonstration

### 1. Continuous Action BipedalWalker  

* Script : bipedalwalker_ppo.py, bipedalwalker_ddpg.py  
* Environment : BipedalWalker-v2 
* Orange : ppo, Blue: ddpg
* Episode : 2165
* Image : PPO

###### BipedalWalker
<div align="center">
  <img src="sources/bipedalwalker_ppo.gif" width="50%" height='300'><img src="sources/bipedalwalker.png" width="50%" height='300'>
</div>

### 2. Continuous Action Pendulum

* Script : pendulum_ppo.py, pendulum_ddpg.py  
* Environment : Pendulum-v0
* Orange : ddpg, Blue: ppo
* Episode : 300
* Image : PPO

###### Pendulum
<div align="center">
  <img src="sources/pendulum.gif" width="50%" height='300'><img src="sources/pendulum.png" width="50%" height='300'>
</div>

### 3. Discrete Action CNN Breakout

* Script : breakout_rollout_a2c.py, breakout_rollout_ppo.py, breakout_rollout_vpg.py
* Environment : BreakoutDeterministic-v4 with Multi-processing
* Orange : a2c, Blue : ppo, Red : vpg
* Episode : 600
* Image : PPO

###### Breakout
<div align="center">
  <img src="sources/breakout.gif" width="30%" height='300'><img src="sources/breakout.png" width="50%" height='300'>
</div>

### 4. Discrete Action MLP LunarLander

* Script : lunarLander_rollout_a2c.py, lunarLander_rollout_ppo.py, lunarLander_rollout_vpg.py
* Environment : LunarLander-v2 with Multi-processing
* Orange : a2c, Blue : ppo, Red : vpg
* Episode : 350
* Image : PPO

###### LunarLander
<div align="center">
  <img src="sources/lunarlander.gif" width="50%" height='300'><img src="sources/lunarlander.png" width="50%" height='300'>
</div>


## Member

- [https://github.com/chagmgang](https://github.com/chagmgang)
- [https://github.com/SunandBean](https://github.com/SunandBean)
- [https://github.com/TinkTheBoush](https://github.com/TinkTheBoush)

## License

We do not have the copyright to this repository.

Please 'just' use these code and just 'refer' the url of repository in any form.

[MIT License](./LICENSE)

## Reference

[1] [mario_rl](https://github.com/jcwleo/mario_rl)

[2] [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

[3] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)

[4] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

[5] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

[6] [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)

[7] [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

### Please fork this repository and contribute to strengthen the tensorflow reinforcement learning ecosystem

### Support us in any form. Thank you

Content us to [chagmgang@gmail.com](chagmgang@gmail.com)
