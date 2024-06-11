## rlbase

This is a codebase that implements simple reinforcement learning algorithms in JAX. It also has support for several environments. The idea is to have solid single-file implementations of various RL algorithms for research use. This codebase contains both online and offline methods.

Online Algorithms Implemented:
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO): `algs_online/ppo.py`
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC): `algs_online/sac.py`
- [Twin Delayed DDPG](https://arxiv.org/abs/1802.09477) (TD3): `algs_online/td3.py`

Offline Algorithms Implemented:
- [Behavior Cloning](https://www.semanticscholar.org/paper/A-Framework-for-Behavioural-Cloning-Bain-Sammut/1f4731d5133cb96ab30e08bf39dffa874aebf487) (BC): `algs_offline/bc.py`
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) (IQL): `algs_offline/iql.py`

Environments Supported:
- (Online) Gym Mujoco Locomotion: `HalfCheetah-v2, CartPole-v1, etc`
- (Online) Deepmind Control: `cheetah_run, pendulum_swingup, etc`
- (Offline) D4RL Mujoco Locomotion: `halfcheetah-medium-expert-v2, etc`
- (Offline) D4RL AntMaze + Goal Conditioned: `antmaze-large-diverse-v2, gc-antmaze-large-diverse-v2`
- (Offline) ExORL: `exorl_cheetah_walk, etc` 
- See `envs/env_helper.py` for full list

### Instllation

For the cleanest installation, create a conda environment:
```
conda env create -f deps/environment.yml
```
You can also refer to the singularity script in `deps/base_container.def` for full reproducability.

### Reproduction

We've provided a set of stable results comparing each algorithm to a reference implementation. For full training curves, see the [wandb reports for online results](https://wandb.ai/kvfransmit/rlbase_benchmark/reports/rlbase_stable-Online-Results--Vmlldzo4Mjk3OTEw) and the [wandb reports for offline results](https://wandb.ai/kvfransmit/rlbase_benchmark/reports/rlbase_stable-Offline-Results--Vmlldzo4Mjk4MDYw).

You can reproduce these results using the commands available at `run_baselines.py`.
The basic starting point is to run the individual file, e.g.
```
python algs_online/ppo.py --env_name walker_walk --agent.gamma 0.99
```

Offline Results
| Env                                | Best Performance (ours) | Original Performance (reference paper) |
| :--------------------------------- | :---------------------: | ---------------------------------: |
| exorl_cheetah_run                  |   257.5 (IQL-DDPG)  | ~250 (TD3) [source (exorl)](https://arxiv.org/pdf/2201.13425)|
| exorl_walker_run                   |   471.9 (IQL-DDPG)  | ~200 (TD3) [source (exorl)](https://arxiv.org/pdf/2201.13425)|
| halfcheetah-medium-expert-v2       |   83.8 (IQL)        | 90.7 (TD3+BC) [source (iql)](https://arxiv.org/pdf/2110.06169)|
| walker2d-medium-expert-v2          |   106.8 (BC)        | 110.1 (TD3+BC) [source (iql)](https://arxiv.org/pdf/2110.06169)|
| hopper-medium-expert-v2            |   98.9 (IQL)        | 98.7 (CQL) [source (iql)](https://arxiv.org/abs/2004.07219)|
| gc-antmaze-large-diverse-v2        |   52.5 (IQL)        | 50.7 (IQL) [source (hiql)](https://arxiv.org/abs/2307.11949)|
| gc-maze2d-large-v1                 |   97.5 (IQL)        | N/A |

Online Results
| Env                                | Best Performance (ours)   | Best Performance (reference paper) |
| :--------------------------------- | :---------------------: | ---------------------------------: |
| HalfCheetah-v2                     | 11029 (SAC)               | 12138.8 (SAC) [source (tianshou)](https://github.com/thu-ml/tianshou)|
| Walker2d-v2                        | 5101.8 (SAC-Tianshoulike) | 5007 (SAC)[source (tianshou)](https://github.com/thu-ml/tianshou)|
| Hopper-v2                          | 2714.4 (REDQ)             | 3542.2 (SAC)[source (tianshou)](https://github.com/thu-ml/tianshou)|
| cheetah_run                        | 918.9 (REDQ)              | 800 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)|
| walker_run                         | 835.7 (TD3)               | 700 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)|
| hopper_hop                         | 474.9 (TD3)               | 210 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)||
| quadruped_run                      | 920.8 (TD3)               | 700 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)||
| humanoid_run                       | 211.8 (REDQ)              | 90 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)||
| pendulum_swingup                   | 790.2 (SAC)               | 920 (SAC) [source (pytorch-sac)](https://github.com/denisyarats/pytorch_sac)||

### History

This code is based largely off the [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) repo, and takes inspiration also from [jaxrl](https://github.com/ikostrikov/jaxrl) and [cleanrl](https://github.com/vwxyzjn/cleanrl). 
