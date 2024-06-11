## rlbase

This is a codebase that implements simple reinforcement learning algorithms in JAX. It also has support for several environments. The idea is to have solid single-file implementations of various RL algorithms for research use. This codebase contains both online and offline methods.

Online Algorithms Implemented:
- Proximal Policy Optimization (PPO): `algs_online/ppo.py`
- Soft Actor-Critic (SAC): `algs_online/sac.py`
- Twin Delayed DDPG (TD3): `algs_online/td3.py`

Offline Algorithms Implemented:
- Behavior Cloning (BC): `algs_offline/bc.py`
- Implicit Q-Learning (IQL): `algs_offline/iql.py`

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

We've provided a set of stable results comparing each algorithm to a reference implementation. See this [wandb report]() for the full training curves.

You can reproduce these results using the commands available at `run_baselines.py`.
The basic starting point is to run the individual file, e.g.
```
python algs_online/ppo.py --env_name walker_walk --agent.gamma 0.99
```

### History

This code is based largely off the [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) repo, and takes inspiration also from [jaxrl](https://github.com/ikostrikov/jaxrl) and [cleanrl](https://github.com/vwxyzjn/cleanrl). 