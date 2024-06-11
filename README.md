## rlbase

This is a codebase for implementing simple reinforcement learning algorithms. It also has several environments integrated.
The idea is to have solid single-file implementations of various RL algorithms.
The code is based off `jaxrl`, `jaxrl_m`, `cleanrl`.

To run the code, create a conda environment using the `deps/environment.yml` file. 

To run an algorithm, run a command such as 
```
python algs/ppo.py --env_name walker_walk
```
A list of commands to recrate experiments is available at `script/run_baselines.py`