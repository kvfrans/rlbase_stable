job_list = []
project = 'rlbase_baselines'
online_envs = ['HalfCheetah-v2', 'Walker2d-v2', 'Pendulum-v1', 'MountainCarContinuous-v0', 'walker_walk', 'cheetah_run', 'humanoid_run', 'quadruped_run', 'pointmass_easy']
offline_envs = ['antmaze-large-diverse-v2', 'maze2d-large-v1', 'gc-maze2d-large-v1', 'gc-antmaze-large-diverse-v2', 'halfcheetah-medium-expert-v2', 'walker2d-medium-expert-v2', 'hopper-medium-expert-v2', 'exorl_walker_run', 'exorl_cheetah_run']

# SAC
for seed in range(4):
    base = f"python algs_online/sac.py --wandb.project {project} --wandb.group SAC --seed {seed} "
    for env in online_envs:
        job_list.append(base + f"--env_name {env}")

# SAC-Tianshoulike
# https://github.com/thu-ml/tianshou/tree/master
# Tianshou has strong results numbers, so this runs SAC with the same hyperparameters as Tianshou.
for seed in range(4):
    base = f"python algs_online/sac.py --wandb.project {project} --wandb.group SAC-Tianshoulike --seed {seed} "
    base += "--agent.target_entropy_multiplier 1.0 --agent.activation relu --agent.use_layer_norm 0 --agent.actor_lr 1e-3 --agent.critic_lr 1e-3 --batch_size 256 --agent.hidden_dims '(256,256)' --start_steps 10000"
    for env in online_envs:
        job_list.append(base + f"--env_name {env}")

# REDQ
for seed in range(4):
    base = f"python algs_online/sac.py --wandb.project {project} --wandb.group REDQ --seed {seed} "
    base += "--agent.num_q 5 --agent.utd 5"
    for env in online_envs:
        job_list.append(base + f"--env_name {env}")

# PPO
for seed in range(4):
    base = f"python algs_online/ppo.py --wandb.project {project} --wandb.group PPO --seed {seed} "
    for env in online_envs:
        job_list.append(base + f"--env_name {env}")

# TD3
for seed in range(4):
    base = f"python algs_online/td3.py --wandb.project {project} --wandb.group TD3 --seed {seed} "
    for env in online_envs:
        job_list.append(base + f"--env_name {env}")

# IQL
for seed in range(4):
    base = f"python algs_offline/iql.py --wandb.project {project} --wandb.group IQL --seed {seed} "
    for env in offline_envs:
        base += f"--env_name {env}"
        if 'gc' in env:
            base += " --goal_conditioned 1"
        if 'exorl' in env:
            base += " --agent.actor_loss_type ddpg"
        job_list.append(base)

# BC
for seed in range(4):
    base = f"python algs_offline/bc.py --wandb.project {project} --wandb.group BC --seed {seed} "
    for env in offline_envs:
        base += f"--env_name {env}"
        if 'gc' in env:
            base += " --goal_conditioned 1"
        job_list.append(base)

print(job_list)
# Launch these with your job runner of choice