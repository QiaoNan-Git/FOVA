import argparse
import random

import gym
import d4rl

import numpy as np
import torch

import swanlab
import sys

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"   


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset, qlearning_dataset_checktraj
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import FedMFVPolicyTrainer
from offlinerlkit.policy import VCQLCSLPolicy
# from tianshou.data import Batch
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils import sample, collectTrajs, extract_and_combine_trajs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--algo-name", type=str, default="fova") # experiment name
    parser.add_argument("--task", type=str, default="hopper-medium-expert")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--lmbda", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000) # not used anymore
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu") # device = 'cuda'+":3"
    parser.add_argument("--local-num", type=int, default=5) # add local
    parser.add_argument("--local-step-per-epoch", type=int, default=100) # add local
    parser.add_argument("--local-data-size", type=int, default=10000)
    parser.add_argument("--from-full-dataset", type=bool, default=False)
    parser.add_argument("--num-traj", type=int, default=0)
    parser.add_argument("--swanlab", type=str, default="FOVA")
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--no-normalized", dest="no_normalized", action="store_false", default=True)
    

    return parser.parse_args()



def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    # dataset = qlearning_dataset(env)
    traj_length_list, episode_end_list, dataset = qlearning_dataset_checktraj(env, dataset=None, terminate_on_end=False, num_traj=args.num_traj)


    # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    if 'antmaze' in args.task:
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 4.0
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.traj_seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    
    # create policy



    # Normalization
    # create buffer
    dataset_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    dataset_buffer.load_dataset(dataset)
    # Compute mean and std of original data
    obs_mean, obs_std = dataset_buffer.normalize_obs()
    # If normalization is disabled, force mean=0, std=1
    if not args.no_normalized:  # Note: using `not` to invert logic
        obs_mean, obs_std = np.zeros_like(obs_mean), np.ones_like(obs_std)
    # Create scaler
    scaler = StandardScaler(mu=obs_mean, std=obs_std)
    print(f"Normalization: {'Yes' if args.no_normalized else 'No'} "
      f"(obs_mean={obs_mean.item(0):.2f}, "
      f"obs_std={obs_std.item(0):.2f})")


    # create policy list
    policies = []
    for i in range(args.local_num):
        # create policy model
        actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
        critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )

        local_num=args.local_num
        actor = ActorProb(actor_backbone, dist, args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1 = Critic(critic1_backbone, args.device)
        critic2 = Critic(critic2_backbone, args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        if args.auto_alpha:
            target_entropy = args.target_entropy if args.target_entropy \
                else -np.prod(env.action_space.shape)

            args.target_entropy = target_entropy

            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha

        policy = VCQLCSLPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=env.action_space,
            tau=args.tau,
            gamma=args.gamma,
            alpha=alpha,
            cql_weight=args.cql_weight,
            temperature=args.temperature,
            max_q_backup=args.max_q_backup,
            deterministic_backup=args.deterministic_backup,
            with_lagrange=args.with_lagrange,
            lagrange_threshold=args.lagrange_threshold,
            cql_alpha_lr=args.cql_alpha_lr,
            num_repeart_actions=args.num_repeat_actions,
            lmbda=args.lmbda,
            beta=args.beta,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            max_action=args.max_action,
            scaler=scaler,
        )
        policies.append(policy)



    # create buffer
    buffers = []
    episode_start_list = [end - length + 1 for end, length in zip(episode_end_list, traj_length_list)]
    for i in range(args.local_num):
        collected_trajs, total_length = collectTrajs(episode_start_list, episode_end_list, args.local_data_size)
               
        local_dataset = extract_and_combine_trajs(dataset, collected_trajs)
        buffer = ReplayBuffer(
                buffer_size=len(local_dataset["observations"]),
                obs_shape=args.obs_shape,
                obs_dtype=np.float32,
                action_dim=args.action_dim,
                action_dtype=np.float32,
                device=args.device
            )
        buffer.load_dataset_withNextActions_0(local_dataset)
        buffer.buffer_normalize_obs(obs_mean=obs_mean, obs_std=obs_std)   # buffer normalization
        buffers.append(buffer)

    # log

    log_dirs = make_log_dirs(args.task, f"{args.algo_name}-lspe{args.local_step_per_epoch}-cqlw_{args.cql_weight}", args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainers  = FedMFVPolicyTrainer(
            policies=policies,
            eval_env=env,
            buffers=buffers,
            logger=logger,
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            eval_episodes=args.eval_episodes,
            local_num=args.local_num,
            local_step_per_epoch=args.local_step_per_epoch
        )    
    
    config = {
        "algo_name": args.algo_name,
        "task": args.task,
        "local_num": args.local_num,
        "local_data_size": args.local_data_size,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "seed": args.seed,
        "cql_weight": args.cql_weight,
    }
    experiment_name= f"FORL-algo_{args.algo_name}-envs_{args.task}-ln_{args.local_num}-lds_{args.local_data_size}-lmbda_{args.lmbda}-beta_{args.beta}-s_{args.seed}-cqlw_{args.cql_weight}"
    swanlab.init(project=args.swanlab, entity="", experiment_name=experiment_name, config=config)

    #train
    policy_trainers.train()
    




if __name__ == "__main__":
    train()
