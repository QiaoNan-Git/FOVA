from copy import deepcopy
import time
import os
import copy
import numpy as np
import torch
import gym
import swanlab
import sys

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

# WANDB = False
LOG = 3
WANDB = True
AGGRE = True
# Model-free policy trainer
class FedMFVPolicyTrainer:
    def __init__(
        self,
        policies: List[BasePolicy],
        eval_env: gym.Env,
        buffers: List[ReplayBuffer],
        logger: Logger,
        epoch: int,
        step_per_epoch: int,
        batch_size: int,
        eval_episodes: int,
        local_num: int,
        local_step_per_epoch: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.policies = policies
        self.eval_env = eval_env
        self.buffers = buffers
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self._local_num = local_num
        self._local_step_per_epoch = local_step_per_epoch
        self.lr_scheduler = lr_scheduler
        self.weights = None
        self.gamma = np.array([0.99 ** n for n in range(2000)])
        

    def train(self, buffers_new: Optional[List[ReplayBuffer]] = None) -> Dict[str, float]:
        #init parameter
        start_time = time.time()
        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        ep_reward_mean_local, ep_reward_std_local, ep_length_mean_local, ep_length_std_local, norm_ep_rew_mean_local, norm_ep_rew_std_local = [[None] * self._local_num for _ in range(6)]
        policy_local, actor_local, critic1_local, critic2_local = [[None] * self._local_num for _ in range(4)]
        true_reward, esti_value = [[None] * self._local_num for _ in range(2)]
        record_loss_list = [{} for _ in self.policies]     
        server_policy = deepcopy(self.policies[0])

        for epoch in range(1, self._epoch + 1):

            start_epoch_time = time.time()
            for policy_idx, policy in enumerate(self.policies):

                policy.train()

                for it in range(self._local_step_per_epoch):
                    batch = self.buffers[policy_idx].sample_withNextActions(self._batch_size)
                    loss = policy.learn(batch)
        
                    for k, v in loss.items():
                        self.logger.logkv_mean(k, v)
                record_loss_list[policy_idx] = loss
        
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
                # evaluate current policy
                eval_info = self._evaluate(policy)
                ep_reward_mean_local[policy_idx], ep_reward_std_local[policy_idx] = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean_local[policy_idx], ep_length_std_local[policy_idx] = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                norm_ep_rew_mean_local[policy_idx] = self.eval_env.get_normalized_score(ep_reward_mean_local[policy_idx]) * 100
                norm_ep_rew_std_local[policy_idx] = self.eval_env.get_normalized_score(ep_reward_std_local[policy_idx]) * 100

                # Server-side aggregation of actor and critic
                critic1_local[policy_idx] = policy.critic1.state_dict()
                critic2_local[policy_idx] = policy.critic2.state_dict()
                actor_local[policy_idx] = policy.actor.state_dict()
            
            end_epoch_time = time.time()
            num_timesteps += 1
            ep_length_mean_global = np.mean(ep_length_mean_local)
            ep_length_std_global = np.mean(ep_length_std_local)
            ep_reward_mean_global = np.mean(ep_reward_mean_local)
            ep_reward_std_global = np.mean(ep_reward_std_local)
            norm_ep_rew_mean_global = np.mean(norm_ep_rew_mean_local)
            norm_ep_rew_std_global = np.mean(norm_ep_rew_std_local)



            
            # Upload and aggregation
            if self.weights is not None:
                weights = self.weights
            else:
                N = len(self.policies)
                weights = [1 / N] * N

            for policy_idx, policy in enumerate(self.policies):
                weight = weights[policy_idx]
                for main_param, agent_param in zip(
                    server_policy.critic1.parameters(), policy.critic1.parameters()
                ):
                    if policy_idx == 0:
                        main_param.data.copy_(agent_param * weight)
                    else:
                        main_param.data.copy_(main_param + agent_param * weight)

                for main_param, agent_param in zip(
                    server_policy.critic2.parameters(), policy.critic2.parameters()
                ):
                    if policy_idx == 0:
                        main_param.data.copy_(agent_param * weight)
                    else:
                        main_param.data.copy_(main_param + agent_param * weight)

                for main_param, agent_param in zip(
                    server_policy.actor.parameters(), policy.actor.parameters()
                ):
                    if policy_idx == 0:
                        main_param.data.copy_(agent_param * weight)
                    else:
                        main_param.data.copy_(main_param + agent_param * weight)

                for main_param, agent_param in zip(
                    server_policy.actor_global.parameters(), policy.actor.parameters()
                ):
                    if policy_idx == 0:
                        main_param.data.copy_(agent_param * weight)
                    else:
                        main_param.data.copy_(main_param + agent_param * weight)

            # Distribute aggregated parameters to local policies
            for policy_idx, policy in enumerate(self.policies):
                if AGGRE:
                    for main_agent_param, agent_param in zip(
                        server_policy.critic1.parameters(), policy.critic1.parameters()
                    ):
                        agent_param.data.copy_(main_agent_param)
                    for main_agent_param, agent_param in zip(
                        server_policy.critic2.parameters(), policy.critic2.parameters()
                    ):
                        agent_param.data.copy_(main_agent_param)
                    for main_agent_param, agent_param in zip(
                        server_policy.actor.parameters(), policy.actor.parameters()
                    ):
                        agent_param.data.copy_(main_agent_param)
                for main_agent_param, agent_param in zip(
                    server_policy.actor_global.parameters(), policy.actor_global.parameters()
                ):
                    agent_param.data.copy_(main_agent_param)



                
            
            server_eval_info = self._evaluate(server_policy)
            server_ep_reward, server_ep_length = np.mean(server_eval_info["eval/episode_reward"]), np.mean(server_eval_info["eval/episode_length"])
            server_norm_ep_rew_mean = self.eval_env.get_normalized_score(server_ep_reward) * 100
            
            
            if WANDB is True:
                self.logger.logkv("eval/avg_normalized_episode_reward", norm_ep_rew_mean_global)
                self.logger.logkv("eval/server_normalized_episode_reward", server_norm_ep_rew_mean)
                self.logger.set_timestep(num_timesteps)
                self.logger.dumpkvs()
            
            if WANDB is True:
                second = time.time() - start_time  # Directly compute time difference (float)
                swanlab.log({
                    "global_eval/episode_reward": ep_reward_mean_global,
                    "global_eval/server_episode_reward": server_ep_reward,
                    "global_eval/normalized_episode_reward": norm_ep_rew_mean_global,
                    "global_eval/episode_length": ep_length_mean_global,
                    "global_eval/server_normalized_episode_reward": server_norm_ep_rew_mean,
                    "epoch": num_timesteps,
                    "second": second,
                    "step": epoch,
                })

                for policy_idx, policy in enumerate(self.policies):
                    swanlab.log({
                        f"local_eval/normalized_episode_reward_{policy_idx}": norm_ep_rew_mean_local[policy_idx],
                    })
                    
                    self.logger.logkv(f"eval/normalized_episode_reward_{policy_idx}", norm_ep_rew_mean_local[policy_idx])

                for (i, record_loss) in enumerate(record_loss_list):
                    record_loss_with_prefix = {f"local_loss_eval/num_{i}_{key}": value for key, value in record_loss.items()}
                    swanlab.log(record_loss_with_prefix)
        
        
        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}
        


    def _evaluate(self, policy) -> Dict[str, List[float]]:
        policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def V_evaluate(self, policy) -> Dict[str, List[float]]:
        policy.eval()
        obs = self.eval_env.reset()
        esti_value_info_buffer = []
        true_reward_info_buffer = []
        num_episodes = 0
        true_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = policy.select_action(obs.reshape(1, -1), deterministic=True)
            if num_episodes == 0:
                tensor_obs = torch.tensor([obs], dtype=torch.float32, requires_grad=True).to(policy.actor.device)  # Here we set requires_grad=True
                tensor_a = torch.tensor(action, dtype=torch.float32, requires_grad=True).to(policy.actor.device)  # Here we set requires_grad=True
                q1 = policy.critic1(tensor_obs, tensor_a)
                q2 = policy.critic2(tensor_obs, tensor_a)
                value = min(q1, q2).item()
                esti_value_info_buffer.append(
                    {"esti_value": value,}
                )
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            true_reward += self.gamma[episode_length] * reward
            episode_length += 1

            obs = next_obs

            if terminal:
                true_reward_info_buffer.append(
                    {"true_reward": true_reward, "episode_length": episode_length}
                )
                num_episodes += 1
                true_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        return {
            "eval/true_reward": [ep_info["true_reward"] for ep_info in true_reward_info_buffer],
            "eval/esti_value": [ep_info["esti_value"] for ep_info in esti_value_info_buffer],
        }
