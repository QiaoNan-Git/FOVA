import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
import sys

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Optional, List, Callable
from offlinerlkit.policy import SACPolicy
from offlinerlkit.utils.noise import GaussianNoise

from typing import Dict, Union, Tuple, Optional, List

from offlinerlkit.utils.scaler import StandardScaler
LAMBDA = 1.0#0.6

BETA=1.0

class VCQLCSLPolicy(SACPolicy):
    """
    Conservative Q-Learning <Ref: https://arxiv.org/abs/2006.04779>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        lmbda: float = 1.0,
        beta: float = 1.0,
        exploration_noise: Optional[Callable] = GaussianNoise,
        max_action: Optional[float] = 1.0,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.action_space = action_space
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        self._is_auto_alpha = True
        self.actor_global =  deepcopy(actor)
        self.actor_last =  deepcopy(actor)
        self.critic1_last =  deepcopy(critic1)
        self.critic2_last =  deepcopy(critic2)
        self.max_exp_scale = 1.0
        
        self.lmbda = lmbda
        self.beta = beta

        self.exploration_noise = exploration_noise if exploration_noise is not None else None
        self.max_action = max_action if max_action is not None else 1.0
        self.scaler = scaler if scaler is not None else None

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        action = super().select_action(obs, deterministic)
        # with torch.no_grad():
        #     # action = self.actor(obs).cpu().numpy()
        #     action, _ = self.actforward(obs)
        #     # print("yes")
        # if not deterministic:
        #     action = action + self.exploration_noise(action.shape)
        #     action = np.clip(action, -self.max_action, self.max_action) 
        return action

    def get_actor_global_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor_global(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
    
    # def calc_pi_values(
    #     self,
    #     obs_pi: torch.Tensor,
    #     obs_to_pred: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     act, log_prob = self.actforward(obs_pi)

    #     q1 = self.critic1(obs_to_pred, act)
    #     q2 = self.critic2(obs_to_pred, act)

    #     return q1 - log_prob.detach(), q2 - log_prob.detach()
    
    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        local_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi) # pi
        local_action, local_log_prob = local_action, torch.zeros_like(log_prob) # pi_beta
        global_next_actions, global_next_log_probs = self.get_actor_global_actions(obs_pi) # pi_server
        
        pi_value_q1 = self.critic1(obs_to_pred, act) - log_prob.detach()
        pi_value_q2 = self.critic2(obs_to_pred, act) - log_prob.detach()
        
        local_pi_value_q1 = self.critic1(obs_to_pred, local_action) - local_log_prob.detach()
        local_pi_value_q2 = self.critic2(obs_to_pred, local_action) - local_log_prob.detach()
        
        global_pi_value_q1 = self.critic1(obs_to_pred, global_next_actions) - global_next_log_probs.detach()
        global_pi_value_q2 = self.critic2(obs_to_pred, global_next_actions) - global_next_log_probs.detach()
        
        vote_pi_valie_q1_stack = torch.stack([pi_value_q1, local_pi_value_q1, global_pi_value_q1])
        vote_pi_valie_q2_stack = torch.stack([pi_value_q2, local_pi_value_q2, global_pi_value_q2])
        
        vote_pi_valie_q1, _ = torch.max(vote_pi_valie_q1_stack, dim=0)
        vote_pi_valie_q2, _ = torch.max(vote_pi_valie_q2_stack, dim=0)

        return vote_pi_valie_q1, vote_pi_valie_q2

    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]
        self.actions, self.next_actions = actions, batch["next_actions"]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        # print(obss)
        
        dist = self.actor(obss)
        local_log_probs_kl = dist.log_prob(actions)
        
        global_actions, _ = self.get_actor_global_actions(obss)
        # global_log_probs_kl = dist.log_prob(global_actions)
        
        q1_pi, q2_pi = self.critic1_old(obss, a), self.critic2_old(obss, a)  # \pi  
        q1_local, q2_local = self.critic1_old(obss, actions), self.critic2_old(obss, actions)  # \beta\pi
        q1_gloabl, q2_gloabl = self.critic1_old(obss, global_actions), self.critic2_old(obss, global_actions)
        q1_piv_stack = torch.stack([q1_pi, q1_local, q1_gloabl]).to(self.actor.device)
        q1_piv, _ = torch.max(q1_piv_stack, dim=0)
        q2_piv_stack = torch.stack([q2_pi, q2_local, q2_gloabl]).to(self.actor.device)
        q2_piv, _ = torch.max(q2_piv_stack, dim=0)
        v = torch.min(q1_piv, q2_piv)
        
        # adv_local = torch.min(q1_local, q2_local) - v
        adv_global = torch.min(q1_gloabl, q2_gloabl) - v
        # adv_pi = torch.min(q1_pi, q2_pi) - v
        
        # exp_adv_local = torch.clip(torch.exp(adv_local), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        # exp_adv_pi = torch.clip(torch.exp(adv_pi), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        
        # no clip
        exp_adv_global = torch.clip(torch.exp((1.0/self.beta)*adv_global), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        #clip
        # exp_adv_global = torch.exp(torch.clip(adv_global, -self.max_exp_scale, self.max_exp_scale)).to(self.actor.device)

        # print(exp_adv_global.shape)
        # print((exp_adv_global * local_log_probs_kl).shape)
        # sys.exit()

        a_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean() 
        # b_loss = - LAMBDA * (exp_adv_pi * local_log_probs_kl).mean()  #pl
        b_loss = - self.lmbda * (exp_adv_global * local_log_probs_kl).mean()  #sl
        # b_loss = - self.lmbda * (local_log_probs_kl).mean()  # local kl
        # c_loss = - self.lmbda * (local_log_probs_kl).mean()  # local kl
        # actor_loss = b_loss
        actor_loss = a_loss + b_loss
        # actor_loss = a_loss + b_loss + c_loss
        # b_loss = - LAMBDA * (exp_adv_local * global_log_probs_kl).mean()  #ls
        # b_loss = - LAMBDA * (exp_adv_local * local_log_probs_kl).mean() 
         
        # b_loss = - LAMBDA * (local_log_probs_kl).mean()  # local kl
        # c_loss = - self.lmbda * (local_log_probs_kl).mean()  # localkld+d-d>0 
        c_loss = b_loss
        # c_loss = - (1 - LAMBDA) * (local_log_probs_kl).mean()  # local kl
        # c_loss = - (1 - LAMBDA) * (global_log_probs_kl).mean()  # global kl
        # actor_loss = a_loss + b_loss + c_loss
        # actor_loss = b_loss
        # actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()  # min

        # sys.exit()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)  # min
                # next_q = torch.mean(torch.cat([tmp_next_q1, tmp_next_q2], dim=0))  # mean
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                global_next_actions, global_next_log_probs = self.get_actor_global_actions(next_obss)
                
                next_q1_piv_stack = torch.stack([
                    self.critic1_old(next_obss, next_actions),  # \pi
                    self.critic1_old(next_obss, self.next_actions), # \beta\pi
                    self.critic1_old(next_obss, global_next_actions) # \bar\pi
                ])
                next_q1_piv, q1_max_indices = torch.max(next_q1_piv_stack, dim=0)
                # q1_num_selected = [(q1_max_indices == i).sum().item() for i in range(3)]
                
                # sys.exit()
                
                
                next_q2_piv_stack = torch.stack([
                    self.critic2_old(next_obss, next_actions),
                    self.critic2_old(next_obss, self.next_actions),
                    self.critic2_old(next_obss, global_next_actions)
                ])
                next_q2_piv, q2_max_indices = torch.max(next_q2_piv_stack, dim=0)
                # q2_num_selected = [(q2_max_indices == i).sum().item() for i in range(3)]
                # print(q1_num_selected, q2_num_selected)
                
                
                
                next_q = torch.min(
                    next_q1_piv,
                    next_q2_piv)
                # next_q = torch.min(
                #     self.critic1_old(next_obss, next_actions),
                #     self.critic2_old(next_obss, next_actions)
                # ) 


                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        q1_loss, q2_loss = critic1_loss, critic2_loss


        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            
        tmp_actions = self.actions.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, self.actions.shape[-1])
        tmp_next_actions = self.next_actions.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, self.next_actions.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss, tmp_actions)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss, tmp_next_actions)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        
        # print(q1.size())
        # print(cat_q1.size())
        # print(torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature)
        # print(q1.mean() * self._cql_weight) 

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
            
        # conservative_loss1 = \
        #     torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
        # conservative_loss2 = \
        #     torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2
        

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "loss/aloss": a_loss.item(),
            "loss/bloss": b_loss.item(),
            "loss/closs": c_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        else:
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result
    


    def update_actor(self, batch: Dict, pmoe_policy: Optional[float] = None, policies: Optional[List] = None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]
        self.actions, self.next_actions = actions, batch["next_actions"]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        # print(obss)
        
        dist = self.actor(obss)
        local_log_probs_kl = dist.log_prob(actions)
        
        global_actions, _ = self.get_actor_global_actions(obss)
        global_log_probs_kl = dist.log_prob(global_actions)
        
        q1_pi, q2_pi = self.critic1_old(obss, a), self.critic2_old(obss, a)  # \pi  
        q1_local, q2_local = self.critic1_old(obss, actions), self.critic2_old(obss, actions)  # \beta\pi
        q1_gloabl, q2_gloabl = self.critic1_old(obss, global_actions), self.critic2_old(obss, global_actions)
        q1_piv_stack = torch.stack([q1_pi, q1_local, q1_gloabl]).to(self.actor.device)
        q1_piv, _ = torch.max(q1_piv_stack, dim=0)
        q2_piv_stack = torch.stack([q2_pi, q2_local, q2_gloabl]).to(self.actor.device)
        q2_piv, _ = torch.max(q2_piv_stack, dim=0)
        v = torch.min(q1_piv, q2_piv)
        
        # adv_local = torch.min(q1_local, q2_local) - v
        adv_global = torch.min(q1_gloabl, q2_gloabl) - v
        # adv_pi = torch.min(q1_pi, q2_pi) - v
        
        # exp_adv_local = torch.clip(torch.exp(adv_local), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        # exp_adv_pi = torch.clip(torch.exp(adv_pi), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        
        # no clip
        exp_adv_global = torch.clip(torch.exp((1.0/self.beta)*adv_global), -self.max_exp_scale, self.max_exp_scale).to(self.actor.device)
        #clip
        # exp_adv_global = torch.exp(torch.clip(adv_global, -self.max_exp_scale, self.max_exp_scale)).to(self.actor.device)

        # print(exp_adv_global.shape)
        # print((exp_adv_global * local_log_probs_kl).shape)
        # sys.exit()

        a_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean() 
        # b_loss = - LAMBDA * (exp_adv_pi * local_log_probs_kl).mean()  #pl
        b_loss = - self.lmbda * (exp_adv_global * local_log_probs_kl).mean()  #sl
        # c_loss = - self.lmbda * (local_log_probs_kl).mean()  # local kl
        # actor_loss = b_loss
        actor_loss = a_loss + b_loss
        # actor_loss = a_loss + b_loss + c_loss
        # b_loss = - LAMBDA * (exp_adv_local * global_log_probs_kl).mean()  #ls
        # b_loss = - LAMBDA * (exp_adv_local * local_log_probs_kl).mean() 
         
        # b_loss = - LAMBDA * (local_log_probs_kl).mean()  # local kl
        c_loss = - self.lmbda * (local_log_probs_kl).mean()  # localkld+d-d>0 
        # c_loss = - (1 - LAMBDA) * (local_log_probs_kl).mean()  # local kl
        # c_loss = - (1 - LAMBDA) * (global_log_probs_kl).mean()  # global kl
        # actor_loss = a_loss + b_loss + c_loss
        # actor_loss = b_loss
        # actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()  # min

        # sys.exit()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        return actor_loss, {
            "loss/actor": actor_loss.item(),
            "loss/aloss": a_loss.item(),
            "loss/bloss": b_loss.item(),
            "loss/closs": c_loss.item(),
            "alpha": self._alpha.item(),
        }

    

    def update_critic(self, batch: Dict, pmoe_policy: Optional[float] = None, policies: Optional[List] = None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]
        self.actions, self.next_actions = actions, batch["next_actions"]

        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)  # min
                # next_q = torch.mean(torch.cat([tmp_next_q1, tmp_next_q2], dim=0))  # mean
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                global_next_actions, global_next_log_probs = self.get_actor_global_actions(next_obss)
                
                next_q1_piv_stack = torch.stack([
                    self.critic1_old(next_obss, next_actions),  # \pi
                    self.critic1_old(next_obss, self.next_actions), # \beta\pi
                    self.critic1_old(next_obss, global_next_actions) # \bar\pi
                ])
                next_q1_piv, q1_max_indices = torch.max(next_q1_piv_stack, dim=0)
                # q1_num_selected = [(q1_max_indices == i).sum().item() for i in range(3)]
                
                # sys.exit()
                
                
                next_q2_piv_stack = torch.stack([
                    self.critic2_old(next_obss, next_actions),
                    self.critic2_old(next_obss, self.next_actions),
                    self.critic2_old(next_obss, global_next_actions)
                ])
                next_q2_piv, q2_max_indices = torch.max(next_q2_piv_stack, dim=0)
                # q2_num_selected = [(q2_max_indices == i).sum().item() for i in range(3)]
                # print(q1_num_selected, q2_num_selected)
                
                
                
                next_q = torch.min(
                    next_q1_piv,
                    next_q2_piv)
                # next_q = torch.min(
                #     self.critic1_old(next_obss, next_actions),
                #     self.critic2_old(next_obss, next_actions)
                # ) 


                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        q1_loss, q2_loss = critic1_loss, critic2_loss


        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            
        tmp_actions = self.actions.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, self.actions.shape[-1])
        tmp_next_actions = self.next_actions.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, self.next_actions.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss, tmp_actions)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss, tmp_next_actions)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        
        # print(q1.size())
        # print(cat_q1.size())
        # print(torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature)
        # print(q1.mean() * self._cql_weight) 

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
            
        # conservative_loss1 = \
        #     torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
        # conservative_loss2 = \
        #     torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2
        

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        return critic1_loss, critic2_loss, {
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "cql_alpha": cql_alpha.item() if self._with_lagrange else 0.0,
        }

