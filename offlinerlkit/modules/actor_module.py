import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from typing import Union, Optional

from torch.distributions import Normal, TanhTransform, TransformedDistribution
EPS = 1e-7

# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
    
    def get_log_density(self, observations, action):
        logits = self.backbone(observations)
        mu = self.dist_net.mu(logits)
        if not self.dist_net._unbounded:
            mu = self.dist_net._max * torch.tanh(mu)
        if self.dist_net._c_sigma:
            sigma = torch.clamp(self.dist_net.sigma(logits), min=self.dist_net._sigma_min, max=self.dist_net._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.dist_net.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        action_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_prob = torch.sum(action_distribution.log_prob(action_clip), dim=-1)

        return logp_prob
        
        print(sigma)
        print(mu)
        sys.exit()
        dist = self.dist_net(logits)
        base_network_output = dist.mode()
        
        
        # print(dist)
        # print(dist.mode())
        # print(dist.mode().shape)
        # sys.exit()
        # print(base_network_output)
        print(self.dist_net.output_dim)
        print(base_network_output.shape)
        mean, log_std = torch.split(base_network_output, int(self.dist_net.output_dim), dim=-1)
        # mean, log_std = torch.split(base_network_output, getattr(dist_net, "output_dim"), dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.tanh_gaussian.log_std_min, self.tanh_gaussian.log_std_max)
        std = torch.exp(log_std)
        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_prob = torch.sum(action_distribution.log_prob(action_clip), dim=-1)
        
    # def log_prob(self, action, raw_action=None):
    #     if raw_action is None:
    #         raw_action = self.arctanh(action)
    #     log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
    #     eps = 1e-6
    #     log_prob = log_prob - torch.log((1 - action.pow(2)) + eps).sum(-1, keepdim=True)
    #     return log_prob


# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions