
import math
import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod

from torch import Tensor
from torch.distributions import MultivariateNormal


# def standard_normal_logprob(z):
#     """
#     z: [B, zdim]
#     """
#     dim = z.size(-1)
#     log_z = -0.5 * dim * math.log(2 * math.pi)
#     return log_z - z.pow(2) / 2
# 
# def log_normal_logprob(z, mu, var):
#      log_norm = torch.log(torch.norm(z, dim=2))
#      logz = -1.0 * math.log(2) - 1.5 * math.log(2 * math.pi) - 0.5 * math.log(var)
#      return logz - 3.0 * log_norm - (log_norm - mu).pow(2) / (2 * var)


# -----------------------------------------------------------------------------------------
class Distribution(nn.Module):

    @abstractmethod
    def logp(self, mu: Tensor, var: Tensor, z: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, mu: Tensor, var: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def standard_logp(self, z: Tensor):
        raise NotImplementedError()
    
    @abstractmethod
    def standard_sample(self, shape, device):
        raise NotImplementedError()


# -----------------------------------------------------------------------------------------
class GaussianDistribution(Distribution):
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, pc_channel: int, mu: float, vars: float, temperature: float=1.0, device='cuda:0'):
        super(GaussianDistribution, self).__init__()

        mu   = torch.ones(pc_channel) * mu
        vars = torch.eye(pc_channel) * vars
        self.prior = MultivariateNormal(mu.to(device), vars.to(device))

        assert temperature >= 0.0 and temperature <= 1.0
        self.temperature = temperature * temperature  # temperature annealing
        # self.standard_prior = MultivariateNormal(mu, vars * self.temperature)

    @staticmethod
    def likelihood(mean, logs, x):
        """
        x: [B, C]
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.)\
            + GaussianDistribution.Log2PI)

    @staticmethod
    def standard_likelood(x):
        return -0.5 * (x ** 2 + GaussianDistribution.Log2PI)

    def logp(self, mu: Tensor, var: Tensor, z: Tensor):
        likelihood = GaussianDistribution.likelihood(mu, var, z)
        sum_dims = tuple([i for i in range(1, likelihood.dim())])
        return torch.sum(likelihood, dim=sum_dims, keepdim=False)

    def sample(self, mu: Tensor, var: Tensor):
        mean, std = torch.zeros_like(mu), torch.ones_like(var)
        eps = torch.normal(mean=mean, std=(std * self.temperature))
        return mu + torch.exp(var) * eps

    def standard_logp(self, z: Tensor):
        # logp_z = self.prior.log_prob(z.cpu())

        logp_z = GaussianDistribution.standard_likelood(z)
        sum_dims = tuple([i for i in range(1, logp_z.dim())])
        logp_z = torch.sum(logp_z, dim=sum_dims) # [B,]
        return logp_z

    def standard_sample(self, shape, device, temperature=None):
        # return self.prior.sample(shape[:-1])
        # return self.standard_prior.sample(shape[:-1]).to(device)

        temp = temperature ** 2 if temperature is not None else self.temperature
        return torch.normal(mean=torch.zeros(shape), std=(torch.ones(shape) * temp)).to(device)


