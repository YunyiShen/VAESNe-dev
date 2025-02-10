# mixture of expert VAEs from https://github.com/iffsid/mmvae/ https://arxiv.org/abs/1911.03393
from itertools import combinations

import torch
import torch.nn as nn
import math
import os
import shutil
import sys
import time

import torch
import torch.distributions as dist
import torch.nn.functional as F

import torch
from numpy import prod


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)
    



class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pz_params = None  # defined in subclass

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons


def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


