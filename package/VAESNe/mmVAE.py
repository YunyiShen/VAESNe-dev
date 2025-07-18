# mixture of expert VAEs from https://github.com/iffsid/mmvae/ https://arxiv.org/abs/1911.03393
from itertools import combinations
import torch.nn as nn
import math
import os
import sys
import time

import torch
import torch.distributions as dist
import torch.nn.functional as F
from numpy import prod
from .util_layers import get_mean, log_mean_exp, kl_divergence

### base class of mmVAE

class MMVAE(nn.Module):
    def __init__(self, vaes, prior_dist = dist.Laplace):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList(vaes)
        self.modelName = None  # filled-in per sub-class
        #self.params = params
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



class photospecMMVAE(nn.Module):
    def __init__(self, vaes, prior_dist = dist.Laplace, beta = 1., length_ratio = 982/60):
        super(photospecMMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList(vaes)
        self.modelName = "photospectra"
        #self.params = params
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(vaes[0].latent_len, vaes[0].latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(vaes[0].latent_len, vaes[0].latent_dim), requires_grad=False)  # logvar
        ])
        self.vaes[0].llik_scaling = 1./beta
        self.vaes[1].llik_scaling = 1./beta
        self.vaes[0].llik_scaling *= length_ratio

    @property
    def pz_params(self):
        return self._pz_params


    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            #breakpoint()
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.decode(zs, x[d]) 
        #breakpoint()
        return qz_xs, px_zs, zss

    def generate(self, N, x):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N, x[0][0].shape[0]])) # #samples, batch for conditioning
            for d, vae in enumerate(self.vaes):
                px_z = vae.decode(latents, x[d])
                #data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
                data.append(px_z.mean)
        return data  # list of generations---one for each modality

    def reconstruct(self, data, K=1):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data, K=K)
            # cross-modal matrix of reconstructions
            recons = [[px_z.mean for px_z in r] for r in px_zs]
        return recons
    
    def crossmodgen(self, x_in, x_out, direction = [0,1], K = 1):
        self.eval()
        with torch.no_grad():
            zs = self.vaes[direction[0]].encode(LC, mean = False).rsample(torch.Size([K]))
            return self.vaes[direction[1]].decode(zs, x_out).mean


