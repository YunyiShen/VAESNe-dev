from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .util_layers import kl_divergence, get_mean, log_mean_exp

"""
mmvae loss
Mixture of Experts for multimodal variational autoencoders
"""

def expand_first_dim(t, K):
    shape = t.shape
    return t.unsqueeze(0).expand((K,) + shape)

def elbo(model, x, K=1):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x, K)
    data = expand_first_dim(x[0], K)
    lpx_z = px_z.log_prob(data).reshape(*px_z.batch_shape[:2], -1) * model.llik_scaling # take data
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - kld.sum((-1,-2))[None,:]).mean()#.mean(0).sum()


def m_elbo(model, x, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum([-1,-2]))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d][0]).view(*px_zs[d][d].batch_shape[:2], -1)  # added x[d][0], assume that x take the form of [(flux, time, band, mask), (flux, wavelength, phase, mask)], sum over data dimensions, kept batch and sample size
            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum((-1. -2)) # sum over two latent dimensions
            lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum([-1,-2]) # -1 -2 for two latent dimensions
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum([-1,-2]) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d][0]).view(*px_z.batch_shape[:2], -1) # added x[d][0], assume that x take the form of [(flux, time, band, mask), (flux, wavelength, phase, mask)], , sum over data dimensions, kept batch and sample size
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        
        lpx_z = torch.stack(lpx_z).sum(0) # sum over batch K
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def is_multidata(dataB):
    return isinstance(dataB, list)

def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0][0].size(0) if is_multidata(x) else x[0].size(0)
    S = sum([1.0 / (K * np.prod(_x[0].size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * np.prod(x[0].size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    #breakpoint()
    return min(B, S)

def m_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    n_chunk = len(x[0][0].split(S)) # not the best way to do it
    #x_split = zip(*[(_x.split(S) for _x in __x) for __x in x]) # LC and spectra all saved in tuples
    #lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = []
    
    for i in range(n_chunk):
        #breakpoint()
        split_i = tuple(tuple(tensor.split(S)[i]  for tensor in tensor_tuple) for tensor_tuple in x)
        #breakpoint()
        lw.append(_m_iwae(model, split_i, K))
    
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


### contrastive loss ###

def negInfoNCE(model, x, temperature = 0.07):
    '''
    assue model.forward will give two encodes, with projection
    '''
    z1, z2 = model(x)
    z1 = F.normalize(z1, dim = -1)
    z2 = F.normalize(z2, dim = -1)

    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    #breakpoint()

    return -(F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


