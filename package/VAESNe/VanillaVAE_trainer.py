import torch
import numpy as np
from torch import nn, optim
from .losses import VAEloss, elbo
#from .Metrics import *
import matplotlib.pyplot as plt
import math
import gc


def safelog10(x):
    tmp = max(1e-10, x)
    return math.log10(tmp)



def training_step2(network, optimizer, data_loader, 
                  loss_fn = elbo, K=1,release_memory = False):
    """Train the model for one epoch

    Args:
        network: (nn.Module) the neural network
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data
        loss_fn: (function) loss function to use

    Returns:
        average of all loss values, accuracy, nmi
    """
    network.train()
    total_loss = 0.
    num_batches = 0.
    device = next(network.parameters()).device
    for (flux, time, band, mask) in data_loader: # flux, time, band, mask for photometry and flux, wavelength, phase, mask for spectra
        optimizer.zero_grad()
        flux = flux.to(device)
        time = time.to(device)
        band = band.to(device)
        mask = mask.to(device)
        x = (flux, time, band, mask)
        loss = -loss_fn(network, x)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        num_batches += 1.
        if release_memory:
            del x, flux, mask,time, band
            gc.collect()
            torch.cuda.empty_cache()

    return [total_loss / num_batches]