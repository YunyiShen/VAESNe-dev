import torch
import numpy as np
from torch import nn, optim
from .losses import elbo
#from .Metrics import *
import matplotlib.pyplot as plt
import math
import gc


def safelog10(x):
    tmp = max(1e-10, x)
    return math.log10(tmp)



def training_step(network, optimizer, data_loader, 
                  loss_fn = elbo, K=1,
                  multimodal = False,
                  release_memory = False):
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
    for x in data_loader: # flux, time, band, mask for photometry and flux, wavelength, phase, mask for spectra
        optimizer.zero_grad()
        if multimodal:
            x = [tuple(_x.to(device) for _x in modality) for modality in x]
        else:
            x = tuple(_x.to(device) for _x in x)
        loss = -loss_fn(network, x)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        num_batches += 1.
        if release_memory:
            del x
            gc.collect()
            torch.cuda.empty_cache()

    return total_loss / num_batches