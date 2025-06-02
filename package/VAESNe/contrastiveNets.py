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

from .util_layers import singlelayerMLP
from .SpectraLayers import spectraTransformerEncoder
from .PhotometricLayers import photometricTransformerEncoder




class ContraPhotSpec(nn.Module):
    '''
    contrastive photometric and spectra pretraining
    '''
    def __init__(
        self, 
        # latent things
        latent_len,
        latent_dim,
        proj_dim,
        # photometric things
        num_bands,
        photo_model_dim,
        photo_num_heads, 
        photo_ff_dim, 
        photo_num_layers,
        photo_dropout,

        # spectra things
        spec_model_dim, 
        spec_num_heads, 
        spec_num_layers,
        spec_ff_dim, 
        spec_dropout,
        selfattn
    ):

        super(ContraPhotSpec, self).__init__()
        self.photometry_encoder = photometricTransformerEncoder(
                                 num_bands, 
                                 latent_len,
                                 latent_dim,
                                 photo_model_dim, 
                                 photo_num_heads, 
                                 photo_ff_dim, 
                                 photo_num_layers,
                                 photo_dropout,
                                 selfattn
                                 )
        self.photo_proj = singlelayerMLP(latent_len * latent_dim, proj_dim)

        self.spectra_encoder = spectraTransformerEncoder(
                latent_len,
                 latent_dim,
                 spec_model_dim, 
                 spec_num_heads, 
                 spec_num_layers,
                 spec_ff_dim, 
                 spec_dropout,
                 selfattn
                 )

        self.spectra_proj = singlelayerMLP(latent_len * latent_dim, proj_dim)

        self.latent_dim = latent_dim
        self.latent_len = latent_len
        self.proj_dim = proj_dim


    def forward(self, x):
        photo_flux, time, band, photo_mask = x[0]
        spec_flux, wavelength, phase, spec_mask = x[1]

        z1 = self.photometry_encoder(photo_flux, time, band, photo_mask)
        z2 = self.spectra_encoder(spec_flux, wavelength, phase, spec_mask)

        z1 = self.photo_proj(z1.reshape(z1.shape[0], -1))
        z2 = self.spectra_proj(z2.reshape(z2.shape[0], -1))

        return z1, z2

    def photo_enc(self, x):
        photo_flux, time, band, photo_mask = x
        self.eval()
        with torch.no_grad():
            return self.photometry_encoder(photo_flux, time, band, photo_mask)
    
    def spectra_enc(self,x):
        self.eval()
        spec_flux, wavelength, phase, spec_mask = x
        with torch.no_grad():
            return self.spectra_encoder(spec_flux, wavelength, phase, spec_mask)

