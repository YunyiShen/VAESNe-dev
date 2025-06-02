import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .PhotometricLayers import photometricTransformerEncoder
from .util_layers import MLP
from .SpectraLayers import spectraTransformerEncoder

class VAEregressionHead(nn.Module):
    def __init__(self, vae, 
                outdim, 
                freeze_vae = True,
                MLPlatent = [64, 64]
                ):

        super(VAEregressionHead, self).__init__()
        if freeze_vae:
            for param in vae.parameters():
                param.requires_grad = False
        self.vae = vae
        self.outfc = MLP(self.vae.latent_len * self.vae.latent_dim, outdim, MLPlatent)
    
    def forward(self, x):
        h = self.vae.encode(x, True)
        h = h.view(h.shape[0], -1) # flatten the latent
        return self.outfc(h)

class contrasphotoregressionHead(nn.Module):
    def __init__(self, contrastnet, 
                outdim, 
                freeze_contrastnet = True,
                MLPlatent = [64, 64]
                ):

        super(contrasphotoregressionHead, self).__init__()
        if freeze_contrastnet:
            for param in contrastnet.parameters():
                param.requires_grad = False
        self.contrastnet = contrastnet
        self.outfc = MLP(self.contrastnet.latent_len * self.contrastnet.latent_dim, outdim, MLPlatent)
    
    def forward(self, x):
        h = self.contrastnet.photo_enc(x)
        h = h.view(h.shape[0], -1) # flatten the latent
        return self.outfc(h)


class contrasspecregressionHead(nn.Module):
    def __init__(self, contrastnet, 
                outdim, 
                freeze_contrastnet = True,
                MLPlatent = [64, 64]
                ):

        super(contrasspecregressionHead, self).__init__()
        if freeze_contrastnet:
            for param in contrastnet.parameters():
                param.requires_grad = False
        self.contrastnet = contrastnet
        self.outfc = MLP(self.contrastnet.latent_len * self.contrastnet.latent_dim, outdim, MLPlatent)
    
    def forward(self, x):
        h = self.contrastnet.spectra_enc(x)
        h = h.view(h.shape[0], -1) # flatten the latent
        return self.outfc(h)



class photoend2endregression(nn.Module):
    def __init__(self, 
                outdim,
                num_bands = 6, 
                 latent_len = 4,
                 latent_dim = 4,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32,
                 num_layers = 4,
                 dropout=0.1,
                 selfattn=False,
                 MLPlatent = [64, 64]
                 
    ):
        super().__init__()
        self.enc = photometricTransformerEncoder(
                                 num_bands, 
                                 latent_len,
                                 latent_dim,
                                 model_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 dropout,
                                 selfattn
                                 )
        self.outfc = MLP(latent_dim * latent_len, outdim, MLPlatent)
        self.latent_dim = latent_dim
        self.latent_len = latent_len
    
    def forward(self, x):
        flux, time, band, mask = x
        h = self.enc(flux, time, band, mask)
        h = h.view(h.shape[0], -1) # flatten the latent
        return self.outfc(h)



class specend2endregression(nn.Module):
    def __init__(self, 
                outdim, 
                 latent_len = 4,
                 latent_dim = 4,
                 model_dim = 32, 
                num_heads = 4, 
                num_layers = 4,
                ff_dim = 32, 
                dropout=0.1,
                selfattn = False,
                 MLPlatent = [64, 64]
                 
    ):
        super().__init__()
        self.enc = spectraTransformerEncoder(
                                 latent_len,
                                latent_dim,
                                model_dim, 
                                num_heads, 
                                num_layers,
                                ff_dim, 
                                dropout,
                                selfattn
                                 )
        self.outfc = MLP(latent_dim * latent_len, outdim, MLPlatent)
        self.latent_dim = latent_dim
        self.latent_len = latent_len
    
    def forward(self, x):
        flux, wavelength, phase, mask = x
        h = self.enc(flux, wavelength, phase, mask)
        h = h.view(h.shape[0], -1) # flatten the latent
        return self.outfc(h)

