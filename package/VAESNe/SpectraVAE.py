import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .SpectraLayers import spectraTransformerDecoder, spectraTransformerEncoder

from .base_vae import VAE
import torch.distributions as dist

class SpectraEnc(nn.Module):
    def __init__(self, 
                 latent_length,
                 latent_dim,
                 model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout=0.1):
        super(SpectraEnc, self).__init__()

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = spectraTransformerEncoder(
                2 * latent_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, dropout)
        self.latent_dim = latent_dim
        self.latent_length = latent_length


    def forward(self, flux, wavelength, phase, mask = None):
        bottleneck = self.inference_transformer(flux, 
                                                wavelength, 
                                                phase,
                                                mask)

        
        # q(z|x,y)
        mu = bottleneck[:,:self.latent_length,:]
        var = F.softplus( bottleneck[:,self.latent_length:,:])
        
        return mu, var

class SpectraDec(nn.Module):
    def __init__(self, spectra_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout=0.1):
        super(vanillaSpectraGenerativeNet, self).__init__()

        # p(x|z)
        self.generativetransformer = spectraTransformerDecoder(
                spectra_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout)

    
    # p(x|z)
    def pxz(self, wavelength, phase, z, mask = None):
        flux = self.generativetransformer(wavelength, phase, z, mask)
        return flux

    def forward(self, wavelength, phase, z, mask = None):
        x_rec = self.pxz(wavelength, phase, z, mask)
        var = torch.ones_like(x_rec) 
        if mask is not None:
            var += 1e10 * mask # if masked variance is large 
        return x_rec, var


class SpectraVAE(VAE):
    def __init__(self, spectra_length = 982,
                latent_length = 4,
                latent_dim = 2,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout = 0.1):
        super(SpectraVAE, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,
            SpectraEnc(latent_length,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout),
            SpectraDec(spectra_length,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout),
            params = [
                    spectra_length,
                    latent_length,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout]
        )
        self.llik_scaling = 1.
        self.modelName = 'spectrum'
    
    def forward(self, x, K = 1):
        flux, wavelength, phase, mask = x
        self._qz_x_params = self.enc(flux, wavelength, phase, mask)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(wavelength, phase, zs, mask))
        return qz_x, px_z, zs
    
    def reconstruct(self, x):
        flux, wavelength, phase, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, wavelength, phase, mask))
            zs = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(wavelength, phase, zs, mask))
            recon = get_mean(px_z)
        return recon
    
    def generate(self, N, wavelength, phase, mask = None):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            zs = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(wavelength, phase, zs, mask))
            data = px_z.mean
        return data