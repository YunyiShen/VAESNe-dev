import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .SpectraLayers import spectraTransformerDecoder, spectraTransformerEncoder

from .base_vae import VAE
import torch.distributions as dist
from .util_layers import MLP

class SpectraEnc(nn.Module):
    def __init__(self, 
                 latent_len,
                 latent_dim,
                 model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout=0.1,
                selfattn = False):
        super(SpectraEnc, self).__init__()

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = spectraTransformerEncoder(
                2 * latent_len,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout,
                 selfattn
                 )
        self.latent_dim = latent_dim
        self.latent_len = latent_len


    def forward(self, flux, wavelength, phase, mask = None):
        bottleneck = self.inference_transformer(flux, 
                                                wavelength, 
                                                phase,
                                                mask)

        
        # q(z|x,y)
        mu = bottleneck[:,:self.latent_len,:]
        var = F.softplus( bottleneck[:,self.latent_len:,:])
        
        return mu, var

class SpectraDec(nn.Module):
    def __init__(self, 
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout=0.1,
                 selfattn = False):
        super(SpectraDec, self).__init__()

        # p(x|z)
        self.generativetransformer = spectraTransformerDecoder(
                
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout,
                 selfattn
                 )

    
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
    def __init__(self, 
                latent_len = 4,
                latent_dim = 2,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout = 0.1,
                selfattn = False,
                beta = 1.,
                prior = dist.Laplace,
                likelihood = dist.Laplace,
                posterior = dist.Laplace):
        super(SpectraVAE, self).__init__(
            prior,  # prior
            likelihood,  # likelihood
            posterior,  # posterior
            SpectraEnc(latent_len,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout,
                    selfattn
                    ),
            SpectraDec(
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout),
            params = [
                    
                    latent_len,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout,
                    selfattn
                    ]
        )
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(latent_len, latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(latent_len, latent_dim), requires_grad=False)  # logvar
        ])
        self.llik_scaling = 1./beta
        self.modelName = 'spectrum'
        
        self.latent_len = latent_len
        self.latent_dim = latent_dim
    
    def forward(self, x, K = 1):
        flux, wavelength, phase, mask = x
        self._qz_x_params = self.enc(flux, wavelength, phase, mask)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        #breakpoint()
        '''
        px_z_loc, px_z_scale = self.dec(wavelength.unsqueeze(0).expand(K, -1, -1).view(-1, wavelength.shape[-1]), 
                                        phase.unsqueeze(0).expand(K, -1).view(-1), 
                                        zs.view(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).view(-1, mask.shape[-1]))
        px_z_loc = px_z_loc.view(K, -1, flux.shape[-1])
        px_z_scale = px_z_scale.view(K, -1, flux.shape[-1])
        px_z = self.px_z(px_z_loc, px_z_scale)
        '''
        px_z = self.decode(zs, x)
        #breakpoint()
        return qz_x, px_z, zs
    
    def reconstruct(self, x, K = 1):
        flux, wavelength, phase, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, wavelength, phase, mask))
            zs = qz_x.rsample([K])  # no dim expansion
            px_z = self.decode(zs, x)
            recon = px_z.mean
        return recon
    
    def encode(self, x, mean = True):
        flux, time, band, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, time, band, mask))
        if mean:
            return qz_x.mean
        return qz_x
    
    def decode(self, zs, x):
        _, wavelength, phase, mask = x 
        K = zs.shape[0]
        px_z_loc, px_z_scale = self.dec(wavelength.unsqueeze(0).expand(K, -1, -1).reshape(-1, wavelength.shape[-1]), 
                                        phase.unsqueeze(0).expand(K, -1).reshape(-1), 
                                        zs.reshape(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).reshape(-1, mask.shape[-1]))
        px_z_loc = px_z_loc.reshape(K, -1, wavelength.shape[1])
        px_z_scale = px_z_scale.reshape(K, -1, wavelength.shape[1])

        return self.px_z(px_z_loc, px_z_scale)
    
    def generate(self, N, x):
        self.eval()
        _, wavelength, phase, mask = x
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            zs = pz.rsample(torch.Size([N,1]))
            px_z = self.decode(zs, x)
            data = px_z.mean.unsqueeze(0)
        return data




class BrightSpectraVAE(VAE):
    def __init__(self, 
                latent_len = 4,
                latent_dim = 2,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout = 0.1,
                selfattn = False,
                beta = 1.,
                prior = dist.Laplace,
                likelihood = dist.Laplace,
                posterior = dist.Laplace):
        assert latent_len > 1, "Need at least one token for overall brightness"
        super(BrightSpectraVAE, self).__init__(
            prior,  # prior
            likelihood,  # likelihood
            posterior,  # posterior
            SpectraEnc(latent_len,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout,
                    selfattn
                    ),
            SpectraDec(
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout),
            params = [
                    
                    latent_len,
                    latent_dim,
                    model_dim, 
                    num_heads, 
                    num_layers,
                    ff_dim, 
                    dropout,
                    selfattn
                    ]
        )
        
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(latent_len, latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(latent_len, latent_dim), requires_grad=False)  # logvar
        ])
        self.llik_scaling = 1./beta
        self.modelName = 'spectrum'
        
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.brightnessfc = MLP(latent_dim + 1, 1, [model_dim]) # phase is added
    
    def forward(self, x, K = 1):
        flux, wavelength, phase, mask = x
        self._qz_x_params = self.enc(flux, wavelength, phase, mask)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        #breakpoint()
        '''
        px_z_loc, px_z_scale = self.dec(wavelength.unsqueeze(0).expand(K, -1, -1).view(-1, wavelength.shape[-1]), 
                                        phase.unsqueeze(0).expand(K, -1).view(-1), 
                                        zs.view(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).view(-1, mask.shape[-1]))
        px_z_loc = px_z_loc.view(K, -1, flux.shape[-1])
        px_z_scale = px_z_scale.view(K, -1, flux.shape[-1])
        px_z = self.px_z(px_z_loc, px_z_scale)
        '''
        px_z = self.decode(zs, x)
        #breakpoint()
        return qz_x, px_z, zs
    
    def reconstruct(self, x, K = 1):
        flux, wavelength, phase, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, wavelength, phase, mask))
            zs = qz_x.rsample([K])  # no dim expansion
            px_z = self.decode(zs, x)
            recon = px_z.mean
        return recon
    
    def encode(self, x, mean = True):
        flux, time, band, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, time, band, mask))
        if mean:
            return qz_x.mean
        return qz_x
    
    def decode(self, zs, x):
        _, wavelength, phase, mask = x 
        K = zs.shape[0]
        phase_expand = phase.unsqueeze(0).expand(K, -1)
        brightness = torch.concat((zs[:, :,0, :], phase_expand[:,:,None]), dim = -1)
        brightness = self.brightnessfc(brightness)
        px_z_loc, px_z_scale = self.dec(wavelength.unsqueeze(0).expand(K, -1, -1).reshape(-1, wavelength.shape[-1]), 
                                        phase_expand.reshape(-1), 
                                        zs.reshape(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).reshape(-1, mask.shape[-1]))
        px_z_loc = px_z_loc.reshape(K, -1, wavelength.shape[1])
        px_z_loc = px_z_loc + brightness - px_z_loc.mean(axis = 2)[:, :, None] 
        px_z_scale = px_z_scale.reshape(K, -1, wavelength.shape[1])

        return self.px_z(px_z_loc, px_z_scale)
    
    def generate(self, N, x):
        self.eval()
        _, wavelength, phase, mask = x
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            zs = pz.rsample(torch.Size([N,1]))
            px_z = self.decode(zs, x)
            data = px_z.mean.unsqueeze(0)
        return data