import torch
import torch.nn.init as init
from torch.nn import functional as F
from .PhotometricLayers import photometricTransformerEncoder, photometricTransformerDecoder
from .base_vae import VAE
import torch.distributions as dist
from torch import nn

class PhotometricEnc(nn.Module):
    def __init__(self, 
                num_bands,
                latent_len,
                latent_dim,
                model_dim,
                num_heads, 
                ff_dim, 
                num_layers,
                dropout=0.1):
        super(PhotometricEnc, self).__init__()

        self.inference_transformer = photometricTransformerEncoder(
                                 num_bands, 
                                 2 * latent_len,
                                 latent_dim,
                                 model_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 dropout)
        self.latent_dim = latent_dim
        self.latent_len = latent_len
                                   

    def forward(self, flux, time, band, mask = None):
        
        bottleneck = self.inference_transformer(flux, 
                                                time, 
                                                band,
                                                mask)

        
        # q(z|x,y)
        #breakpoint()
        #mu = bottleneck[:,:,:self.latent_dim] # should it be dimension or should it be length??
        #var = F.softplus( bottleneck[:,:,self.latent_dim:])
        mu = bottleneck[:,:self.latent_len,:]
        var = F.softplus( bottleneck[:,self.latent_len:,:])
        
        return mu, var

class PhotometricDec(nn.Module):
    def __init__(self, photometric_length,
                 latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout=0.1):
        super(PhotometricDec, self).__init__()

        # p(x|z)
        self.generativetransformer = photometricTransformerDecoder(
                photometric_length,
                latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout)

    
    # p(x|z)
    def pxz(self, time, band, z, mask = None):
        # note that the z here is latent not redshift
        flux = self.generativetransformer(time, band, z, mask)
        return flux

    def forward(self, time, band, z, mask = None):
        x_rec = self.pxz(time, band, z, mask)
        var = torch.ones_like(x_rec) 
        if mask is not None:
            var += 1e8 * mask # if masked variance is large 
        return x_rec, var


class PhotometricVAE(VAE):
    def __init__(self, photometric_length = 60,
                num_bands = 6,
                latent_len = 8,
                latent_dim = 4,
                model_dim = 64, 
                num_heads = 4, 
                ff_dim = 64, 
                num_layers = 4,
                dropout = 0.1,
                prior = dist.Laplace,
                likelihood = dist.Laplace,
                posterior = dist.Laplace
                ):
        super(PhotometricVAE, self).__init__(
            prior,  # prior
            likelihood,  # likelihood
            posterior,  # posterior
            PhotometricEnc(num_bands,
                latent_len,
                latent_dim,
                model_dim,
                num_heads, 
                ff_dim, 
                num_layers,
                dropout),
            PhotometricDec(photometric_length,
                 latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout),
            params = [photometric_length,
                num_bands,
                latent_len,
                latent_dim,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout]
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(latent_len, latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(latent_len, latent_dim), requires_grad=False)  # logvar
        ])
        self.llik_scaling = 1.
        self.modelName = 'light_curve'
        self.photometric_length = photometric_length
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        #self.dataSize = dataSize
    
    def forward(self, x, K = 1):
        flux, time, band, mask = x
        self._qz_x_params = self.enc(flux, time, band, mask)
        if torch.isnan(self._qz_x_params[0]).any() or torch.isnan(self._qz_x_params[1]).any():
            breakpoint()
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        '''
        px_z_loc, px_z_scale = self.dec(time.unsqueeze(0).expand(K, -1, -1).view(-1, time.shape[-1]), 
                                        band.unsqueeze(0).expand(K, -1, -1).view(-1, band.shape[-1]), 
                                        zs.view(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).view(-1, mask.shape[-1]))
        
        px_z_loc = px_z_loc.view(K, -1, flux.shape[-1])
        px_z_scale = px_z_scale.view(K, -1, flux.shape[-1])

        px_z = self.px_z(px_z_loc, px_z_scale)
        '''
        px_z = self.decode(zs, x)
        return qz_x, px_z, zs
    
    
    def encode(self, x):
        flux, time, band, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, time, band, mask))
        return qz_x.mean

    def decode(self, zs, x):
        _, time, band, mask = x
        K = zs.shape[0]
        px_z_loc, px_z_scale = self.dec(time.unsqueeze(0).expand(K, -1, -1).view(-1, time.shape[-1]), 
                                        band.unsqueeze(0).expand(K, -1, -1).view(-1, band.shape[-1]), 
                                        zs.view(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).view(-1, mask.shape[-1]))
        
        px_z_loc = px_z_loc.view(K, -1, self.photometric_length)
        px_z_scale = px_z_scale.view(K, -1, self.photometric_length)

        return self.px_z(px_z_loc, px_z_scale)

    def reconstruct(self, x):
        flux, time, band, mask = x
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(flux, time, band, mask))
            zs = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(time, band, zs, mask))
            recon = get_mean(px_z)
        return recon
    
    def generate(self, N, time, band, mask = None):
        self.eval()
        with torch.no_grad():
            px_z_param = self.dec(time.unsqueeze(0).expand(K, -1, -1).view(-1, time.shape[-1]), 
                                        band.unsqueeze(0).expand(K, -1, -1).view(-1, band.shape[-1]), 
                                        zs.view(-1, zs.shape[-2], zs.shape[-1]), 
                                        mask.unsqueeze(0).expand(K, -1, -1).view(-1, mask.shape[-1]))
            px_z = self.px_z(px_z_param)
            data = px_z.mean
        return data

