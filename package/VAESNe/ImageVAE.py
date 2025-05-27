import torch
import torch.nn.init as init
from torch.nn import functional as F
from .ImageLayers import HostImgTransformerEncoder, HostImgTransformerDecoder, HostImgTransformerDecoderHybrid
from .base_vae import VAE
import torch.distributions as dist
from torch import nn

class HostImgEnc(nn.Module):
    def __init__(self, 
                    img_size, 
                    latent_len,
                    latent_dim,
                    
                    patch_size=4, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False
                ):
        super(HostImgEnc, self).__init__()

        self.inference_transformer = HostImgTransformerEncoder(
                                 img_size, 
                    2 * latent_len,
                    latent_dim,
                    
                    patch_size, 
                    in_channels,
                    focal_loc,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout, 
                    selfattn)
        self.latent_dim = latent_dim
        self.latent_len = latent_len
                                   

    def forward(self, image, event_loc = None):
        bottleneck = self.inference_transformer(image, event_loc)
        # q(z|x,y)
        #breakpoint()
        #mu = bottleneck[:,:,:self.latent_dim] # should it be dimension or should it be length??
        #var = F.softplus( bottleneck[:,:,self.latent_dim:])
        mu = bottleneck[:,:self.latent_len,:]
        var = F.softplus( bottleneck[:,self.latent_len:,:])
        
        return mu, var

class HostImgDec(nn.Module):
    def __init__(self, img_size,
                latent_dim,
                patch_size=4, 
                in_channels=3,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout=0.1, 
                selfattn=False,
                hybrid = True
                ):
        super(HostImgDec, self).__init__()

        # p(x|z)
        if hybrid:
            self.generativetransformer = HostImgTransformerDecoderHybrid(
                img_size,
                latent_dim,
                patch_size, 
                in_channels,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout, 
                selfattn)
        else:
            # pixel directly
            self.generativetransformer = HostImgTransformerDecoder(
                img_size,
                latent_dim,
                in_channels,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout, 
                selfattn)

    
    # p(x|z)
    def pxz(self, z):
        # note that the z here is latent not redshift
        flux = self.generativetransformer(z)
        return flux

    def forward(self, z):
        x_rec = self.pxz(z)
        var = torch.ones_like(x_rec) 
        return x_rec, var


class HostImgVAE(VAE):
    def __init__(self, img_size, 
                    latent_len,
                    latent_dim,
                    
                    patch_size=4, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False,
                    hybrid = True,
                    beta = 1.,
                prior = dist.Laplace,
                likelihood = dist.Laplace,
                posterior = dist.Laplace
                ):
        super(HostImgVAE, self).__init__(
            prior,  # prior
            likelihood,  # likelihood
            posterior,  # posterior
            HostImgEnc(img_size, 
                    latent_len,
                    latent_dim,
                    
                    patch_size, 
                    in_channels,
                    focal_loc,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout, 
                    selfattn),
            HostImgDec(img_size,
                latent_dim,
                patch_size, 
                in_channels,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout,
                selfattn,
                hybrid
                ),
            params = [img_size, 
                    latent_len,
                    latent_dim,
                    
                    patch_size, 
                    in_channels,
                    focal_loc,
                    model_dim, 
                    num_heads, 
                    ff_dim, 
                    num_layers,
                    dropout, 
                    selfattn]
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(latent_len, latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(latent_len, latent_dim), requires_grad=False)  # logvar
        ])
        self.llik_scaling = 1./beta
        self.modelName = 'HostImage'
        self.image_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.focal_loc = focal_loc
    
    def forward(self, x, K = 1):
        if self.focal_loc:
            image, event_loc = x
        else:
            image, event_loc = x[0], None # this is acually a bit hacky, because the training step make x a tuple
        self._qz_x_params = self.enc(image, event_loc)
        if torch.isnan(self._qz_x_params[0]).any() or torch.isnan(self._qz_x_params[1]).any():
            breakpoint()
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.decode(zs)
        return qz_x, px_z, zs
    
    
    def encode(self, x, mean = True):
        if self.focal_loc:
            image, event_loc = x
        else:
            image, event_loc = x[0], None
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(image, event_loc))
        if mean:
            return qz_x.mean
        return qz_x

    def decode(self, zs):
        K = zs.shape[0]
        px_z_loc, px_z_scale = self.dec(zs.reshape(-1, zs.shape[-2], zs.shape[-1]))
        
        px_z_loc = px_z_loc.reshape(K, -1, self.in_channels, self.image_size, self.image_size)
        px_z_scale = px_z_scale.reshape(K, -1, self.in_channels, self.image_size, self.image_size)
        #breakpoint()
        return self.px_z(px_z_loc, px_z_scale)

    def reconstruct(self, x, K=1):
        if self.focal_loc:
            image, event_loc = x
        else:
            image, event_loc = x[0], None
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(image, event_loc))
            zs = qz_x.rsample([K])  # no dim expansion
            px_z = self.decode(zs)
            recon = px_z.mean
        return recon
    
    def generate(self, N):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            zs = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(zs))
            data = px_z.mean
        return data



