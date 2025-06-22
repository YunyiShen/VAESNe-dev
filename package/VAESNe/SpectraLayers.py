import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import * # useful base layers

###############################
# Transformers for spectra data
###############################


class spectraTransformerDecoder(nn.Module):
    def __init__(self, 
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
        '''
        super(spectraTransformerDecoder, self).__init__()
        
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] 
                                                )
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim]) # expand bottleneck to flux and time
        #self.get_photo = singlelayerMLP(model_dim, 1) # expand bottleneck to flux and wavelength
        self.get_flux = singlelayerMLP(model_dim, 1)
    
    def forward(self, wavelength, phase, bottleneck, mask=None):
        '''
        Args:
            wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
            phase: phase of the spectra being taken [batch_size, 1]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        wavelength_embd = self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        x =  wavelength_embd #+ phase_embd
        h = x
        bottleneck = self.contextfc(bottleneck)
        bottleneck = torch.concat([bottleneck, phase_embd], dim=1)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return self.get_flux(x + h).squeeze(-1) # residual connection

# this will generate bottleneck, in encoder
class spectraTransformerEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout = 0.1, 
                 selfattn = False, 
                 concat = True):
        '''
        Transformer encoder for spectra, with cross attention pooling
        Args:
            num_bands: number of bands, currently embedded as class
            bottleneck_length: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given spectra

        '''
        super(spectraTransformerEncoder, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        
        
        
        self.flux_embd = nn.Linear(1, model_dim)
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)
        self.concat = concat
        if concat:
            self.spectrafc = MLP(2*model_dim, model_dim, [model_dim])
            self.wavelength_embd_layer = SinusoidalPositionalEmbedding(model_dim)# expand phase to bottleneck

        else:
            self.spectrafc = None
            self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck


    def forward(self, wavelength, flux, phase, mask=None):
        '''
        Args:
            wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
            flux: flux of the spectra being taken of shape [batch_size, spectra_length]
            phase: phase of the spectra being taken [batch_size, 1]
            mask: which are not measured [batch_size, spectra_length]
        Return:
            Encoded spectra of shape [batch_size, bottleneck_length, bottleneck_dim]
        '''
        if self.concat:
            flux_embd = self.spectrafc(torch.cat([ self.flux_embd(flux[:, :, None]), self.wavelength_embd_layer(wavelength)], axis = -1))
        else:

            flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        context = torch.cat([flux_embd, phase_embd], dim=1) # concatenate flux and phase embd
        if mask is not None:
           # add a false at end to account for the added phase embd
           mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).bool().to(mask.device) ], dim=1)
        x = self.initbottleneck[None, :, :]
        x = x.repeat(context.shape[0], 1, 1)
        h = x
        
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=mask)
        return self.bottleneckfc(x+h) # residual connection
        


