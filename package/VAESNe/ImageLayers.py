import torch
import torch.nn as nn
from .util_layers import *
import math

class HostImgTransformerEncoder(nn.Module):
    def __init__(self, 
                    img_size,
                    bottleneck_length,
                    bottleneck_dim,
                    patch_size=4, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False, 
                    sincosin = True):
        super().__init__()
        assert img_size % patch_size == 0, "image size has to be divisible to patch size"
        self.focal_loc = focal_loc
        self.model_dim = model_dim
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, model_dim)
        if sincosin:
            self.pos_embed = SinusoidalPositionalEmbedding2D(model_dim, img_size//patch_size,img_size//patch_size)()
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, model_dim))
        if self.focal_loc:
            self.eventloc_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        else:
            self.eventloc_embd = None

        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)

    def forward(self, image, event_loc = None):
        image_embd = self.patch_embed(image)  # [B, N, D]

        image_embd = image_embd + self.pos_embed  # [B, N, D]
        if self.focal_loc:
            if event_loc is not None:
                event_loc_embd = self.eventloc_embd(event_loc) # [B, 2, D]
            else:
                event_loc_embd = self.eventloc_embd(torch.zeros(image_embd.shape[0], 2))
            context = torch.cat([image_embd, event_loc_embd], dim=1) # [B, N+2, D]
        else:
            context = image_embd
        x = self.initbottleneck[None, :, :]
        x = x.repeat(context.shape[0], 1, 1)
        h = x
        
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=None)
        return self.bottleneckfc(x+h) # residual connection


class HostImgTransformerDecoder(nn.Module):
    def __init__(self,
                img_size,
                bottleneck_dim,
                #patch_size=4, 
                in_channels=3,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout=0.1, 
                selfattn=False, 
                mlpdecoder = True):
        super().__init__()
        #assert img_size % patch_size == 0, "patch size has to be divisible to image size"
        self.img_size = img_size
        #self.patch_size = patch_size
        self.in_channels = in_channels
        #breakpoint()
        #self.grid_size = img_size // patch_size
        #self.num_patches = self.grid_size ** 2
        #self.patch_dim = in_channels * patch_size * patch_size
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.init_img_embd = SinusoidalPositionalEmbedding2D(model_dim, img_size, img_size)() #nn.Parameter(torch.randn(img_size ** 2, model_dim))
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] )
        
        if mlpdecoder:
            self.decoder = MLP(model_dim, in_channels, [model_dim])
        else: 
            self.decoder = nn.Linear(model_dim, in_channels)
    
    def forward(self, bottleneck):
        x =  self.init_img_embd[None,:,:].expand(bottleneck.shape[0], -1, -1) # expand in batch
        #breakpoint()
        h = x
        bottleneck = self.contextfc(bottleneck)
        #breakpoint()
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck)
        h = h+x
        h = self.decoder(h)
        h = h.view(x.shape[0], self.img_size, self.img_size, self.in_channels).permute(0, 3, 1, 2)
        #h = h.view(x.shape[0], self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.in_channels)
         # [B, W//patch, W//patch, channel*patch*patch]
        #h = h.permute(0, 5, 1, 3, 2, 4)
        return h#.reshape(x.shape[0], self.in_channels, self.img_size, self.img_size)        