import torch
import torch.nn as nn
from .util_layers import *

class HostImgEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                    bottleneck_dim,
                    img_size=224, 
                    patch_size=16, 
                    in_channels=3,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False):
        super().__init__()
        self.model_dim = model_dim
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.eventloc_embd = SinusoidalMLPPositionalEmbedding(model_dim)

        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)

    def forward(self, image, event_loc):
        image_embd = self.patch_embed(image)  # [B, N, D]
        image_embd = image_embd + self.pos_embed  # [B, N, D]
        event_loc_embd = self.eventloc_embd(event_loc) # [B, 2, D]
        context = torch.cat([image_embd, event_loc_embd], dim=1) # [B, N+2, D]
        x = self.initbottleneck[None, :, :]
        x = x.repeat(context.shape[0], 1, 1)
        h = x
        
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=None)
        return self.bottleneckfc(x+h) # residual connection


class HostImgDecoder(nn.Module):
    def __init__(self,
                bottleneck_dim,
                img_size=224, 
                patch_size=16, 
                in_channels=3,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout=0.1, 
                selfattn=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.init_img_embd = nn.Parameter(torch.randn(patch_dim , model_dim))

        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] )
    
    def forward(self, bottleneck):
        x =  self.init_img_embd[None,:,:]
        h = x
        bottleneck = self.contextfc(bottleneck)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return self.depatchify(x + h)
        


    def depatchify(self, patches):
        """
        Convert [B, N, patch_dim] → [B, C, H, W]
        """
        B, N, _ = patches.shape
        P = self.patch_size
        C = self.in_channels
        H = W = self.grid_size

        patches = patches.view(B, H, W, C, P, P)       # [B, H, W, C, P, P]
        patches = patches.permute(0, 3, 1, 4, 2, 5)    # [B, C, H, P, W, P]
        img = patches.reshape(B, C, H * P, W * P)      # [B, C, H*P, W*P]
        return img