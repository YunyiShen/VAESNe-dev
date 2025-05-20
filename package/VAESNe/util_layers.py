import torch
from torch import nn
from torch.nn import functional as F
import math



########### simple MLPs ###############
class singlelayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(singlelayerMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = [64,64]):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim[i]))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


############ Transformer use ##################
################# positional encoding ###################

class learnable_fourier_encoding(nn.Module):
    def __init__(self, dim = 64):
        '''
        Learnable Fourier encoding for position, 
        MLP([sin(fc(x)), cos(fc(x))])
        Args:
            dim: dimension
        '''
        super(learnable_fourier_encoding, self).__init__()
        self.freq = nn.Linear(1, dim, bias=False)
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x[:, :, None]
        encoding = torch.cat([torch.sin(self.freq(x)), 
                              torch.cos(self.freq(x))], dim=-1)
        encoding = nn.ReLU( self.fc1(encoding) )
        encoding = self.fc2(encoding)
        return encoding


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim = 64):
        '''
        The usual sinusoidal positional encoding
        args: 
            dim: the dimension
        '''
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        # Create the positional encoding matrix

    def forward(self, x):
        # x: [batch_size, seq_len]
        sine = torch.sin(x[:,:,None] * self.div_term[None,None,:].to(x.device)) 
        cosine = torch.cos(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        return torch.cat([sine, cosine], dim=-1)

class SinusoidalMLPPositionalEmbedding(nn.Module):
    def __init__(self, dim = 64):
        '''
        The usual sinusoidal positional encoding with an extra MLP, inspired by https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html
        '''
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        sine = torch.sin(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        cosine = torch.cos(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        encoding = torch.cat([sine, cosine], dim=-1)
        encoding = F.relu( self.fc1(encoding) )
        encoding = self.fc2(encoding)
        return encoding


class RelativePosition(nn.Module):
    '''
    relative positional encoding for discrete distances
    '''
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(self.embeddings_table.device)
        embeddings = self.embeddings_table[final_mat].to(self.embeddings_table.device)

        return embeddings

######################### attention blocks ######################

class MultiHeadAttentionLayer_relative(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        '''
        Multiheaded attention with relative positional encoding
        '''
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, 
                 dropout=0.1, 
                 context_self_attn = False):
        '''
        Usual transformer block allowing context
        '''
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                                dropout=dropout, batch_first=True)
        if context_self_attn:
            self.context_self_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                                dropout=dropout, batch_first=True)
            self.layernorm_context = nn.LayerNorm(embed_dim)
        else:
            self.context_self_attn = None
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        # we made x [batch, seq_len, embed_dim]


        attn_output, _ = self.self_attn(x, x, x, 
                                        key_padding_mask = mask)
            # in decoder mask whereever not observed
        x = self.layernorm1(x + self.dropout(attn_output))

        # Cross-attention (if context is provided)
        if context is not None:
            if self.context_self_attn:
                context_attn_output, _ = self.context_self_attn(context, context, context,
                                                                key_padding_mask=context_mask)
                context = self.layernorm_context(context + self.dropout(context_attn_output))
            #breakpoint()
            cross_attn_output, _ = self.cross_attn(x, context, context,
                                                       key_padding_mask=context_mask)
            x = self.layernorm2(x + self.dropout(cross_attn_output))

        # Feedforward
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))

        return x


############## vae use ###################
def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Reshape layer        
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
         
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
                return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 
    
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y

########### image use ############
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # shape: [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # shape: [B, embed_dim, N]
        x = x.transpose(1, 2)  # shape: [B, N, embed_dim]
        return x



class TransformerModel(nn.Module):
    '''
    A minimal transformer model
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, selfattn = True):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, selfattn) 
            for _ in range(num_layers)
        ])

    def forward(self, x, context=None):
        for layer in self.layers:
            x = layer(x, context)
        return x