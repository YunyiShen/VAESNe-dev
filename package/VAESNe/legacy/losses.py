
class VAEloss(nn.Module):
    def __init__(self, beta=1.0):
        super(VAEloss, self).__init__()
        self.beta = beta

    def forward(self, x, x_rec, mu, var, mask=None):
        # Reconstruction loss
        if mask is not None:
            rec_loss = F.mse_loss(x_rec, x, reduction='none')
            rec_loss = rec_loss * (1.-1.*mask) # this mask is for attention so true is missing data
            rec_loss = rec_loss.sum() / (1.-1.*mask).sum()
        else:
            rec_loss = F.mse_loss(x_rec, x, reduction='mean')
        
        # flatten the latent variables
        
        mu = mu.reshape(mu.size(0), -1)
        var = var.reshape(var.size(0), -1)
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
        #breakpoint()
        return rec_loss, self.beta * kl_loss

