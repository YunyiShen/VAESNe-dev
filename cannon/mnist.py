from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VAESNe.ImageVAE import HostImgVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo
from torch.optim import AdamW
import torch
import numpy as np

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.ToTensor(),
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist, batch_size=32, shuffle=True)

lr = 1e-3
epochs = 50
latent_len = 4
latent_dim = 4
beta = 0.1
patch_size = 3
model_dim = 32
num_layers = 4

my_vaesne = HostImgVAE(
                    img_size = 60, 
                    latent_len = latent_len,
                    latent_dim = latent_dim,
                    
                    patch_size=patch_size, 
                    in_channels=1,
                    focal_loc = False,
                    model_dim = model_dim, 
                    num_heads = 4, 
                    ff_dim = model_dim, 
                    num_layers = num_layers,
                    dropout=0.1, 
                    selfattn=False, 
                    beta = beta
).to(device)

#breakpoint()

optimizer = AdamW(my_vaesne.parameters(), lr=lr)
all_losses = np.ones(epochs) + np.nan
steps = np.arange(epochs)
from tqdm import tqdm
progress_bar = tqdm(range(epochs))
for i in progress_bar:
    loss = training_step(my_vaesne, optimizer, train_loader, elbo)
    all_losses[i] = loss
    if (i + 1) % 5 == 0:
        plt.plot(steps, all_losses)
        plt.show()
        plt.savefig("./logs/mnist.png")
        plt.close()
        torch.save(my_vaesne, f'../ckpt/mnist_{latent_len}-{latent_dim}_{lr}_{epochs}_patch{patch_size}_beta{beta}_modeldim{model_dim}_numlayers{num_layers}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")



