import glob
import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from VAESNe.data_util import ImagePathDataset
from VAESNe.ImageVAE import HostImgVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo

png_files = np.array(glob.glob("../data/ZTFBTS/hostImgs/*.png"))
n_imgs = len(png_files)
n_train = int(n_imgs * 0.8)
training_list = png_files[:n_train]
testing_list = png_files[n_train:]

np.savez("./train_test_split.npz", train = training_list, 
        test = testing_list)

training_data = ImagePathDataset(training_list.tolist())
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

lr = 2.5e-4
epochs = 500
my_vaesne = HostImgVAE(
                    img_size = 60, 
                    latent_len = 4,
                    latent_dim = 2,
                    
                    patch_size=4, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False
)

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
        plt.savefig("./logs/training_ZTFimg.png")
        plt.close()
        torch.save(my_vaesne, f'../ckpt/first_hostimgvaesne_4-2_{lr}_{epochs}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")


