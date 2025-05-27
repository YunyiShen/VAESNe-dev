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

from VAESNe.data_util import ImagePathDataset, ImagePathDatasetAug
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

training_data = ImagePathDatasetAug(training_list.tolist(), factor = 5)
#breakpoint()
fig, axes = plt.subplots(5, 10, figsize=(10, 8))  # 4 rows, 5 columns
axes = axes.flatten()

for i in range(50):
    img = training_data[i][0].permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC for matplotlib
    axes[i].imshow(img/2 + 0.5)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
plt.savefig("./hostimg_original.pdf")
plt.close()

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

lr = 1e-3
epochs = 150
latent_len = 4
latent_dim = 4
beta = 0.5
patch_size = 2
model_dim = 32
num_layers = 4
hybrid = True

my_vaesne = HostImgVAE(
                    img_size = 60, 
                    latent_len = latent_len,
                    latent_dim = latent_dim,
                    
                    patch_size=patch_size, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = model_dim, 
                    num_heads = 4, 
                    ff_dim = model_dim, 
                    num_layers = num_layers,
                    dropout=0.1, 
                    selfattn=False, 
                    beta = beta,
                    hybrid = hybrid
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
        plt.savefig("./logs/training_ZTFimg.png")
        plt.close()
        torch.save(my_vaesne, f'../ckpt/ZTF_hostimgvaesne_{latent_len}-{latent_dim}_{lr}_{epochs}_patch{patch_size}_beta{beta}_modeldim{model_dim}_numlayers{num_layers}_hybrid{hybrid}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")


