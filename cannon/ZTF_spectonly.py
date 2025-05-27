import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt


from VAESNe.SpectraVAE import BrightSpectraVAE, SpectraVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE

torch.manual_seed(0)

### dataset ###
data = np.load("../data/train_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
phase = data['phase']

flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

#### do some data augmentation ####
factor = 10
flux = flux.repeat((10,1))
wavelength = wavelength.repeat((10,1))
mask = mask.repeat((10,1))
phase = phase.repeat((10))

flux = flux + torch.randn_like(flux) * 0.01 # some noise 
phase = phase + torch.randn_like(phase) * 0.001
mask = torch.logical_or(mask, torch.rand_like(flux)<=0.05) # do some random masking



spectra_train_dataset = TensorDataset(flux, wavelength, phase, mask)

train_loader = DataLoader(spectra_train_dataset, batch_size=16, shuffle=True)
lr = 1e-3 #2.5e-4
epochs = 200
latent_len = 4
latent_dim = 4
beta = 0.5
model_dim = 32

my_spectravae = SpectraVAE(
    # data parameters
    spectra_length = flux.shape[1],

    # model parameters
    latent_len = latent_len,
    latent_dim = latent_dim,
    model_dim = model_dim, 
    num_heads = 4, 
    ff_dim = model_dim, 
    num_layers = 4,
    dropout = 0.1,
    selfattn = False, #True
    beta = beta
    ).to(device)


optimizer = AdamW(my_spectravae.parameters(), lr=lr)
all_losses = np.ones(epochs) + np.nan
steps = np.arange(epochs)

from tqdm import tqdm
progress_bar = tqdm(range(epochs))
for i in progress_bar:
    loss = training_step(my_spectravae, optimizer,train_loader, 
                    loss_fn = elbo, 
                    multimodal = False)
    all_losses[i] = loss
    if (i + 1) % 5 == 0:
        plt.plot(steps, all_losses)
        plt.xlabel("training epochs")
        plt.ylabel("loss")
        plt.show()
        plt.savefig("./logs/ZTF_training_spect.png")
        plt.close()
        torch.save(my_spectravae, f'../ckpt/ZTF_spectravaesne_{latent_len}-{latent_dim}_{lr}_{epochs}_beta{beta}_modeldim{model_dim}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")

