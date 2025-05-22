import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt


from VAESNe.SpectraVAE import SpectraVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo

torch.manual_seed(0)


data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']


flux, wavelength, mask = data['flux'][training_idx,:], data['wavelength'][training_idx,:], data['mask'][training_idx,:]
phase = data['phase'][training_idx]

flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase_test = data['phase'][testing_idx]


flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

flux_test = torch.tensor(flux_test, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength_test, dtype=torch.float32)
mask_test = torch.tensor(mask_test == 0)
phase_test = torch.tensor(phase_test, dtype=torch.float32)

# do some data augmentation on flux and time, the data is already repeated multiple times 
flux = flux + 0.02 * torch.randn_like(flux)
#wavelength = wavelength + 0.1 * torch.randn(wavelength.shape[0])[:,None] # shift all time in a single light curve by the same amount
# randomly set some masks to be True
mask = torch.logical_or(mask, torch.rand_like(flux) < 0.05)
#breakpoint()

# split loaded data into training and validation
train_dataset = TensorDataset(flux, wavelength, phase, mask)
test_dataset = TensorDataset(flux_test, wavelength_test, phase_test, mask_test)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)


lr = 2.5e-4
epochs = 500

my_vaesne = SpectraVAE(
    # data parameters
    spectra_length = 982,

    # model parameters
    latent_len = 4,
    latent_dim = 2,
    model_dim = 32, 
    num_heads = 4, 
    ff_dim = 32, 
    num_layers = 4,
    dropout = 0.1,
    selfattn = False#True
    ).to(device)

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
        plt.savefig("./logs/training_spec.png")
        plt.close()
        torch.save(my_vaesne, f'../ckpt/first_specvaesne_4-2_{lr}_{epochs}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")


