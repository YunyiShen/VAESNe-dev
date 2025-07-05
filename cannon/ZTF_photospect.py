import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
from matplotlib import pyplot as plt
import os

from VAESNe.SpectraVAE import BrightSpectraVAE, SpectraVAE
from VAESNe.PhotometricVAE import BrightPhotometricVAE, PhotometricVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE

torch.manual_seed(0)

### dataset ###
# data = np.load("../data/train_data_align_with_simu_minimal.npz")
# data = np.load("/n/holystore01/LABS/iaifi_lab/Lab/specgen_shen_gagliano/generative-spectra-lightcurves/data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz")
data = np.load("/n/holystore01/LABS/iaifi_lab/Lab/specgen_shen_gagliano/VAESNe-dev/data/train_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
phase = data['phase']

flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

print("Flux mean:", flux.mean().item(), "std:", flux.std().item())

### photometry ### 
photoflux, phototime, photomask = data['photoflux'], data['phototime'], data['photomask']
photoband = data['photowavelength']

photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)

print("Photoflux mean:", photoflux.mean().item(), "std:", photoflux.std().item())




### photometry ### 
photoflux, phototime, photomask = data['photoflux'], data['phototime'], data['photomask']
photoband = data['photowavelength']

photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)

num_bands = int(photoband.max().item() + 1)
print("num_bands:", num_bands)
#breakpoint()

#### augmentation ####
#### do some data augmentation ####
factor = 10

# spectra
flux = flux.repeat((10,1))
wavelength = wavelength.repeat((10,1))
mask = mask.repeat((10,1))
phase = phase.repeat((10))

flux = flux + torch.randn_like(flux) * 0.01 # some noise 
phase = phase + torch.randn_like(phase) * 0.001
mask = torch.logical_or(mask, torch.rand_like(flux)<=0.1) # do some random masking

# photometry 
photoflux = photoflux.repeat((10,1))
photoband = photoband.repeat((10,1))
photomask = photomask.repeat((10,1))
phototime = phototime.repeat((10,1))

photoflux = photoflux + torch.randn_like(photoflux) * 0.01 # some noise 
phototime = phototime + torch.randn_like(phototime) * 0.001
photomask = torch.logical_or(photomask, torch.rand_like(photoflux)<=0.1) # do some random masking



### data loader ### 
spectra_train_dataset = TensorDataset(flux, wavelength, phase, mask)
photometric_train_dataset = TensorDataset(photoflux, phototime, photoband, photomask)
photo_spect_train = multimodalDataset(photometric_train_dataset, 
                spectra_train_dataset)

# train_loader = DataLoader(photo_spect_train, batch_size=16, shuffle=True)
train_loader = DataLoader(photo_spect_train, batch_size=16, shuffle=True)  # Use batch size 4 if using the real data, otherwise wiill run out of memory

lr = 1e-3 #2.5e-4
epochs = 200
K=2
latent_len = 4
latent_dim = 4
beta = 0.5
model_dim = 32
num_layers = 4

my_spectravae = SpectraVAE(
    # data parameters
    # spectra_length = flux.shape[1],  # This parameter doesn't exist in SpectraVAE class

    # model parameters
    latent_len = latent_len,
    latent_dim = latent_dim,
    model_dim = model_dim, 
    num_heads = 4, 
    ff_dim = model_dim, 
    num_layers = num_layers,
    dropout = 0.1,
    selfattn = True #True
    ).to(device)


my_photovae = PhotometricVAE(
    # photometric_length = photoflux.shape[1],  # This parameter doesn't exist in PhotometricVAE class
    num_bands = num_bands,  # 2 (this leads to index out of range error)
    # model parameters
    latent_len = latent_len,
    latent_dim = latent_dim,
    model_dim = model_dim, 
    num_heads = 4, 
    ff_dim = model_dim, 
    num_layers = num_layers,
    dropout = 0.1,
    selfattn = False #True
    ).to(device)

my_mmvae = photospecMMVAE(vaes = [my_photovae, my_spectravae], beta = beta).to(device)

optimizer = AdamW(my_mmvae.parameters(), lr=lr)
all_losses = np.ones(epochs) + np.nan
steps = np.arange(epochs)

# Make log & checkpoint directories if they don't exist
os.makedirs("./logs", exist_ok=True)
os.makedirs("../ckpt", exist_ok=True)

from tqdm import tqdm
progress_bar = tqdm(range(epochs))
for i in progress_bar:
    loss = training_step(my_mmvae, optimizer,train_loader, 
                    loss_fn = lambda model, x: m_iwae(model, x, K=K), 
                    multimodal = True)
    all_losses[i] = loss
    # Compute approximate per-element loss
    # Adjust these estimates based on your exact batch tensor shapes
    batch_size = 16  # or use your variable if dynamically set
    # estimate total number of elements (approximate)
    sequence_length_flux = flux.shape[1]
    sequence_length_photo = photoflux.shape[1]
    num_elements = batch_size * (sequence_length_flux + sequence_length_photo)

    per_element_loss = loss / num_elements
    print(f"Epoch {i}: total loss={loss:.2f}, per-element loss={per_element_loss:.4f}")
    if (i + 1) % 5 == 0:
        plt.plot(steps, all_losses)
        plt.xlabel("training epochs")
        plt.ylabel("loss")
        plt.show()
        plt.savefig("./logs/ZTF_training_specphoto.png")
        plt.close()
        torch.save(my_mmvae, f'../ckpt/ZTF_photospectravaesne_{latent_len}-{latent_dim}_{lr}_{epochs}_K{K}_beta{beta}_modeldim{model_dim}_numlayer{num_layers}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")

