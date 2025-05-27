import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt



from VAESNe.training_util import training_step
from VAESNe.losses import negInfoNCE
from VAESNe.data_util import multimodalDataset
from VAESNe.contrastiveNets import ContraPhotSpec

torch.manual_seed(0)


data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']

######## spectra dataset #######
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
# randomly set some masks to be True
mask = torch.logical_or(mask, torch.rand_like(flux) < 0.05)
#breakpoint()

# split loaded data into training and validation
spectra_train_dataset = TensorDataset(flux, wavelength, phase, mask)
spectra_test_dataset = TensorDataset(flux_test, wavelength_test, phase_test, mask_test)
spectra_test_dataset, spectra_val_dataset = random_split(spectra_test_dataset, [0.5, 0.5])

########### photometry #######
photoflux, phototime, photomask = data['photoflux'][training_idx,:], data['phototime'][training_idx,:], data['photomask'][training_idx,:]
photoband = data['photowavelength'][training_idx,:]

photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
photoband_test = data['photowavelength'][testing_idx]


photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)

photoflux_test = torch.tensor(photo_flux_test, dtype=torch.float32)
phototime_test = torch.tensor(phototime_test, dtype=torch.float32)
photomask_test = torch.tensor(photomask_test == 0)
photoband_test = torch.tensor(photoband_test, dtype=torch.long)


# do some data augmentation on flux and time, the data is already repeated multiple times 
photoflux = photoflux + 0.02 * torch.randn_like(photoflux)
phototime = phototime + 0.1 * torch.randn(phototime.shape[0])[:,None] # shift all time in a single light curve by the same amount
# randomly set some masks to be True
photomask = torch.logical_or(photomask, torch.rand_like(photoflux) < 0.05)
#breakpoint()

# split loaded data into training and validation
photometric_train_dataset = TensorDataset(photoflux, phototime, photoband, photomask)
photometric_test_dataset = TensorDataset(photoflux_test, phototime_test, photoband_test, photomask_test)
photometric_test_dataset, photometric_val_dataset = random_split(photometric_test_dataset, [0.5, 0.5])


photo_spect_train = multimodalDataset(photometric_train_dataset, 
                spectra_train_dataset)

train_loader = DataLoader(photo_spect_train, batch_size=16, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

lr = 2.5e-4
epochs = 500


contrastnet = ContraPhotSpec(
    # latent things
        latent_len = 4,
        latent_dim = 4,
        proj_dim = 8,
        # photometric things
        num_bands = 6,
        photo_model_dim = 32,
        photo_num_heads = 4, 
        photo_ff_dim = 32, 
        photo_num_layers = 4,
        photo_dropout = 0.1,

        # spectra things
        spec_model_dim = 32, 
        spec_num_heads = 4, 
        spec_num_layers = 4,
        spec_ff_dim = 32, 
        spec_dropout = 0.1,
        selfattn = False#True
).to(device)

optimizer = AdamW(contrastnet.parameters(), lr=lr)
all_losses = np.ones(epochs) + np.nan
steps = np.arange(epochs)

from tqdm import tqdm
progress_bar = tqdm(range(epochs))
for i in progress_bar:
    loss = training_step(contrastnet, optimizer,train_loader, 
                    loss_fn = lambda model, x: negInfoNCE(model, x, temperature=0.1), 
                    multimodal = True)
    all_losses[i] = loss
    if (i + 1) % 5 == 0:
        plt.plot(steps, all_losses)
        plt.xlabel("training epochs")
        plt.ylabel("loss")
        plt.show()
        plt.savefig("./logs/training_specphotocontrast.png")
        plt.close()
        torch.save(contrastnet, f'../ckpt/goldstein_photospectra_contrast_4-4_{lr}_{epochs}.pth')
    progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")






