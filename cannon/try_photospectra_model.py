import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.mmVAE import photospecMMVAE


data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']

flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase_test = data['phase'][testing_idx]

photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
photoband_test = data['photowavelength'][testing_idx]



idx = 17

flux_test = torch.tensor(flux_test, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength_test, dtype=torch.float32)
mask_test = torch.tensor(mask_test == 0)
phase_test = torch.tensor(phase_test, dtype=torch.float32)



photoflux_test = torch.tensor(photo_flux_test, dtype=torch.float32)
phototime_test = torch.tensor(phototime_test, dtype=torch.float32)
photomask_test = torch.tensor(photomask_test == 0)
photoband_test = torch.tensor(photoband_test, dtype=torch.long)


trained_vae = torch.load("../ckpt/first_photospectravaesne_4-2_0.00025_500.pth",
                         map_location=torch.device('cpu'), weights_only = False)

data = [
    ### photometry
    (photoflux_test[idx][None,:],
    phototime_test[idx][None,:],
    photoband_test[idx][None,:],
    photomask_test[idx][None,:]), 
    ### spectra
    (flux_test[idx][None,:],
    wavelength_test[idx][None,:],
    phase_test[idx][None],
    mask_test[idx][None,:])
]

#breakpoint()
reconstruction = trained_vae.reconstruct(data) # [0][0] LC->LC, [1][0]: spec->LC, [0][1]: LC-> Spec, [1][1]: spec-> spec

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

for i in range(6):
    thisband_gt = photoflux_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]
    thisband_time = phototime_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]
    thisband_rec = reconstruction[0][0][0,0].detach()
    thisband_crossrec = reconstruction[1][0][0,0].detach()
    
    thisband_rec = thisband_rec[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    thisband_crossrec = thisband_crossrec[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    
    axs[0].plot(thisband_time, thisband_gt)
    axs[0].scatter(thisband_time, thisband_gt, s=20, marker='o')

    axs[1].plot(thisband_time, thisband_rec)
    axs[1].scatter(thisband_time, thisband_rec, s=20, marker='x')

    axs[2].plot(thisband_time, thisband_crossrec)
    axs[2].scatter(thisband_time, thisband_crossrec, s=20, marker='x')

# invert y
axs[0].set_ylim(-2, 6)
axs[1].set_ylim(-2, 6)
axs[2].set_ylim(-2, 6)
axs[0].invert_yaxis()
axs[0].set_title("Ground truth")
axs[1].invert_yaxis()
axs[1].set_title("Reconstruction-LC")
axs[2].invert_yaxis()
axs[2].set_title("Reconstruction-Spectra")


#plt.tight_layout()

plt.show()
plt.savefig("LC_reconstruction.png")
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.plot(wavelength_test[idx], flux_test[idx], label='ground truth')
axs.plot(wavelength_test[idx], reconstruction[1][1][0,0].detach().numpy(), label='Rec-spec')
axs.plot(wavelength_test[idx], reconstruction[0][1][0,0].detach().numpy(), label='Rec-LC')




# invert y
axs.set_ylim(-2, 2)
axs.legend()



#plt.tight_layout()

plt.show()
plt.savefig("spectra_reconstruction.png")
plt.close()




