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
from VAESNe.PhotometricVAE import PhotometricVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE


### dataset ###
test_data = np.load("../data/test_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = test_data['flux'], test_data['wavelength'], test_data['mask']
phase = test_data['phase']

wavelength_mean, wavelength_std = test_data['wavelength_mean'], test_data['wavelength_std']
flux_mean, flux_std = test_data['flux_mean'], test_data['flux_std']
phase_mean, phase_std = test_data['spectime_mean'], test_data['spectime_std']
phototime_mean, phototime_std = test_data['combined_time_mean'], test_data['combined_time_std']
photoflux_mean, photoflux_std = test_data['combined_mean'], test_data['combined_std']
#breakpoint()


flux_test = torch.tensor(flux, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength, dtype=torch.float32)
mask_test = torch.tensor(mask == 0)
phase_test = torch.tensor(phase, dtype=torch.float32)


idx = 22
data = (flux_test[idx][None,:],
    wavelength_test[idx][None,:],
    phase_test[idx][None], #* 0 + torch.tensor((40-phase_mean)/phase_std, dtype = torch.float32),
    mask_test[idx][None,:])



#breakpoint()
### 
trained_vae = torch.load("../ckpt/ZTF_spectravaesne_4-2_0.001_200_beta0.5_modeldim32.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False)


#breakpoint()
with torch.no_grad():
    reconstruction = trained_vae.reconstruct(data, K=100) # [0][0] LC->LC, [1][0]: spec->LC, [0][1]: LC-> Spec, [1][1]: spec-> spec

    #photo_only_recon = photo_only.reconstruct(data[0])
    #spectra_only_recon = spectra_only.reconstruct(data[1], K=100)
    #photo_encode = photo_only.encode(data[0])[None, ...]
    #spectra_encoded = spectra_only.encode(data[1])[None, ...]
    #photo_encode_spec_decode = spectra_only.decode(photo_encode, data[1]).mean
    #spec_encode_photo_decode = photo_only.decode(spectra_encoded, data[0]).mean




import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(10, 6))


axs.plot(wavelength_test[idx][torch.logical_not(mask_test[idx])] * wavelength_std + wavelength_mean, 
         flux_test[idx][torch.logical_not(mask_test[idx])] * flux_std + flux_mean, 
         label='ground truth', color = "red")

axs.plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
         reconstruction.mean(axis = 0)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
         label='Rec-spec', color = "blue")

axs.fill_between(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean,
                    reconstruction.quantile(dim=0, q=0.025)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
                    reconstruction.quantile(dim=0, q=0.975)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean,
                    color = "blue", alpha=0.3)






# invert y
axs.set_ylabel("log Fnu")
axs.set_xlabel("wavelength (Å)")
axs.legend()

plt.show()
plt.savefig("ZTF_spectraonly_reconstruction.pdf")
plt.close()



N = 30
generation = trained_vae.generate(x = data, N = N)
#breakpoint()

fig, axs = plt.subplots(2, 1, figsize=(10, 5))

for i in range(N):
    axs[0].plot(wavelength_test[i][torch.logical_not(mask_test[i])]* wavelength_std + wavelength_mean, 
         flux_test[i][torch.logical_not(mask_test[i])].detach().numpy() * flux_std + flux_mean, 
         label='Spec-gt', alpha = 0.5)

    #breakpoint()
    axs[1].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
         generation[0][i,0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
         label='Spec-samples', alpha = 0.5)
for i in range(2):
    axs[i].set_ylabel("log Fnu")
    axs[i].set_xlabel("wavelength (Å)")
    axs[i].set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
plt.show()
plt.savefig("ZTF_spectraonly_priorsamples.pdf")
plt.close()

print( phase_test[idx] * phase_std + phase_mean)