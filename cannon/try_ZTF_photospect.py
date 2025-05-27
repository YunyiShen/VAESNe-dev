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

### photometry ### 
photoflux, phototime, photomask = test_data['photoflux'], test_data['phototime'], test_data['photomask']
photoband = test_data['photowavelength']

photoflux_test = torch.tensor(photoflux, dtype=torch.float32)
phototime_test = torch.tensor(phototime, dtype=torch.float32)
photomask_test = torch.tensor(photomask == 0)
photoband_test = torch.tensor(photoband, dtype=torch.long)



idx = 150
data = [
    ### photometry
    (photoflux_test[idx][None,:],
    phototime_test[idx][None,:],
    photoband_test[idx][None,:],
    photomask_test[idx][None,:]), 
    ### spectra
    (flux_test[idx][None,:],
    wavelength_test[idx][None,:],
    phase_test[idx][None], #* 0 + torch.tensor((40-phase_mean)/phase_std, dtype = torch.float32),
    mask_test[idx][None,:])
]


#breakpoint()
### 
trained_vae = torch.load("../ckpt/ZTF_brightphotospectravaesne_4-2_0.001_200_K8_beta0.1_modeldim32_numlayer4.pth", # trained with K=1 on iwae
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
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

for i in range(2):
    thisband_gt = photoflux_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))] * photoflux_std + photoflux_mean
    thisband_time = phototime_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))] * phototime_std + phototime_mean

    #breakpoint()
    thisband_rec = reconstruction[0][0][0,0].detach() * photoflux_std + photoflux_mean
    thisband_crossrec = reconstruction[1][0][0,0].detach() * photoflux_std + photoflux_mean
    #thisband_photoonly = photo_only_recon[0,0] * photoflux_std + photoflux_mean
    #thisband_crossmodel = spec_encode_photo_decode[0,0] * photoflux_std + photoflux_mean



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

    #axs[3].plot(thisband_time, thisband_photoonly)
    #axs[3].scatter(thisband_time, thisband_photoonly, s=20, marker='x')

    #axs[4].plot(thisband_time, thisband_crossmodel)
    #axs[4].scatter(thisband_time, thisband_crossmodel, s=20, marker='x')

# invert y
axs[0].set_ylabel("AbsMag")
axs[1].set_xlabel("days")
ylow = -2 * photoflux_std + photoflux_mean
yhigh = 6 * photoflux_std + photoflux_mean

axs[0].set_ylim(ylow, yhigh)
axs[1].set_ylim(ylow, yhigh)
axs[2].set_ylim(ylow, yhigh)
#axs[3].set_ylim(ylow, yhigh)
#axs[4].set_ylim(ylow, yhigh)

axs[0].invert_yaxis()
axs[0].set_title("Ground truth")
axs[1].invert_yaxis()
axs[1].set_title("Reconstruction-LC")
axs[2].invert_yaxis()
axs[2].set_title("Reconstruction-Spectra")

#axs[3].invert_yaxis()
#axs[3].set_title("photometry only model")

#axs[4].invert_yaxis()
#axs[4].set_title("cross model")


#plt.tight_layout()

plt.show()
plt.savefig("ZTF_LC_reconstruction.pdf")
plt.close()


fig, axs = plt.subplots(2, 1, figsize=(10, 12))


axs[0].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])] * wavelength_std + wavelength_mean, 
         flux_test[idx][torch.logical_not(mask_test[idx])] * flux_std + flux_mean, 
         label='ground truth', color = "red")
axs[1].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])] * wavelength_std + wavelength_mean, 
         flux_test[idx][torch.logical_not(mask_test[idx])] * flux_std + flux_mean, color = "red")
#axs[2].plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
#         flux_test[idx] * flux_std + flux_mean, color = "red")



axs[0].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
         reconstruction[1][1].mean(axis = 0)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
         label='Rec-spec', color = "blue")

axs[0].fill_between(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean,
                    reconstruction[1][1].quantile(dim=0, q=0.05)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
                    reconstruction[1][1].quantile(dim=0, q=0.95)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean,
                    color = "blue", alpha=0.3)




axs[1].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
        reconstruction[0][1].mean(axis = 0)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
        label='Rec-LC', color = "green")
axs[1].fill_between(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean,
                    reconstruction[0][1].quantile(dim=0, q=0.05)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
                    reconstruction[0][1].quantile(dim=0, q=0.95)[0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean,
                    color = "green", alpha=0.3)

for i in range(30):
    axs[1].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
        reconstruction[0][1][i, 0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
        alpha = 0.3)

'''
axs[2].plot(wavelength_test[idx]* wavelength_std + wavelength_mean, 
        spectra_only_recon.mean(axis = 0)[0].detach().numpy() * flux_std + flux_mean, 
        label='spec-only', color = "orange")
axs[2].fill_between(wavelength_test[idx]* wavelength_std + wavelength_mean,
                    spectra_only_recon.quantile(dim=0, q=0.05)[0].detach().numpy() * flux_std + flux_mean, 
                    spectra_only_recon.quantile(dim=0, q=0.95)[0].detach().numpy() * flux_std + flux_mean,
                    color = "orange", alpha=0.3)
'''

'''
axs.plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
        spectra_only_recon[0].detach().numpy() * flux_std + flux_mean, 
        label='spec only')
axs.plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
         photo_encode_spec_decode[0,0].detach().numpy() * flux_std + flux_mean, 
         label='cross model')
'''



# invert y
axs[0].set_ylabel("log Fnu")
axs[1].set_ylabel("log Fnu")
#axs[2].set_ylabel("log Fnu")
axs[1].set_xlabel("wavelength (Å)")
#axs[0].set_ylim(-2* flux_std + flux_mean, 
#              2* flux_std + flux_mean)
#axs[1].set_ylim(-2* flux_std + flux_mean, 
#              2* flux_std + flux_mean)
#axs[2].set_ylim(-2* flux_std + flux_mean, 
#              2* flux_std + flux_mean)
axs[0].legend()
axs[1].legend()
#axs[2].legend()
#thisphase = int(phase_test[idx] * phase_std + phase_mean)
#axs[0].set_title(f"spectra at phase {thisphase}")


#plt.tight_layout()

plt.show()
plt.savefig("ZTF_spectra_reconstruction.pdf")
plt.close()



N = 30
generation = trained_vae.generate(x = data, N = N)
#breakpoint()

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for i in range(N):
    axs[0].plot(wavelength_test[i][torch.logical_not(mask_test[i])]* wavelength_std + wavelength_mean, 
         flux_test[i][torch.logical_not(mask_test[i])].detach().numpy() * flux_std + flux_mean, 
         label='Spec-gt', alpha = 0.5)


    axs[1].plot(wavelength_test[idx][torch.logical_not(mask_test[idx])]* wavelength_std + wavelength_mean, 
         generation[1][i,0][torch.logical_not(mask_test[idx])].detach().numpy() * flux_std + flux_mean, 
         label='Spec-samples', alpha = 0.5)
for i in range(2):
    axs[i].set_ylabel("log Fnu")
    axs[i].set_xlabel("wavelength (Å)")
    axs[i].set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
plt.show()
plt.savefig("ZTF_spectra_priorsamples.pdf")
plt.close()

print( phase_test[idx] * phase_std + phase_mean)