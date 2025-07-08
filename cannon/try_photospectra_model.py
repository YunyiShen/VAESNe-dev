import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from VAESNe.mmVAE import photospecMMVAE


# data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
data = np.load("/n/holystore01/LABS/iaifi_lab/Lab/specgen_shen_gagliano/generative-spectra-lightcurves/data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz")
training_idx = data['training_idx']
testing_idx = data['testing_idx']
# print length of testing_idx
print("Number of testing spectra:", len(testing_idx))

# Check if reconstructed spectra are sampled along the same wavelength grid as the training spectra
# w = data['wavelength']  # shape [num_spectra, N_waves]
# first = w[0]  # the first grid (row)
# all_same = np.allclose(w, first[None, :])  # check if every row matches the first
# print("All wavelength grids identical?", all_same)

flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase_test = data['phase'][testing_idx]

photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
photoband_test = data['photowavelength'][testing_idx]

flux_mean, flux_std = data['flux_mean'], data['flux_std']
wavelength_mean, wavelength_std = data['wavelength_mean'], data['wavelength_std']
phase_mean, phase_std = data['phase_mean'], data['phase_std']

phototime_mean, phototime_std = data['phototime_mean'], data['phototime_std']
photoflux_mean, photoflux_std = data['photoflux_mean'], data['photoflux_std']

flux_test = torch.tensor(flux_test, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength_test, dtype=torch.float32)
mask_test = torch.tensor(mask_test == 0)
phase_test = torch.tensor(phase_test, dtype=torch.float32)



photoflux_test = torch.tensor(photo_flux_test, dtype=torch.float32)
phototime_test = torch.tensor(phototime_test, dtype=torch.float32)
photomask_test = torch.tensor(photomask_test == 0)
photoband_test = torch.tensor(photoband_test, dtype=torch.long)


# trained_vae = torch.load("../ckpt/goldstein_brightphotospectravaesne_4-4_0.001_500_K8_beta1.0_modeldim32.pth", # trained with K=1 on iwae
#                          map_location=torch.device('cpu'), weights_only = False)
trained_vae = torch.load("../ckpt/goldstein_photospectravaesne_4-4_0.0001_200_K2_beta1.0_modeldim32_concatTrue.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False)

# photo_only = torch.load("../ckpt/first_photovaesne_4-2_0.00025_500.pth",
#                          map_location=torch.device('cpu'), weights_only = False)
photo_only = torch.load("../ckpt/goldstein_photovaesne_4-4_0.00025_200.pth",
                         map_location=torch.device('cpu'), weights_only = False)

# spectra_only = torch.load('../ckpt/first_specvaesne_4-2_0.00025_500.pth',
#                          map_location=torch.device('cpu'), weights_only = False)
spectra_only = torch.load('../ckpt/goldstein_specvaesne_4-4_0.00025_200_concatTrue.pth',
                         map_location=torch.device('cpu'), weights_only = False)

trained_vae.eval()
photo_only.eval()
spectra_only.eval()

os.makedirs("figures", exist_ok=True)


########### test on one test data ###########
idx = 17

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
with torch.no_grad():
    reconstruction = trained_vae.reconstruct(data, K=100) # [0][0] LC->LC, [1][0]: spec->LC, [0][1]: LC-> Spec, [1][1]: spec-> spec

    photo_only_recon = photo_only.reconstruct(data[0])
    spectra_only_recon = spectra_only.reconstruct(data[1], K=100)
    photo_encode = photo_only.encode(data[0])[None, ...]
    spectra_encoded = spectra_only.encode(data[1])[None, ...]
    photo_encode_spec_decode = spectra_only.decode(photo_encode, data[1]).mean
    spec_encode_photo_decode = photo_only.decode(spectra_encoded, data[0]).mean


fig, axs = plt.subplots(1, 5, figsize=(15, 5))

for i in range(6):
    thisband_gt = photoflux_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))] * photoflux_std + photoflux_mean
    thisband_time = phototime_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))] * phototime_std + phototime_mean

    #breakpoint()
    thisband_rec = reconstruction[0][0][0,0].detach() * photoflux_std + photoflux_mean
    thisband_crossrec = reconstruction[1][0][0,0].detach() * photoflux_std + photoflux_mean
    thisband_photoonly = photo_only_recon[0,0] * photoflux_std + photoflux_mean
    thisband_crossmodel = spec_encode_photo_decode[0,0] * photoflux_std + photoflux_mean



    thisband_rec = thisband_rec[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    thisband_crossrec = thisband_crossrec[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    
    thisband_photoonly = thisband_photoonly[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    
    thisband_crossmodel = thisband_crossmodel[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]

    
    axs[0].plot(thisband_time, thisband_gt)
    axs[0].scatter(thisband_time, thisband_gt, s=20, marker='o')

    axs[1].plot(thisband_time, thisband_rec)
    axs[1].scatter(thisband_time, thisband_rec, s=20, marker='x')

    axs[2].plot(thisband_time, thisband_crossrec)
    axs[2].scatter(thisband_time, thisband_crossrec, s=20, marker='x')

    axs[3].plot(thisband_time, thisband_photoonly)
    axs[3].scatter(thisband_time, thisband_photoonly, s=20, marker='x')

    axs[4].plot(thisband_time, thisband_crossmodel)
    axs[4].scatter(thisband_time, thisband_crossmodel, s=20, marker='x')

# invert y
axs[0].set_ylabel("AbsMag")
axs[2].set_xlabel("days")
ylow = -2 * photoflux_std + photoflux_mean
yhigh = 6 * photoflux_std + photoflux_mean

axs[0].set_ylim(ylow, yhigh)
axs[1].set_ylim(ylow, yhigh)
axs[2].set_ylim(ylow, yhigh)
axs[3].set_ylim(ylow, yhigh)
axs[4].set_ylim(ylow, yhigh)

axs[0].invert_yaxis()
axs[0].set_title("Ground truth")
axs[1].invert_yaxis()
axs[1].set_title("Reconstruction-LC")
axs[2].invert_yaxis()
axs[2].set_title("Reconstruction-Spectra")

axs[3].invert_yaxis()
axs[3].set_title("photometry only model")

axs[4].invert_yaxis()
axs[4].set_title("cross model")


#plt.tight_layout()

plt.show()
plt.savefig("figures/LC_reconstruction.png")
plt.close()


fig, axs = plt.subplots(3, 1, figsize=(10, 12))


axs[0].plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
         flux_test[idx] * flux_std + flux_mean, 
         label='ground truth', color = "red")
axs[1].plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
         flux_test[idx] * flux_std + flux_mean, color = "red")
axs[2].plot(wavelength_test[idx] * wavelength_std + wavelength_mean, 
         flux_test[idx] * flux_std + flux_mean, color = "red")



axs[0].plot(wavelength_test[idx]* wavelength_std + wavelength_mean, 
         reconstruction[1][1].mean(axis = 0)[0].detach().numpy() * flux_std + flux_mean, 
         label='Rec-spec', color = "blue")

axs[0].fill_between(wavelength_test[idx]* wavelength_std + wavelength_mean,
                    reconstruction[1][1].quantile(dim=0, q=0.025)[0].detach().numpy() * flux_std + flux_mean, 
                    reconstruction[1][1].quantile(dim=0, q=0.975)[0].detach().numpy() * flux_std + flux_mean,
                    color = "blue", alpha=0.3)




axs[1].plot(wavelength_test[idx]* wavelength_std + wavelength_mean, 
        reconstruction[0][1].mean(axis = 0)[0].detach().numpy() * flux_std + flux_mean, 
        label='Rec-LC', color = "green")
axs[1].fill_between(wavelength_test[idx]* wavelength_std + wavelength_mean,
                    reconstruction[0][1].quantile(dim=0, q=0.025)[0].detach().numpy() * flux_std + flux_mean, 
                    reconstruction[0][1].quantile(dim=0, q=0.975)[0].detach().numpy() * flux_std + flux_mean,
                    color = "green", alpha=0.3)


axs[2].plot(wavelength_test[idx]* wavelength_std + wavelength_mean, 
        spectra_only_recon.mean(axis = 0)[0].detach().numpy() * flux_std + flux_mean, 
        label='spec-only', color = "orange")
axs[2].fill_between(wavelength_test[idx]* wavelength_std + wavelength_mean,
                    spectra_only_recon.quantile(dim=0, q=0.025)[0].detach().numpy() * flux_std + flux_mean, 
                    spectra_only_recon.quantile(dim=0, q=0.975)[0].detach().numpy() * flux_std + flux_mean,
                    color = "orange", alpha=0.3)


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
axs[2].set_ylabel("log Fnu")
axs[2].set_xlabel("wavelength (Å)")
axs[0].set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
axs[1].set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
axs[2].set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
axs[0].legend()
axs[1].legend()
axs[2].legend()
thisphase = int(phase_test[idx] * phase_std + phase_mean)
axs[0].set_title(f"spectra at phase {thisphase}")


#plt.tight_layout()

plt.show()
plt.savefig("figures/spectra_reconstruction.png")
plt.close()



N = 30
generation = trained_vae.generate(x = data, N = N)
#breakpoint()

fig, axs = plt.subplots(1, 1, figsize=(10, 5))

for i in range(N):
    axs.plot(wavelength_test[idx]* wavelength_std + wavelength_mean, 
         generation[1][i,0].detach().numpy() * flux_std + flux_mean, 
         label='Spec-samples', alpha = 0.5)

axs.set_ylabel("log Fnu")
axs.set_xlabel("wavelength (Å)")
axs.set_ylim(-2* flux_std + flux_mean, 
              2* flux_std + flux_mean)
plt.show()
plt.savefig("figures/spectra_priorsamples.png")
plt.close()


# MSE vs. phase figures
trained_vae, photo_only, spectra_only = trained_vae.to(device), photo_only.to(device), spectra_only.to(device)
N_test, T_phot = photoflux_test.shape
_,      T_spec = flux_test.shape

# preallocate reconstruction arrays
rec_lc2lc   = np.zeros((N_test, T_phot), dtype=np.float32)
rec_lc2spec = np.zeros((N_test, T_spec), dtype=np.float32)
rec_spec2lc = np.zeros((N_test, T_phot), dtype=np.float32)
rec_spec2spec = np.zeros((N_test, T_spec), dtype=np.float32)

# split into batches to avoid CUDA out of memory
batch_size = 128
for i in range(0, N_test, batch_size):
    j = min(i + batch_size, N_test)
    sub_photo = (
        photoflux_test [i:j].to(device),
        phototime_test [i:j].to(device),
        photoband_test [i:j].to(device),
        photomask_test [i:j].to(device),
    )
    sub_spec = (
        flux_test      [i:j].to(device),
        wavelength_test[i:j].to(device),
        phase_test     [i:j].to(device),
        mask_test      [i:j].to(device),
    )
    with torch.no_grad():
        recon = trained_vae.reconstruct([sub_photo, sub_spec], K=1)
        lc2lc_batch   = recon[0][0].squeeze(0).cpu().numpy()   # [B, T_phot]
        lc2spec_batch = recon[0][1].squeeze(0).cpu().numpy()   # [B, T_spec]
        spec2lc_batch = recon[1][0].squeeze(0).cpu().numpy()   # [B, T_phot]
        spec2spec_batch = recon[1][1].squeeze(0).cpu().numpy() # [B, T_spec]
    rec_lc2lc  [i:j] = lc2lc_batch
    rec_lc2spec[i:j] = lc2spec_batch
    rec_spec2lc[i:j] = spec2lc_batch
    rec_spec2spec[i:j] = spec2spec_batch
    del recon, lc2lc_batch, lc2spec_batch, spec2lc_batch, spec2spec_batch
    torch.cuda.empty_cache()

# ground-truth arrays
pf_np   = photoflux_test.cpu().numpy()  # [N_test, T_phot]
flux_np = flux_test.cpu().numpy()       # [N_test, T_spec]

# un-normalize phase to get days since explosion
phase_days = (phase_test * phase_std + phase_mean).cpu().numpy()

# compute reconstruction MSE
err_lc2lc   = ((rec_lc2lc   - pf_np   )**2).mean(axis=1)
err_spec2spec = ((rec_spec2spec - flux_np)**2).mean(axis=1)
err_lc2spec = ((rec_lc2spec - flux_np)**2).mean(axis=1)
err_spec2lc = ((rec_spec2lc - pf_np   )**2).mean(axis=1)

# discretize phase_days to get exact integer days
disc_phases = np.round(phase_days).astype(int)
unique_phases = np.unique(disc_phases)
# print("Discrete phase values:", unique_phases)

mappings = [
    ("LC → LC",     err_lc2lc,     "blue"),
    ("Spec → Spec", err_spec2spec, "red"),
    ("LC → Spec",   err_lc2spec,   "green"),
    ("Spec → LC",   err_spec2lc,   "orange"),
]

for name, errs, color in mappings:
    # Compute per-phase averages
    avg_err = []
    for ph in unique_phases:
        mask = (disc_phases == ph)
        avg_err.append(errs[mask].mean() if mask.any() else np.nan)
    avg_err = np.array(avg_err)
    plt.figure(figsize=(6,4))
    # Scatter raw errors at each discrete phase
    for ph in unique_phases:
        mask = (disc_phases == ph)
        xs = np.full(mask.sum(), ph)  # all points exactly at ph
        plt.scatter(xs, errs[mask], alpha=0.3, color=color)
    # Plot average MSE for each phase
    plt.plot(unique_phases, avg_err, '-o', color='black', lw=2, label="Avg. MSE")
    plt.xlabel("Days since explosion")
    plt.ylabel("Reconstruction MSE")
    plt.ylim(0, 1)
    plt.title(f"MSE vs. Phase ({name})")
    plt.xticks(unique_phases)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"figures/error_vs_phase_{name.replace(' → ','_to_')}.png")
    plt.close()


# Plot worst 3 reconstructions for each type
k = 3

# Plot worst 3 LC reconstructions
# Filter band
code_to_props = {
    0: ("u", "purple"),
    1: ("g", "green"),
    2: ("r", "red"),
    3: ("i", "orange"),
    4: ("z", "blue"),
}

lc_combined = [
    ("LC → LC",   rec_lc2lc,   err_lc2lc),
    ("Spec → LC", rec_spec2lc, err_spec2lc),
]

for name, rec_array, err_array in lc_combined:
    worst_idx = np.argsort(err_array)[-k:][::-1]

    for sn in worst_idx:
        # de-normalize time & flux
        t_all   = phototime_test[sn].cpu().numpy() * phototime_std + phototime_mean
        gt_all  = pf_np[sn]     * photoflux_std + photoflux_mean
        rec_all = rec_array[sn] * photoflux_std + photoflux_mean
        fig, ax = plt.subplots(figsize=(8,5))

        for code, (band, color) in code_to_props.items():
            mask = (photoband_test[sn].cpu().numpy() == code) & (~photomask_test[sn].cpu().numpy())
            if not mask.any():
                continue
            t_band   = t_all[mask]
            gt_band  = gt_all[mask]
            rec_band = rec_all[mask]
            ax.plot(t_band, gt_band, 'o-', color=color)
            ax.plot(t_band, rec_band, 'x--', color=color)

        ax.set_xlabel("Days since explosion")
        ax.set_ylabel("AbsMag")
        ax.invert_yaxis()
        ax.set_title(f"{name} (worst idx = {sn})")
        # legend
        band_handles = [
            mpatches.Patch(color=c, label=b)
            for _, (b,c) in sorted(code_to_props.items())
        ]
        shape_handles = [
            mlines.Line2D([], [], marker='o', color='black', linestyle='-',  label='Ground truth'),
            mlines.Line2D([], [], marker='x', color='black', linestyle='--', label='Reconstruction'),
        ]
        leg1 = ax.legend(handles=band_handles, title="Filter band", loc='lower left')
        ax.add_artist(leg1)
        ax.legend(handles=shape_handles, loc='upper right')
        plt.tight_layout()
        name1 = name.replace(' → ','_to_').replace(' ', '')
        plt.savefig(f"figures/worst_{name1}_idx{sn}.png")
        plt.show()
        plt.close()


# Plot worst 3 spectra reconstructions
for name, errs, rec_array, gt_array in [
    ("Spec → Spec", err_spec2spec, rec_spec2spec, flux_np),
    ("LC → Spec",   err_lc2spec,   rec_lc2spec,   flux_np),
]:
    worst_idx = np.argsort(errs)[-k:][::-1]
    for sn in worst_idx:
        # de-normalize
        x_wave = wavelength_test[sn].cpu().numpy() * wavelength_std + wavelength_mean
        gt_spec = gt_array[sn] * flux_std + flux_mean
        rec_spec= rec_array[sn] * flux_std + flux_mean
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.plot(x_wave, gt_spec, '-',  label="GT Spec")
        ax.plot(x_wave, rec_spec, '--', label="Rec Spec")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("log Fν")
        ax.legend()
        ax.set_title(f"{name} (worst idx = {sn})")
        plt.tight_layout()
        plt.savefig(f"figures/worst_{name.replace(' → ','_to_')}_idx{sn}.png")
        plt.show()
        plt.close()
