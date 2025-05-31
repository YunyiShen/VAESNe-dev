import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm
import argparse

from VAESNe.mmVAE import photospecMMVAE

def main():
    parser = argparse.ArgumentParser(description="test setups")
    parser.add_argument('--jobid', type = int, default = 0, help = 'jobid for parallelism')
    parser.add_argument('--totaljobs', type = int, default = 1, help = 'total number of jobs for parallelism')
    parser.add_argument('--latlen', type = int, default = 2, help = 'which model, 2 or 4')
    
    args = parser.parse_args()
    totaljobs = args.totaljobs
    jobid = args.jobid
    latlen = args.latlen



    data = np.load('../../../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
    training_idx = data['training_idx']
    testing_idx = data['testing_idx']

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


    trained_vae = torch.load(f"../../../ckpt/goldstein_photospectravaesne_4-{latlen}_0.00025_200_K2_beta1.0_modeldim32.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False)


    trained_vae.eval()


    ########### test on one test data ###########

    missing_portion = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    initidx = jobid
    idx_to_test = [initidx + i for i in range(5)] #[15, 16, 17, 18, 19]
    this_mask = []
    # should be from the same event
    this_photo_flux = photoflux_test[initidx]
    this_photo_time = phototime_test[initidx]
    this_photo_mask = photomask_test[initidx]
    this_photo_band = photoband_test[initidx]
    spect_res = [[] for _ in range(len(missing_portion))]
    spect_masks = [[] for _ in range(len(missing_portion))]
    spect_phase = [[] for _ in range(len(missing_portion))]
    spect_gt =  [[] for _ in range(len(missing_portion))]


    torch.manual_seed(42)

    for i, missing in tqdm(enumerate(missing_portion)):
        current_mask = this_photo_mask
        observed = ~current_mask
        random_flip = torch.rand_like(current_mask, dtype=torch.float) < missing
        flip_mask = observed & random_flip
        current_mask = this_photo_mask | flip_mask
        this_mask.append(current_mask)
        for idx in tqdm(idx_to_test):
            data = [
                ### photometry
                (this_photo_flux[None,:],
                this_photo_time[None,:],
                this_photo_band[None,:],
                current_mask[None,:]), 
                ### spectra
                (flux_test[idx][None,:],
                wavelength_test[idx][None,:],
                phase_test[idx][None], #* 0 + torch.tensor((40-phase_mean)/phase_std, dtype = torch.float32),
                mask_test[idx][None,:])
                ]


        #breakpoint()
            with torch.no_grad():
                reconstruction = trained_vae.reconstruct(data, K=100) # [0][0] LC->LC, [1][0]: spec->LC, [0][1]: LC-> Spec, [1][1]: spec-> spec
            spect = reconstruction[0][1][:,0].detach().cpu().numpy() * flux_std + flux_mean 
            spect_res[i].append(spect) # order should be phase
            spect_masks[i].append(mask_test[idx].detach().cpu().numpy())
            spect_phase[i].append(phase_test[idx].detach().cpu().numpy() * phase_std + phase_mean)
            spect_gt[i].append(flux_test[idx].detach().cpu().numpy() * flux_std + flux_mean)

    np.savez(f"./more_masking/maskingLC_more_4{latlen}_seed42_inid{initidx}.npz",
        missing_portion = missing_portion,
        LCmasks = this_mask,
        photo_flux = this_photo_flux.detach().cpu().numpy() * photoflux_std + photoflux_mean,
        photo_band = this_photo_band.detach().cpu().numpy(),
        photo_time = this_photo_time.detach().cpu().numpy() * phototime_std + phototime_std,

        # spectra
        spectra_gt = spect_gt,
        spectra = spect_res,
        spectra_masks = spect_masks,
        spectra_phase = spect_phase,
        wavelength = wavelength_test[idx].detach().cpu().numpy() * wavelength_std + wavelength_mean
        )

if __name__ == '__main__': 
    main()