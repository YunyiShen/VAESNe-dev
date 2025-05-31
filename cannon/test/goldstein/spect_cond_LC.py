import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

from VAESNe.mmVAE import photospecMMVAE


def split_indices(N, num_parts):
    chunk_size = N // num_parts
    remainder = N % num_parts
    partitions = []

    start = 0
    for i in range(num_parts):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        partitions.append((start, end))
        start = end

    return partitions


def main():

    parser = argparse.ArgumentParser(description="test setups")
    parser.add_argument('--jobid', type = int, default = 0, help = 'jobid for parallelism')
    parser.add_argument('--totaljobs', type = int, default = 1, help = 'total number of jobs for parallelism')
    args = parser.parse_args()
    totaljobs = args.totaljobs
    jobid = args.jobid

    data = np.load('../../../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
    testing_idx = data['testing_idx']

    n_test = len(testing_idx)
    partitions = split_indices(n_test, totaljobs)
    start_idx, end_idx = partitions[jobid]
    #breakpoint()

    flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
    phase_test = data['phase'][testing_idx]

    photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
    photoband_test = data['photowavelength'][testing_idx]

    identity_test = data['identity'][testing_idx]

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

    ####### load model ########
    trained_vae = torch.load("../../../ckpt/goldstein_photospectravaesne_4-2_0.00025_200_K2_beta1.0_modeldim32.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False)

    photo_only = torch.load("../../../ckpt/goldstein_photovaesne_4-2_0.00025_200.pth",
                         map_location=torch.device('cpu'), weights_only = False)

    spectra_only = torch.load('../../../ckpt/goldstein_specvaesne_4-2_0.00025_200.pth',
                         map_location=torch.device('cpu'), weights_only = False)

    trained_vae.eval()
    photo_only.eval()
    spectra_only.eval()


    data = [
    ### photometry
        (photoflux_test[start_idx:end_idx],
        phototime_test[start_idx:end_idx],
        photoband_test[start_idx:end_idx],
        photomask_test[start_idx:end_idx]), 
        ### spectra
        (flux_test[start_idx:end_idx],
        wavelength_test[start_idx:end_idx],
        phase_test[start_idx:end_idx], #* 0 + torch.tensor((40-phase_mean)/phase_std, dtype = torch.float32),
        mask_test[start_idx:end_idx])
    ]
    #breakpoint()
    with torch.no_grad():
        reconstruction = trained_vae.reconstruct(data, K=100) # [0][0] LC->LC, [1][0]: spec->LC, [0][1]: LC-> Spec, [1][1]: spec-> spec
        LConly = photo_only.reconstruct(data[0], K=100)
        speconly = spectra_only.reconstruct(data[1], K=100)

        lc_encode = trained_vae.vaes[0].encode(data[0], mean = True)
        spec_encode = trained_vae.vaes[1].encode(data[1], mean = True)
        

    

    #breakpoint()
    np.savez(f"./res/photospec_test_{jobid}_{totaljobs}.npz",
        ### LC data
        photoflux = photoflux_test[start_idx:end_idx].detach().cpu().numpy() * photoflux_std + photoflux_mean ,
        phototime = phototime_test[start_idx:end_idx].detach().cpu().numpy() * phototime_std + phototime_mean,
        photoband = photoband_test[start_idx:end_idx].detach().cpu().numpy(),
        photomask = photomask_test[start_idx:end_idx].detach().cpu().numpy(), 
        ### spectra
        flux = flux_test[start_idx:end_idx].detach().cpu().numpy() * flux_std + flux_mean,
        wavelength = wavelength_test[start_idx:end_idx].detach().cpu().numpy() * wavelength_std + wavelength_mean,
        phase = phase_test[start_idx:end_idx].detach().cpu().numpy() * phase_std + phase_mean, 
        mask = mask_test[start_idx:end_idx].detach().cpu().numpy(),
        identity = identity_test[start_idx:end_idx],

        ### results from mmvae
        LC2LC = reconstruction[0][0].detach().cpu().numpy() * photoflux_std + photoflux_mean,
        spec2LC = reconstruction[1][0].detach().cpu().numpy() * photoflux_std + photoflux_mean,
        LC2spec = reconstruction[0][1].detach().cpu().numpy() * flux_std + flux_mean,
        spec2spec = reconstruction[1][1].detach().cpu().numpy() * flux_std + flux_mean,
        LCencode = lc_encode,
        specencode = spec_encode,
        ### results from vae
        LConly = LConly.detach().cpu().numpy() * photoflux_std + photoflux_mean,
        speconly = speconly.detach().cpu().numpy() * flux_std + flux_mean
    )


if __name__ == '__main__': 
    main()




