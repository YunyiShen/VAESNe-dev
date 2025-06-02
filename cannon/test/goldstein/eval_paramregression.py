import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from tqdm import tqdm

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.regression import VAEregressionHead
from VAESNe.data_util import get_goldstein_params, multimodalDataset


########## dealing with data ############
data = np.load('../../../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
testing_idx = data['testing_idx']
photoflux, phototime, photomask = data['photoflux'][testing_idx,:], data['phototime'][testing_idx,:], data['photomask'][testing_idx,:]
photoband = data['photowavelength'][testing_idx,:]

goldstein = data['identity'][testing_idx]
goldstein = np.array([get_goldstein_params(idd) for idd in goldstein ])



photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)
goldstein = torch.tensor(goldstein, dtype=torch.float32)

normalization = torch.load('../../../ckpt/goldstein_normalizing.pt', map_location="cpu")

mean = normalization['mean']
std = normalization['std']

lc_dataset = TensorDataset(photoflux, phototime, photoband, photomask)
goldsteinparam = TensorDataset(goldstein)

dataloader = DataLoader(multimodalDataset(lc_dataset, goldsteinparam), 
                batch_size=128, shuffle=False)

mmvae = torch.load("../../../ckpt/goldstein_photospectravaesne_4-4_0.00025_200_K2_beta1.0_modeldim32_LC2goldstein_0.001_500_128_4.pth", 
    map_location="cpu", weights_only = False
)

contrast = torch.load("../../../ckpt/goldstein_photospectra_contrast_4-4_0.00025_500_LC2goldstein_0.001_500_128_4.pth", 
    map_location="cpu", weights_only = False
)

end2end = torch.load("../../../ckpt/photometryend2end_4-4_modeldim32_LC2goldstein_0.001_500.pth",
    map_location="cpu", weights_only = False
)

# results to save
mmvae_absdiff = []
contrast_absdiff = []
end2end_absdiff = []

for x, label in tqdm(dataloader): 
    mmvae_resi = (mmvae(x) * std + mean - label[0])/std
    contrast_resi = (contrast(x) * std + mean - label[0])/std 
    end2end_resi = (end2end(x) * std + mean - label[0])/std

    mmvae_absdiff.append(np.abs(mmvae_resi.detach().cpu().numpy()))
    contrast_absdiff.append(np.abs(contrast_resi.detach().cpu().numpy()))
    end2end_absdiff.append(np.abs(end2end_resi.detach().cpu().numpy()))
    #breakpoint()


mmvae_absdiff = np.concatenate(mmvae_absdiff, axis = 0)
contrast_absdiff = np.concatenate(contrast_absdiff, axis = 0)
end2end_absdiff = np.concatenate(end2end_absdiff)


mmvae_absdiff_mean, mmvae_absdiff_std = np.mean(mmvae_absdiff, axis = 0), np.std(mmvae_absdiff, axis = 0)

contrast_absdiff_mean, contrast_absdiff_std = np.mean(contrast_absdiff, axis = 0), np.std(contrast_absdiff, axis = 0)

end2end_absdiff_mean, end2end_absdiff_std = np.mean(end2end_absdiff, axis = 0), np.std(end2end_absdiff, axis = 0)

np.savez("avg_absdiff_LC2goldstein_param.npz",
mmvae_mean = mmvae_absdiff_mean, 
mmvae_std = mmvae_absdiff_std,
contrast_mean = contrast_absdiff_mean, 
contrast_std = contrast_absdiff_std,
end2end_mean = end2end_absdiff_mean, 
end2end_std = end2end_absdiff_std
)

