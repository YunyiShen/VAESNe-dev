import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.regression import contrasspecregressionHead
from VAESNe.data_util import get_goldstein_params, multimodalDataset


########## dealing with data ############
data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']
flux, wavelength, mask = data['flux'][training_idx,:], data['wavelength'][training_idx,:], data['mask'][training_idx,:]
phase = data['phase'][training_idx]


goldstein = data['identity'][training_idx]
goldstein = np.array([get_goldstein_params(idd) for idd in goldstein ])

flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

goldstein = torch.tensor(goldstein, dtype=torch.float32)


mean = goldstein.mean(dim=0)
std = goldstein.std(dim=0)

goldstein = (goldstein - mean)/std 
torch.save({'mean': mean, 'std': std}, '../ckpt/goldstein_normalizing.pt')

spec_dataset = TensorDataset(flux, wavelength, phase, mask)
goldsteinparam = TensorDataset(goldstein)

dataloader = DataLoader(multimodalDataset(spec_dataset, goldsteinparam), batch_size=32, shuffle=True)


########## model ############
backbone = "goldstein_photospectra_contrast_4-4_0.00025_500"

trained_contrast = torch.load(f"../ckpt/{backbone}.pth", # trained with K=1 on iwae
                         map_location=device, weights_only = False)

regrehead = contrasspecregressionHead(trained_contrast, 
            outdim = goldstein.shape[1], 
            MLPlatent = [128, 128, 128, 128]).to(device)
regrehead.train()

##### optimizer ####
lr = 1e-3
epochs = 500

optimizer = AdamW(regrehead.parameters(), lr=lr)
loss_fun = torch.nn.MSELoss()

from tqdm import tqdm
progress_bar = tqdm(range(epochs))
for i in progress_bar:
    device = next(regrehead.parameters()).device
    total_loss = 0
    num_batches = 0.
    for x, label in dataloader: # flux, time, band, mask for photometry and flux, wavelength, phase, mask for spectra
        optimizer.zero_grad()
        label = label[0].to(device)
        x = tuple(x_.to(device) for x_ in x)
        #breakpoint()
        pred = regrehead(x)
        loss = loss_fun(pred, label)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        num_batches += 1.
    avg_loss = total_loss / num_batches
    progress_bar.set_postfix(loss=f"epochs:{i}, {avg_loss:.4f}")
    if (i + 1) % 5 == 0:
        torch.save(regrehead, f'../ckpt/{backbone}_spec2goldstein_{lr}_{epochs}_128_4.pth')

