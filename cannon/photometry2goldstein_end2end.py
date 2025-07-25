import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.regression import photoend2endregression
from VAESNe.data_util import get_goldstein_params, multimodalDataset


########## dealing with data ############
data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']
photoflux, phototime, photomask = data['photoflux'][training_idx,:], data['phototime'][training_idx,:], data['photomask'][training_idx,:]
photoband = data['photowavelength'][training_idx,:]

goldstein = data['identity'][training_idx]
goldstein = np.array([get_goldstein_params(idd) for idd in goldstein ])



photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)
goldstein = torch.tensor(goldstein, dtype=torch.float32)


mean = goldstein.mean(dim=0)
std = goldstein.std(dim=0)

goldstein = (goldstein - mean)/std 
torch.save({'mean': mean, 'std': std}, '../ckpt/goldstein_normalizing.pt')

lc_dataset = TensorDataset(photoflux, phototime, photoband, photomask)
goldsteinparam = TensorDataset(goldstein)

dataloader = DataLoader(multimodalDataset(lc_dataset, goldsteinparam), batch_size=32, shuffle=True)


########## model ############
lr = 2.5e-4
epochs = 200
latent_len = 4
latent_dim = 4
model_dim = 32

regrehead = photoend2endregression(

                outdim = goldstein.shape[1],
                num_bands = 6, 
                 latent_len = latent_len,
                 latent_dim = latent_dim,
                 model_dim = model_dim, 
                 num_heads = 4, 
                 ff_dim = model_dim,
                 num_layers = 4,
                 dropout=0.1,
                 selfattn=False,
).to(device)
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
        torch.save(regrehead, f'../ckpt/photometryend2end_{latent_len}-{latent_dim}_modeldim{model_dim}_LC2goldstein_{lr}_{epochs}.pth')

