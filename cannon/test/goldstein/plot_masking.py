import numpy as np
from matplotlib import pyplot as plt
from VAESNe.plot_util import plot_lsst_lc, plot_spectra_samples

results = np.load("./more_masking/maskingLC_more_44_seed42_inid0.npz")
missing_portion = results['missing_portion']
LCmasks = results['LCmasks']
photo_flux = results['photo_flux']
photo_band = results['photo_band']
photo_time = results['photo_time']

spectra_gt = results['spectra_gt']
spectra = results['spectra']
spectra_masks = results['spectra_masks']
spectra_phase = results['spectra_phase']
wavelength = results['wavelength']
fig, axs = plt.subplots(len(missing_portion), 6, figsize=(25 * 2, 15 * 2))
# draw LC first
for i in range(len(missing_portion)):
    plot_lsst_lc(photo_band, photo_flux, photo_time, LCmasks[i], ax = axs[i, 0], label = i==0)

axs[0,0].legend()
axs[i,0].set_xlabel("days")

for i in range(len(missing_portion)): # missing 
    for j in range(len(spectra[0])): # phase
        plot_spectra_samples(spectra_gt[i,j][None, :], wavelength, spectra_masks[i,j], ax = axs[i,j + 1], label = "ground truth", color = "red")
        plot_spectra_samples(spectra[i,j], wavelength, spectra_masks[i,j], ax = axs[i,j + 1], label = "VAESNe")
        axs[i,j + 1].set_ylim(-13.5, -11.4)
        

plt.tight_layout()
plt.show()
plt.savefig("./figs/masking.pdf")


