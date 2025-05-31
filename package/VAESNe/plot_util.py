import matplotlib.pyplot as plt
import numpy as np
def plot_lsst_lc(photoband, photomag, phototime, photomask, ax = None, label = False):
    lsst_bands = ["u", "g", "r", "i", "z", "y"]
    colors = ["purple", "blue", "darkgreen", "lime", "orange", "red"]
    photoband = photoband[~photomask]
    photomag = photomag[~photomask]
    phototime = phototime[~photomask]
    if ax is None:
        fig, ax = plt.subplots()
    for bnd in range(6):
        idx = np.where(photoband == bnd)[0]
        if len(idx) > 0:
            if label:
                ax.scatter(phototime[idx], photomag[idx], label=lsst_bands[bnd], s=5, color=colors[bnd])
            else:
                ax.scatter(phototime[idx], photomag[idx], s=5, color=colors[bnd])
            ax.plot(phototime[idx], photomag[idx], color=colors[bnd], alpha=0.5)
    ax.invert_yaxis()
    if ax is None:
        return fig
    #return ax


def plot_spectra_samples(spectra, wavelength, mask, alpha_level = 0.1, ax = None, color = "blue", label = None):
    if ax is None:
        fig, ax = plt.subplots()
    spectra_mean = np.nanmean(spectra, axis = 0)
    spectra_lw = np.nanquantile(spectra, q = alpha_level/2, axis = 0)
    spectra_hi = np.nanquantile(spectra, q = 1. - alpha_level/2, axis = 0)

    ax.plot(wavelength[~mask], spectra_mean[~mask], label = label, color = color)

    ax.fill_between(wavelength[~mask],
                    spectra_lw[~mask], 
                    spectra_hi[~mask],
                    color = color, alpha=0.3)
    if ax is None:
        return fig
