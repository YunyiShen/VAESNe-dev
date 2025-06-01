import numpy as np
from tqdm import tqdm

def get_metric(spectra, gt, alpha_level = 0.1):
    spectra_mean = np.nanmean(spectra, axis = 0)
    spectra_lw = np.nanquantile(spectra, q = alpha_level/2, axis = 0)
    spectra_hi = np.nanquantile(spectra, q = 1. - alpha_level/2, axis = 0) 

    residule = gt - spectra_mean   
    cover = np.logical_and((gt - spectra_lw)>0, (spectra_hi - gt)>0)
    width = spectra_hi - spectra_lw

    return residule, cover, width


def aggr_phase(resi, cover, width, phase):
    phases = [-10., 0., 10., 20., 30.]
    resi_mean = []
    resi_sd = []
    cover_mean = []
    width_mean = []
    width_sd = []
    mse = []
    for phase_i in phases:
        resi_ = resi[phase == phase_i,:]
        cover_ = cover[phase == phase_i,:]
        width_ = width[phase == phase_i,:]

        resi_mean.append(np.nanmean(resi_, 0))
        resi_sd.append(np.nanstd(resi_, 0))
        cover_mean.append(np.nanmean(1. * cover_, 0))
        width_mean.append(np.nanmean(width, 0))
        width_sd.append(np.nanstd(width, 0))
        
        mse.append(np.nanmean(resi_ ** 2))
    
    return resi_mean, resi_sd, cover_mean, width_mean, width_sd, mse


mm_resi = []
mm_coverage = []
mm_width = []

speconly_resi = []
speconly_coverage = []
speconly_width = []

phase = []

all_ids = np.arange(0, 400)

for jobid in tqdm(all_ids):
    res = np.load(f"./res/photospec44_test_{jobid}_400.npz")
    phase.append( np.round(res['phase']))
    residule, cover, width = get_metric(res['LC2spec'], res['flux'])
    mm_resi.append(residule)
    mm_coverage.append(cover)
    mm_width.append(width)

    # spect only
    residule, cover, width = get_metric(res['speconly'], res['flux'])
    speconly_resi.append(residule)
    speconly_coverage.append(cover)
    speconly_width.append(width)

mm_resi = np.concat(mm_resi, axis = 0)
mm_coverage = np.concat(mm_coverage, axis = 0)
mm_width = np.concat(mm_width, axis = 0)

speconly_resi = np.concat(speconly_resi, axis = 0)
speconly_coverage = np.concat(speconly_coverage, axis = 0)
speconly_width = np.concat(speconly_width, axis = 0)

phase = np.concat(phase, axis = 0)


mmresi_mean, mmresi_sd, mmcover_mean, mmwidth_mean, mmwidth_sd, mmmse = aggr_phase(mm_resi, mm_coverage, mm_width, phase)

speconlyresi_mean, speconlyresi_sd, speconlycover_mean, speconlywidth_mean, speconlywidth_sd, speconlymse = aggr_phase(speconly_resi, speconly_coverage, speconly_width, phase)

np.savez("avg_metrics.npz",
    mm_resi_mean =mmresi_mean,
    mm_resi_sd = mmresi_sd,
    mm_coverage_mean = mmcover_mean,
    mm_width_mean = mmwidth_mean,
    mm_width_sd = mmwidth_sd,
    mm_mse = mmmse,

    speconly_resi_mean =speconlyresi_mean,
    speconly_resi_sd = speconlyresi_sd,
    speconly_coverage_mean = speconlycover_mean,
    speconly_width_mean = speconlywidth_mean,
    speconly_width_sd = speconlywidth_sd,
    speconly_mse = speconlymse,

    wavelength = res['wavelength']
)
