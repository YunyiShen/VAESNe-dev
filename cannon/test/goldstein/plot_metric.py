import numpy as np
import matplotlib.pyplot as plt


metrics = np.load("avg_metrics.npz")

plt.rcParams['font.size'] = 30
fig, axes =  plt.subplots(3, 5, figsize=(30, 15), sharex=False)
fig.subplots_adjust(hspace=0)

phase = [-10,0,10,20,30]
wavelength = metrics['wavelength'][0]
#breakpoint()

########## residual ############
for i in range(5):
    mean_speconly = metrics['speconly_resi_mean'][i]
    sd_speconly = metrics['speconly_resi_sd'][i]
    if i == 0:
        axes[0, i].plot(wavelength, mean_speconly,color='green', label='spectra only VAE', linewidth=2)
    else:
        axes[0, i].plot(wavelength, mean_speconly,color='green', linewidth=2)
    axes[0, i].fill_between(wavelength, 
                              mean_speconly - sd_speconly, 
                              mean_speconly + sd_speconly,color='green', alpha=0.3)
    axes[0, i].set_title(f'Days after peak: {phase[i]}')
    axes[0, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0, i].tick_params(labelbottom=False)
    rangee = 0.4 #2.2 if postfix == "" else 1.75
    axes[0, i].set_ylim(-rangee, rangee)


    mean_mm = metrics['mm_resi_mean'][i]
    sd_mm = metrics['mm_resi_sd'][i]

    if i == 0:
        axes[0, i].plot(wavelength, mean_mm,color='blue', label='mmVAE LC2spec', linewidth=2)
    else:
        axes[0, i].plot(wavelength, mean_mm,color='blue', linewidth=2)
    axes[0, i].fill_between(wavelength, 
                              mean_mm - sd_mm, 
                              mean_mm + sd_mm, 
                              color='blue', alpha=0.3)

axes[0,0].set_ylabel('residual')
#fig.subplots_adjust(bottom=0.01) 
fig.subplots_adjust(left=0.03) 
axes[2,2].set_xlabel('Wavelength (Å)')

fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.2, 0.7), handlelength=1) 

############ coverage ############
for i in range(5):
    mean_mm_coverage = metrics['mm_coverage_mean'][i]
    mean_speconly_coverage = metrics['speconly_coverage_mean'][i]
    
    axes[1, i].plot(wavelength, mean_mm_coverage, color='blue')
    axes[1, i].plot(wavelength, mean_speconly_coverage, color='green')
    
    axes[1, i].axhline(0.9, color='red', linestyle='--', linewidth=1.5)
    axes[1, i].tick_params(labelbottom=False)
    axes[1, i].set_ylim(0.01,1.05)


    mean_mm_width = metrics['mm_width_mean'][i]
    sd_mm_width = metrics['mm_width_sd'][i]

    mean_speconly_width = metrics['speconly_width_mean'][i]
    sd_speconly_width = metrics['speconly_width_sd'][i]

    

    axes[2, i].plot(wavelength, mean_mm_width,
                    color='blue')
    axes[2, i].fill_between(wavelength, 
                              mean_mm_width - sd_mm_width, 
                              mean_mm_width + sd_mm_width, 
                              color='blue',
                              alpha=0.3)
    axes[2, i].plot(wavelength, mean_speconly_width,
                    color='green')
    axes[2, i].fill_between(wavelength, 
                              mean_speconly_width - sd_speconly_width, 
                              mean_speconly_width + sd_speconly_width, 
                              color='green',
                              alpha=0.3)
    
    axes[2, i].tick_params(labelbottom=True)
    axes[2, i].set_ylim(0,.4)

axes[1,0].set_ylabel('CI coverage')
axes[2,0].set_ylabel('CI width')
plt.subplots_adjust(wspace=0.1, hspace=0.1) 

plt.tight_layout(rect=[0.0, 0.0, 1., 1.])
plt.show()
plt.show()
plt.savefig("./figs/metrics.pdf")

print(metrics['mm_mse'])
print(metrics['speconly_mse'])
