import torch
import numpy as np
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import glob
from VAESNe.ImageVAE import HostImgVAE
from VAESNe.data_util import ImagePathDataset, ImagePathDatasetAug

# load data
png_files = np.array(glob.glob("../data/ZTFBTS/hostImgs/*.png"))
n_imgs = len(png_files)
n_train = int(n_imgs * 0.8)
training_list = png_files[:n_train]
testing_list = png_files[n_train:]

testing_data = ImagePathDataset(training_list.tolist())

trained_vae = torch.load("../ckpt/first_hostimgvaesne_4-4_0.001_50_patch2_beta0.5_modeldim32_numlayers4_hybridTrue.pth",
                         map_location=torch.device('cpu'), weights_only = False)


#### reconstruction ####
idxs = np.random.randint(0, 100, 10)


fig, axes = plt.subplots(2, 10, figsize=(20, 6))  # 4 rows, 5 columns
for i, idx in enumerate(idxs):
    original = testing_data[idx]
    original_img = original[0]
    original = original_img.unsqueeze(0)
    original = (original, torch.tensor([]))
    #breakpoint()
    recon = trained_vae.reconstruct(original)[0,0]
    axes[0,i].imshow(original_img.permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)
    axes[1,i].imshow(recon.permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)

plt.tight_layout()
plt.show()
plt.savefig("./hostimg_recon.pdf")
plt.close()



#### prior samples ####
images = trained_vae.generate(50)

fig, axes = plt.subplots(5, 10, figsize=(10, 8))  # 4 rows, 5 columns
axes = axes.flatten()

for i in range(50):
    img = images[i].permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC for matplotlib
    axes[i].imshow(img/2 + 0.5)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
plt.savefig("./hostimg64.pdf")
plt.close()