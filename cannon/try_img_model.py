import torch
import numpy as np
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from VAESNe.ImageVAE import HostImgVAE

trained_vae = torch.load("../ckpt/first_hostimgvaesne_4-2_0.00025_500.pth",
                         map_location=torch.device('cpu'), weights_only = False)


images = trained_vae.generate(50)

fig, axes = plt.subplots(5, 10, figsize=(10, 8))  # 4 rows, 5 columns
axes = axes.flatten()

for i in range(50):
    img = images[i].permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC for matplotlib
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
plt.savefig("./hostimg.pdf")
plt.close()