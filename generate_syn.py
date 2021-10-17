import numpy as np
from numpy.random.mtrand import rand
from tifffile import imsave
from skimage.morphology import disk, dilation
from skimage.util import random_noise
from random import randint
from scipy.ndimage import gaussian_filter


out_path = "/mnt/data/syn/"
num_img = 256
num_obj = 10
for ii in range(num_img):
    im = np.zeros((1280, 800))
    for obj in range(num_obj):
        py = randint(100, 1200)
        px = randint(50, 750)
        im[py, px] = 1
    im = dilation(im>0, disk(10))

    gt = im.astype(np.uint8)
    gt[gt > 0] = 1
    imsave(out_path + f"img_{ii+1}_GT.tiff", gt)

    raw = gaussian_filter(im.astype(np.float32), 3)
    raw = random_noise(raw).astype(np.float32)
    imsave(out_path + f"img_{ii+1}_IM.tiff", raw)
