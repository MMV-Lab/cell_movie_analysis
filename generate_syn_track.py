import numpy as np
from numpy.random.mtrand import rand
from tifffile import imsave
from skimage.morphology import disk, dilation
from skimage.util import random_noise
from random import randint
from scipy.ndimage import gaussian_filter


out_path = "/mnt/data/syn_seg/"
num_img = 8
num_obj = 10
im_list = []
for ii in range(num_img):
    im_list.append(np.zeros((800,1200)))

for obj in range(num_obj):
    px = randint(100, 1200)
    py = randint(100, 700)
    for ii in range(num_img):
        im_list[ii][py, px] = 1
        py = py + randint(-10, 10)
        px = px + randint(-10, 10)

for ii in range(num_img):
    gt = dilation(im_list[ii]>0, disk(10))
    gt = gt.astype(np.uint8)
    gt[gt > 0] = 1
    imsave(out_path + f"img_{ii+1}_seg.tiff", gt)
