import os
import numpy as np
from glob import glob
from scipy import ndimage
from tifffile import imread, imsave
from skimage.morphology import remove_small_objects
from skimage.draw import line
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import pdb

# params
max_matching_dist = 45
approx_inf = 65535
track_display_legnth = 20
min_obj_size = 20
bw_th = -4

# define binarization function
def prepare_binary(fn):
    # generate binary segmentaiton result
    seg = np.squeeze(imread(fn)) > bw_th
    seg = remove_small_objects(seg>0, min_size=min_obj_size)
    return seg

parent_path = "/mnt/data/"
all_movies = glob(parent_path + "timelapse/*.tiff")
shape_list = []
well_name_list = []
for M_idx, movies in enumerate(all_movies):
    movie_basename = os.path.basename(movies)
    well_name = movie_basename[:-5]
    well_name_list.append(well_name)

    seg_path = f"{parent_path}timelapse_seg/{well_name}/"
    # vis_path = f"{parent_path}timelapse_track/{well_name}"
    # os.makedirs(vis_path, exist_ok=True)
    raw_path = f"{parent_path}timelapse/{well_name}"
    track_result = f"{parent_path}timelapse_track/{well_name}_result.npy"


    total_time = len(glob(raw_path + "/*.tiff"))
    s_movie = []
    for tt in [0, total_time-1]:
        seg_fn = seg_path + f"img_{tt}_segmentation.tiff"

        seg = prepare_binary(seg_fn)

        # get label image
        seg_label, num_cells = ndimage.label(seg)
        s= regionprops_table(seg_label, properties=['area'])
        s_movie.append(s['area'])

    s_movie = [item for items in s_movie for item in items]
    shape_list.append(np.array(s_movie))

# pdb.set_trace()    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.violinplot(shape_list, showextrema=False, showmeans=True)
ax.set_ylabel('cell size in pixel')
ax.set_xticks(np.arange(1,1+len(well_name_list)))
ax.set_xticklabels(well_name_list, rotation=45)

# plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.1)
plt.ylim(0, 1250)
#plt.tight_layout()
plt.savefig("shape.png", bbox_inches = 'tight')
