import numpy as np
import math
import matplotlib.pyplot as plt

import pdb

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

traj = np.load("/mnt/data/well_C1_full.npy", allow_pickle=True)
lineage = traj[1]

for _, single_trace in lineage.items():
    v_r = []
    v_theta = []

    try:
        for idx in np.arange(0, len(single_trace)):
                v_r.append(math.dist(single_trace[0], single_trace[idx]))
                v_theta.append(math.atan2(
                    single_trace[idx][0]-single_trace[0][0],
                    single_trace[idx][1]-single_trace[0][1]
                ))
    except Exception:
        pdb.set_trace()

    ax.plot(v_theta, v_r)
plt.savefig("/mnt/data/polar_coordinates_01.png", bbox_inches='tight')
