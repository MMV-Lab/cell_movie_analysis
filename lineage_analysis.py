import numpy as np
import os
import math
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import pdb


def make_polar_plot(lineage, exclude_id, out_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')       

    for track_id, single_trace in lineage.items():
        v_r = []
        v_theta = []
        if track_id in exclude_id:
            continue
        for idx in np.arange(0, len(single_trace)):
            v_r.append(math.dist(single_trace[0], single_trace[idx]))
            v_theta.append(math.atan2(
                single_trace[idx][0]-single_trace[0][0],
                single_trace[idx][1]-single_trace[0][1]
            ))
        ax.plot(v_theta, v_r)
    plt.savefig(out_name)

short_track_cutoff = 5
time_step_in_min = 4
px_size_in_micron = 0.46

filenames = glob("/mnt/data/timelapse_track/well_*_result.npy")
filenames.sort()
well_name = []
mean_v_plot = []
peak_v_plot = []
std_v_plot = []
ang_v_plot = []
for fn in filenames:
    traj = np.load(fn, allow_pickle=True)
    lineage = traj[1]

    track_stat = []
    ignore_track = []
    for track_id, single_trace in lineage.items():
        # exclude short tracks
        if len(single_trace) < short_track_cutoff:
            ignore_track.append(track_id)
            continue

        # calculate mean velocity and angular velocity
        p_prev = single_trace[:-1]
        p_next = single_trace[1:]
        p_displacement = [np.subtract(x, y) for x, y in zip(p_next, p_prev)]
        p_velocity = [np.sqrt(p[0]**2 + p[1]**2) * px_size_in_micron / time_step_in_min for p in p_displacement]
        mean_velocity = np.mean(p_velocity)
        p90_velocity = np.percentile(p_velocity, 90)
        peak_velocity = np.max(p_velocity)
        std_velocity = np.std(p_velocity)

        p_rad = [math.atan2(p[0], p[1]) / time_step_in_min for p in p_displacement]
        p_rad_prev = p_rad[:-1]
        p_rad_next = p_rad[1:]
        p_ang_rad = [np.subtract(x, y) for x, y in zip(p_rad_next, p_rad_prev)]
        p_ang_deg = [math.degrees(p) for p in p_ang_rad]
        mean_ang_velocity = np.mean(p_ang_deg)

        # exclude erythrocyte
        if mean_velocity > 10 * px_size_in_micron / time_step_in_min:
        #and (abs(mean_ang_velocity)<5 or abs(mean_ang_velocity)>175):
            ignore_track.append(track_id)
            continue

        track_stat.append({
            "id": track_id,
            "mean_v": mean_velocity,
            "p90_v": p90_velocity,
            "peak_v": peak_velocity,
            "std_v": std_velocity,
            "mean_ang": mean_ang_velocity
        })

    track_stat_df = pd.DataFrame(track_stat)

    this_well = os.path.basename(fn)[:7]
    mean_v_plot.append(track_stat_df["mean_v"].values)
    peak_v_plot.append(track_stat_df["peak_v"].values)
    std_v_plot.append(track_stat_df["std_v"].values)
    ang_v_plot.append(track_stat_df["mean_ang"].values)
    well_name.append(this_well)

    # make_polar_plot(lineage, ignore_track, f"{this_well}_polar.png")
"""
# bubble plot
tmp_idx = 0
dim_row = 4
dim_col = 3
pt_color = []
pt_size = []
pt_x = []
pt_y = []
for row_idx in range(dim_row):
    for col_idx in range(dim_col):
        pt_color.append(mean_v_plot[tmp_idx].mean())
        pt_size.append(mean_v_plot[tmp_idx].std())
        pt_x.append(row_idx)
        pt_y.append(col_idx)
        tmp_idx += 1
        if tmp_idx >= len(mean_v_plot):
            break

fig = plt.figure()
sc = plt.scatter(np.array(pt_x), np.array(pt_y), s=15000*np.array(pt_size), c=100*np.array(pt_color), alpha=0.5)
f = plt.colorbar(sc, orientation='horizontal', shrink=0.75)
f.ax.tick_params(size=0)
f.ax.get_xaxis().set_visible(False)
ax = plt.gca()
#ax.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)
ax.axis('off')
plt.xlim(-1, dim_row)
plt.ylim(-1, dim_col)
plt.savefig("bubble.png")
"""

"""
# pcolor plot
dim_row = 4
dim_col = 3
color_ar = np.zeros((dim_row, dim_col))
tmp_idx=0
for row_idx in range(dim_row):
    for col_idx in range(dim_col):
        color_ar[row_idx, col_idx] = mean_v_plot[tmp_idx].mean()
        tmp_idx += 1
        if tmp_idx >= len(mean_v_plot):
            break

color_ar[color_ar==0] = np.min(color_ar[color_ar>0])

fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.pcolor(color_ar, edgecolors='k', linewidths=4)
f = plt.colorbar(c, orientation='horizontal', shrink=0.95)
f.ax.tick_params(size=0)
f.ax.get_xaxis().set_visible(False)
ax = plt.gca()
#ax.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)
ax.axis('off')
#plt.xlim(-1, dim_row)
#plt.ylim(-1, dim_col)
plt.savefig("pcolor.png")

"""

"""
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Sumary of cell motion analysis in sample movies', fontsize=12)
plt.style.use('seaborn')

ax = fig.add_subplot(411)
ax.violinplot(mean_v_plot, showextrema=False, showmeans=True)
ax.set_ylabel('mean velocity (um/sec)', rotation=0, ha='right')
ax.axes.get_xaxis().set_visible(False)

ax = fig.add_subplot(412)
ax.violinplot(peak_v_plot, showextrema=False, showmeans=True)
ax.set_ylabel('peak_v_plot (um/sec)', rotation=0, ha='right')
ax.axes.get_xaxis().set_visible(False)

ax = fig.add_subplot(413)
ax.violinplot(std_v_plot, showextrema=False, showmeans=True)
ax.set_ylabel('velocity constancy', rotation=0, ha='right')
ax.axes.get_xaxis().set_visible(False)

ax = fig.add_subplot(414)
ax.violinplot(ang_v_plot, showextrema=False, showmeans=True)
ax.set_ylabel('angular velocity (deg/sec)',rotation=0, ha='right')
ax.set_xticks(np.arange(1,1+len(well_name)))
ax.set_xticklabels(well_name, rotation=45)

# plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.1)
#plt.tight_layout()
plt.savefig("v.png", bbox_inches = 'tight')
"""


"""
## box plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(mean_v_plot, labels=well_name)
ax.set_ylabel('velocity (mm/sec)')
ax.set_title('Mean velocity of each cell')
plt.xticks(rotation=45)
plt.savefig("mean_all.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(peak_v_plot, labels=well_name)
ax.set_ylabel('velocity (mm/sec)')
ax.set_title('Maximum velocity of each cell')
plt.xticks(rotation=45)
plt.savefig("peak_all.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(std_v_plot, labels=well_name)
ax.set_ylabel('velocity constancy')
ax.set_title('Constancy of the velocity of each cell')
plt.xticks(rotation=45)
plt.savefig("std_all.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(ang_v_plot, labels=well_name)
ax.set_ylabel('velocity (deg/sec)')
ax.set_title('Mean angular velocity of each cell')
plt.xticks(rotation=45)
plt.savefig("ang_all.png")

"""