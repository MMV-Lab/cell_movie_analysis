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
time_step_in_min = 2
px_size_in_micron = 1

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
        p_velocity = [np.sqrt(p[0]**2 + p[1]**2) for p in p_displacement]
        mean_velocity = np.mean(p_velocity)
        p90_velocity = np.percentile(p_velocity, 90)
        peak_velocity = np.max(p_velocity)
        std_velocity = np.std(p_velocity)

        p_rad = [math.atan2(p[0], p[1]) for p in p_displacement]
        p_rad_prev = p_rad[:-1]
        p_rad_next = p_rad[1:]
        p_ang_rad = [np.subtract(x, y) for x, y in zip(p_rad_next, p_rad_prev)]
        p_ang_deg = [math.degrees(p) for p in p_ang_rad]
        mean_ang_velocity = np.mean(p_ang_deg)

        # exclude erythrocyte
        if mean_velocity > 20:
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

    make_polar_plot(lineage, ignore_track, f"{this_well}_polar.png")


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

