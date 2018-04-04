#!/usr/bin/env python
# This script generates polar scans from cartesian virtual scans.
# The logic references scans_to_polar.ipynb. Not intended to be
# imported by any other class.
#
# For now, still use the COLD dataset path.

import argparse
from glob import glob
from scipy.misc import imread, imshow, imsave
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np
import pickle
import os
import re
import sys

resolution      = 0.02
num_angle_cells = 56
min_radius      = 0.3
max_radius      = 5
radius_factor   = 1.15

angles = np.linspace(-180, 180, num_angle_cells+1)
r=min_radius
radiuses=[r]
v = 0.04
while r<max_radius:
    r = r+v
    radiuses.append(r)
    v*=radius_factor
radiuses = np.array(radiuses)
num_radius_cells = len(radiuses)-1

def pixel_to_polar(scan_size, x, y):
    """Convert pixel coordinate to polar cell coordinate."""
    c_x = scan_size[0]//2
    c_y = scan_size[1]//2
    x = x - c_x
    y = y - c_y
    r = np.sqrt(x**2 + y**2) * resolution
    alpha = np.arctan2(-y, x) # Angles go clockwise with the -
    return (r, np.degrees(alpha))

def scan_to_polar(scan_image):
    ys,xs = np.meshgrid(np.arange(scan_image.shape[0])+0.5, np.arange(scan_image.shape[1])+0.5)
    rr, aa = pixel_to_polar(scan_image.shape, xs, ys)
    aa = np.digitize(aa, angles) - 1
    rr = np.digitize(rr, np.r_[0, radiuses]) - 1  # Additional cell for stuff near the robot
    polar_scan_elems = [[[] for _ in range(num_angle_cells)] for _ in range(num_radius_cells)]
    for x in range(scan_image.shape[0]):
        for y in range(scan_image.shape[1]):
            r = rr[x,y]
            a = aa[x,y]
            if r>0 and r<=num_radius_cells:
                polar_scan_elems[r-1][a].append(scan_image[x,y])
    for r in range(num_radius_cells):
        for a in range(num_angle_cells):
            vals=polar_scan_elems[r][a]
            free_count = sum(1 for i in vals if i>250)
            occupied_count = sum(1 for i in vals if i<10)
            unknown_count = len(vals) - free_count - occupied_count
            if not vals: # No elements!
                raise Exception("No elements in %s %s" % (r, a))
            if occupied_count/len(vals) > 0.01:        
                    val = 1
            elif free_count/len(vals) > 0.01:
                    val = 0
            else:
                val = -1
            polar_scan_elems[r][a]=val
    return np.array(polar_scan_elems)


def plot_polar_scan(polar_scan):
    a, r = np.meshgrid(np.radians(angles), radiuses)
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'gray'),
                                                        (0.5, 'white'),
                                                        (1, 'black')])

    ax.pcolormesh(a, r, polar_scan+1, cmap=cmap)
    plt.show()


def process_sequence(datapath, outpath, seq_id):
    """Generate polar scans for virtual scans in one sequence, which
    are stored at location `seq_scans_path`."""
    seq_scans_path = os.path.join(datapath, seq_id, 'vscans')
    scan_pgms = sorted(glob(os.path.join(seq_scans_path, "*.pgm"))) 
    floor_name = re.sub("_.*", "", seq_id)
    polar_scans = []
    try:
        for pgm in scan_pgms:
            # Naming convention: {timestamp}_{floor#-class-id}.pgm
            fname = os.path.basename(pgm)
            tstamp = fname.split("_")[0]
            room_id = fname.split("_")[1]
            room_class = room_id.split("-")[1]
            # Load scan and annotations
            scan = imread(pgm)
            polar_scan = scan_to_polar(scan)
            polar_scans.append([room_id, room_class, polar_scan.ravel()])
            sys.stdout.write('.')
            sys.stdout.flush()
    except KeyboardInterrupt as e:
        print("Terminating...")
        raise e
    finally:
        os.makedirs(outpath, exist_ok=True)
        polar_path = os.path.join(outpath, seq_id + "_scans.pkl")
        with open(polar_path, 'wb') as f:
            pickle.dump(polar_scans, f)



if __name__ == "__main__":
    """
    Convert virtual scans to polar scans and save to output
    path. The `datapath` is assumed to be:

    /{datapath}:
        /{seq_id}
            /vscans
                /...scans... (.pgm files)

    The output directory structure will become:
    /{outpath}:
        /{seq_id}_scans.pkl

    `params` are parameters for the polar scans:
      - resolution       (default 0.02)
      - num_angle_cells  (default 56)
      - min_radius       (default 0.3)
      - max_radius       (default 5)
      - radius_factor    (default 1.15)
    """
    parser = argparse.ArgumentParser(description='Mass generates polar scans from cartesian virtual scans.')
    parser.add_argument('datapath', type=str, help='path to directory that contains cartesian virtual scans.')
    parser.add_argument('outpath', type=str, help='path to output directory that will contain polar scans.')
    args = parser.parse_args()
    
    seq_ids = sorted(os.listdir(args.datapath))
    for seq_id in seq_ids:
        process_sequence(args.datapath, args.outpath, seq_id)
        print(seq_id + " done.")
