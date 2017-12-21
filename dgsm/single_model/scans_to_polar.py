from scipy.misc import imread, imshow, imsave
from glob import glob
import numpy as np
import os
from math import degrees, radians
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
import json
import sys
import re
import pickle
import libspn as spn
np.set_printoptions(threshold=np.inf)


DATA_PATH = "/home/kousuke/Research/place-model-unsupervised-exps/data/"
EXT = "mixed"


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


def process_sequence(seq_name, typ):
    scans_path = os.path.join(DATA_PATH, "virtual_scans_"+typ, seq_name)
    scan_pgms = sorted(glob(os.path.join(scans_path, "*.pgm"))) 
    proc_path = os.path.join(DATA_PATH, "proc_scans_"+typ, seq_name)
    floor_name = re.sub("_.*", "", seq_name)
    os.makedirs(proc_path, exist_ok=True)
    polar_scans = []
    for pgm in scan_pgms:
        fname = os.path.basename(pgm)
        tstamp_rid = re.sub(".pgm", "", fname)
        # Load scan and annotations
        scan = imread(pgm)
        with open(os.path.join(scans_path, tstamp_rid+".json")) as f:
            d = json.load(f)
        # Check if we want to keep this scan
        rid = d['room_id']
        rcat = rid.split('-')[1]
        # Store
        imsave(arr=scan, name=os.path.join(proc_path, tstamp_rid + ".png"))
        polar_scan = scan_to_polar(scan)
        polar_scans.append([rid, rcat, polar_scan.ravel()])
        sys.stdout.write('.')
    return polar_scans


def save_sequence(seq, name, typ):
    polar_path = os.path.join(DATA_PATH, "polar_scans_"+typ, os.path.dirname(name))
    os.makedirs(polar_path, exist_ok=True)
    with open(os.path.join(polar_path, os.path.basename(name)), 'wb') as f:
        pickle.dump(seq, f)


def proc_to_polar(seq_name):
    proc_path = os.path.join(DATA_PATH, "proc_scans_"+typ, seq_name)
    proc_pngs = sorted(glob(os.path.join(proc_path, "*.png"))) 
    floor_name = re.sub("_.*", "", seq_name)
    polar_scans = []
    for png in proc_pngs:
        fname = os.path.basename(png)
        tstamp_rid = re.sub(".png", "", fname)
        # Load scan and annotations
        scan = imread(png)
        # Check if we want to keep this scan
        rid = tstamp_rid.split('_')[1]
        rcat = rid.split('-')[1]
        # Store
        polar_scan = scan_to_polar(scan)
        polar_scans.append([rid, rcat, polar_scan.ravel()])
        sys.stdout.write('.')
    return polar_scans


def do_sequence(seq_name, data_typ):
    # processed_seq = process_sequence(seq_name, typ=data_typ)
    processed_seq = proc_to_polar(seq_name, typ=data_typ)
    save_sequence(processed_seq, seq_name, typ=data_typ)


if __name__ == '__main__':

    # Configure params
    resolution = 0.02
    num_angle_cells = 56
    min_radius = 0.3
    max_radius = 5
    radius_factor = 1.15

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

    do_sequence('cold-stockholm/floor4_cloudy_b', EXT)
    do_sequence('cold-stockholm/floor4_night_b', EXT)
    do_sequence('cold-stockholm/floor5_cloudy_b', EXT)
    do_sequence('cold-stockholm/floor5_night_b', EXT)
    do_sequence('cold-stockholm/floor6_cloudy_b', EXT)
    do_sequence('cold-stockholm/floor6_night_b', EXT)
    do_sequence('cold-stockholm/floor7_cloudy_b', EXT)
    do_sequence('cold-stockholm/floor7_night_b', EXT)

    do_sequence('cold-freiburg/seq1_cloudy2', EXT)
    do_sequence('cold-freiburg/seq1_sunny2', EXT)
    do_sequence('cold-freiburg/seq2_cloudy2', EXT)
    do_sequence('cold-freiburg/seq2_sunny2', EXT)
    do_sequence('cold-freiburg/seq3_cloudy2', EXT)
    do_sequence('cold-freiburg/seq3_sunny2', EXT)

    do_sequence('cold-saarbrucken/seq1_cloudy2', EXT)
    do_sequence('cold-saarbrucken/seq1_night2', EXT)
    do_sequence('cold-saarbrucken/seq2_cloudy2', EXT)
    do_sequence('cold-saarbrucken/seq2_night2', EXT)
    do_sequence('cold-saarbrucken/seq3_cloudy2', EXT)
    do_sequence('cold-saarbrucken/seq3_night2', EXT)
    do_sequence('cold-saarbrucken/seq4_cloudy2', EXT)
    do_sequence('cold-saarbrucken/seq4_night2', EXT)
