import os
import pickle
import re
import sys
import json
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = "/home/kousuke/Research/place-model-unsupervised-exps/data/"
DATASETS= ['cold-freiburg', 'cold-saarbrucken', 'cold-stockholm']
EXT = 'mixed'
CLASS_MAP = 'class_map.json'


def plot_polar_scan(polar_scan):
    a, r = np.meshgrid(np.radians(angles), radiuses)
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'gray'),
                                                        (0.5, 'white'),
                                                        (1, 'black')])

    ax.pcolormesh(a, r, polar_scan.reshape(num_radius_cells, num_angle_cells) +1, cmap=cmap)
    plt.show()


def load_sequences(typ, dataset):
    seqdata = []
    for s in sorted(os.listdir(os.path.join(DATA_PATH, "polar_scans_"+typ, dataset))):
        fname = os.path.join(DATA_PATH, "polar_scans_"+typ, dataset, s)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            seqdata.extend(d)
    return seqdata


def save_data(data):
    d_name = os.path.join(DATA_PATH, "datasets")
    os.makedirs(d_name, exist_ok=True)
    with open(os.path.join(d_name, "real_data"), 'wb') as f:
        pickle.dump(data, f)


def save_set_defs():
    d_name = os.path.join(DATA_PATH, "datasets")
    os.makedirs(d_name, exist_ok=True)
    with open(os.path.join(d_name, "set_defs"), 'wb') as f:
        pickle.dump(set_defs, f)


def save_set_defs_json():
    d_name = os.path.join(DATA_PATH, "datasets")
    os.makedirs(d_name, exist_ok=True)
    with open(os.path.join(d_name, "set_defs.json"), 'wt') as f:
        json.dump(set_defs, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
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


    with open(os.path.join(DATA_PATH, 'datasets/', CLASS_MAP)) as f:
            class_map = json.load(f)

    proc_real_data = []
    counter = 0
    for dataset in DATASETS:
        real_data = load_sequences(EXT, dataset)

        for i in real_data:
            rid, rcat, scan = i

            # Give a unique id over all the dataset
            rid = dataset + '/' + rid

            # Map the room categories to the ones we use in the experiments
            if rid in class_map[dataset]["by_id"]:
                rcat = class_map[dataset]["by_id"][rid]
            elif rcat in class_map[dataset]["by_category"]:
                rcat = class_map[dataset]["by_category"][rcat]
            else:
                rcat = "unknown"

            proc_real_data.append([rid, rcat, scan])
            counter += 1

        # rooms_by_category={}
        # for i in proc_real_data:
        #     if i[1] not in rooms_by_category:
        #         rooms_by_category[i[1]] = set()
        #     rooms_by_category[i[1]].add(i[0])

        print(dataset)
        # print({k: len(i) for k,i in rooms_by_category.items()})
        # print(rooms_by_category)
        print(counter)
        counter = 0

    set_defs = {
        'train_rooms_1': [],
        'train_rooms_2': [],
        'train_rooms_3': [],
        'train_rooms_4': [],
        'train_rooms_5': [],
        'train_rooms_6': [],
        'test_rooms_1': [],
        'test_rooms_2': [],
        'test_rooms_3': [],
        'test_rooms_4': [],
        'test_rooms_5': [],
        'test_rooms_6': [],
        'known_categories': ['corridor', 'door', 'medium_room'],
        'novel_categories': ['unknown']
    }

    # Allocate room ids to train/test sets
    already_added = set()
    for data in proc_real_data:
        rid, _, _ = data
        if (rid not in already_added):
            if re.match(r"cold-stockholm/*", rid):
                set_defs['train_rooms_1'].append(rid)
                set_defs['train_rooms_2'].append(rid)
                set_defs['test_rooms_3'].append(rid)
                set_defs['test_rooms_5'].append(rid)
            if re.match(r"cold-freiburg/*", rid):
                set_defs['train_rooms_3'].append(rid)
                set_defs['train_rooms_4'].append(rid)
                set_defs['test_rooms_1'].append(rid)
                set_defs['test_rooms_6'].append(rid)
            if re.match(r"cold-saarbrucken/*", rid):
                set_defs['train_rooms_5'].append(rid)
                set_defs['train_rooms_6'].append(rid)
                set_defs['test_rooms_2'].append(rid)
                set_defs['test_rooms_4'].append(rid)

            already_added.add(rid)

    save_data(proc_real_data)
    save_set_defs()
    save_set_defs_json()
