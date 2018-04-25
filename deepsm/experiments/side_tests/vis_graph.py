#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import json
import deepsm.util as util
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.experiments.common import DGSM_DB_ROOT, TOPO_MAP_DB_ROOT, COLD_ROOT


if __name__ == "__main__":
    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    topo_dataset.load("Stockholm", skip_unknown=True)
    topo_map = topo_dataset.get("Stockholm", "floor7_cloudy_b")

    with open("stockholm7_floor7_cloudy_b_likelihoods.json") as f:
        lh = json.load(f)

    classes = []
    for i in range(util.CategoryManager.NUM_CATEGORIES):
        classes.append(util.CategoryManager.category_map(i, rev=True))

    print(classes)

    abs_log = {}
    for nid in lh:
        class_gt = classes.index(lh[nid][0])
        class_pred = classes.index(lh[nid][1])
        abs_log[nid] = [class_gt, class_pred, list(np.abs(lh[nid][3]))]

    with open("abs_log_lh.json", "w") as f:
        json.dump(abs_log, f)

    # Visualize topo map with node id
    ColdMgr = util.ColdDatabaseManager("Stockholm", COLD_ROOT)

    rcParams['figure.figsize'] = 18, 11

    ax = plt.gca()
    topo_map.visualize(ax, canonical_map_yaml_path=ColdMgr.groundtruth_file("floor7", "map.yaml"),
                       consider_placeholders=True, show_nids=True)
    plt.show()
