#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
from pylab import rcParams

from deepsm.graphspn.tbm.topo_map import TopologicalMap
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.util import CategoryManager, ColdDatabaseManager, compute_view_number
import csv
import matplotlib.pyplot as plt
import os, sys
from pprint import pprint
import random
import numpy as np

from deepsm.experiments.common import COLD_ROOT,\
    TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT


def TEST_topo_map_visualization(dataset, coldmgr, seq_id=None):
    rcParams['figure.figsize'] = 44, 30
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=1, seq_id=seq_id)
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        print(len(topo_map.nodes))
        topo_map.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), show_nids=True)
        plt.savefig('%s.png' % seq_id)
        plt.clf()
        plt.close()


if __name__ == "__main__":
    seq = sys.argv[1]
    CategoryManager.TYPE = "SEVEN"
    CategoryManager.init()
    
    coldmgr = ColdDatabaseManager("Stockholm", None, gt_root=GROUNDTRUTH_ROOT)
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load("Stockholm", skip_unknown=True, skip_placeholders=True, single_component=False)
    TEST_topo_map_visualization(dataset, coldmgr, seq_id=seq)
