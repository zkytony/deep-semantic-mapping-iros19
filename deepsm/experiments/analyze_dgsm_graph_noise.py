#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
import numpy as np
import os, sys
import argparse
from deepsm.graphspn.tbm.dataset import TopoMapDataset
import deepsm.experiments.cold_database_experiment as cde
import deepsm.experiments.paths as paths
import deepsm.util as util
from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT,\
    TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT
from pprint import pprint
import re

def normalize_marginals(marginals):
    """Given an array of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    result = {}
    for nid in marginals:
        likelihoods = np.array(marginals[nid]).flatten()
        normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                           (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
        result[nid] = normalized
    return result

def get_noise_level(topo_map, likelihoods):
    stat_incrct = []
    stat_crct = []    
    groundtruth = topo_map.current_category_map()
    for nid in likelihoods:
        prediction = np.argmax(likelihoods[nid])
        
        if prediction != groundtruth[nid]:
            # incorrect
            truecl_lh = likelihoods[nid][groundtruth[nid]]
            largest_lh = np.max(likelihoods[nid])
            stat_incrct.append(largest_lh - truecl_lh)
        else:
            # made incorrect
            largest2 = heapq.nlargest(2, likelihoods[nid])
            stat_crct.append(largest2[0] - largest2[1])
    return stat_crct, stat_incrct

def compute_noise_level(db_name, test_case):
    """
    db_name e.g. Stockholm
    test_case e.g. 456-7
    """
    coldmgr = util.ColdDatabaseManager(db_name, COLD_ROOT)
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    train_floors, test_floor = test_case.split("-")[0],\
                               test_case.split("-")[1]
    dataset.load(db_name, skip_unknown=True, skip_placeholders=True, single_component=False)

    topo_maps = dataset.get_topo_maps(db_name=db_name, amount=-1)
    stat_incrct = []
    stat_crct = []
    print("Processing %s" % db_name)
    for seq_id in sorted(topo_maps):
        floor = re.search("floor[1-9]+", seq_id).group().split("floor")[-1]
        if floor != test_floor:
            continue

        sys.stdout.write("    " + seq_id + "\n")
        sys.stdout.flush()
        
        topo_map = topo_maps[seq_id]
        results_dir = paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                              db_name,
                                                              "graphs",
                                                              0,
                                                              train_floors,
                                                              test_floor)
        for filename in os.listdir(results_dir):
            if filename.startswith(db_name.lower() + test_case.split("-")[1])\
               and filename.endswith("_likelihoods.json"):
                graph_id = filename.split("_likelihoods.json")[0]
                lh = cde.load_likelihoods(results_dir, graph_id, topo_map,
                                          util.CategoryManager.known_categories())
            
                stat_crct_seq, stat_incrct_seq = get_noise_level(topo_map,
                                                                 likelihoods=normalize_marginals(lh))
                stat_crct.extend(stat_crct_seq)
                stat_incrct.extend(stat_incrct_seq)

    result = {
        '_D_incorrect_': {
            'avg': np.mean(stat_incrct),
            'std': np.std(stat_incrct),
        },
        '_D_correct_': {
            'avg': np.mean(stat_crct),
            'std': np.std(stat_crct),
        }
    }
    return result
    
                
def main():
    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('db_name', type=str, help="e.g. Stockholm")
    parser.add_argument('test_case', type=str, help="e.g. 456-7")
    parser.add_argument("-k", "--category-type", type=str, help="either SEVEN, SIMPLE, FULL, or BINARY", default="SIMPLE")
    args = parser.parse_args()
    
    util.CategoryManager.TYPE = args.category_type
    util.CategoryManager.init()
    
    pprint(compute_noise_level(args.db_name, args.test_case))

if __name__ == "__main__":
    main()
