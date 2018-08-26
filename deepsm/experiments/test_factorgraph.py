#!/usr/bin/env python
#
# Use factor graph to compare with graphspn for the same experiments

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
#matplotlib.use('Agg')

import os
import re
import argparse
import yaml
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.experiments.factor_graph.run_factor_graph_tests import FactorGraphTest
from deepsm.experiments.common import COLD_ROOT, TOPO_MAP_DB_ROOT, BP_EXEC_PATH, BP_RESULTS_ROOT, GRAPHSPN_RESULTS_ROOT
from deepsm.util import CategoryManager, print_in_box
from deepsm.graphspn.tests.tbm.runner import normalize_marginals


def parse_likelihood(path):
    needed = ""
    with open(path) as f:
        r = yaml.load(f)
    lh_log = r['likelihoods']
    return normalize_marginals(lh_log)

    # needed = ""
    # with open(path) as f:
    #     content = f.readlines()
    #     lh_start = False
    #     # Get the part in the test_instance.log file that actually belongs to likelihoods
    #     for line in content:
    #         if not lh_start:
    #             if re.search('likelihoods', line) is None:
    #                 continue
    #             else:
    #                 lh_start = True
    #         else:
    #             if re.search('query', line) is not None:
    #                 lh_start = False

    #         if lh_start:
    #             needed += line.strip()
    # needed = needed.replace('array(', '')
    # needed = needed.replace(')', '')
    # return eval("{" + needed + "}")

def parse_test_info(path):
    test_info = {}
    with open(path) as f:
        content = f.readlines()
        for line in content:
            if re.search('db_name', line) is not None:
                test_info['test_db'] = eval("{" + line + "}")['db_name']
            if re.search('graph_id', line) is not None:
                test_info['graph_id'] = eval("{" + line + "}")['graph_id']
    return test_info


def same_building(args):

    building_configs = {
        "Stockholm": ["456-7", "457-6", "467-5", "567-4"],
        "Freiburg": ["12-3", "13-2", "23-1"],
        "Saarbrucken": ["123-4", "124-3", "134-2", "234-1"],
    }

    results_dir = os.path.join(BP_RESULTS_ROOT, args.exp_name, args.test_case + "_" + args.test_name)
    os.makedirs(results_dir, exist_ok=True)

    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    fg_testers = {}
    print("Loading factors...")
    for db_name in building_configs:
        for case in building_configs[db_name]:
            train_fl = case.split("-")[0]
            test_fl = case.split("-")[1]
            topo_dataset.load(db_name + train_fl, skip_unknown=True, skip_placeholders=args.skip_placeholders)
            topo_dataset.load(db_name + test_fl, skip_unknown=True, skip_placeholders=args.skip_placeholders)
            fg_tester = FactorGraphTest(BP_EXEC_PATH, topo_dataset,
                                        [db_name + train_fl],
                                        db_name + test_fl,
                                        result_dir=results_dir)
            fg_testers[db_name + test_fl] = fg_tester

    graphspn_results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, "results")
    for case_name in sorted(os.listdir(graphspn_results_dir)):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            # In the result directory for case_name, there is a test_instance.log file
            # that contains the likelihoods for this test case.
            print_in_box([case_name])
            case_path = os.path.join(graphspn_results_dir, case_name)
            timestamp = case_name.split("_")[1]
            # Obtain case information
            test_info = parse_test_info(os.path.join(case_path, 'testing_info_%s.log' % timestamp))
            # Prepare parameters for running factor graph test
            fg_tester = fg_testers[test_info['test_db']]
            seq_id = "_".join(test_info['graph_id'].split("_")[1:])
            topo_map = topo_dataset.get(test_info['test_db'], seq_id)
            masked = {nid:-1 for nid in topo_map.nodes}
            groundtruth = topo_map.current_category_map()
            case_results_path = os.path.join(results_dir, case_name)
            os.makedirs(case_results_path, exist_ok=True)
            likelihoods = parse_likelihood(os.path.join(case_path, 'test_instance.log'))
            for nid in topo_map.nodes:
                if nid not in likelihoods:
                    if not topo_map.nodes[nid].placeholder:
                        print(nid)
                    likelihoods[nid] = np.full((CategoryManager.NUM_CATEGORIES,), 1)
                    likelihoods[nid] = (likelihoods[nid] / np.sum(likelihoods[nid])).tolist()
            
            _, stats = fg_tester.run_instance(seq_id, topo_map, masked, groundtruth, likelihoods=likelihoods,
                                              result_path=case_results_path, visualize=True,
                                              avoid_placeholders=True)
            with open(os.path.join(case_results_path, 'report.log'), 'w') as f:
                yaml.dump(stats, f)



def main():
    parser = argparse.ArgumentParser(description="Use factor graphs to run the same test cases for graphspn;" \
                                     "Test cases are located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")
    parser.add_argument("what", type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')
    parser.add_argument("--category-type", type=str, help="either SIMPLE, FULL or BINARY", default="SIMPLE")
    parser.add_argument("--skip-placeholders", help='Skip placeholders. Placeholders will not be part of the graph.', action='store_true')

    args = parser.parse_args()
    
    CategoryManager.TYPE = args.category_type
    CategoryManager.init()
    
    if args.what == "DGSM_SAME_BUILDING":
        same_building(args)

if __name__ == "__main__":
    main()
