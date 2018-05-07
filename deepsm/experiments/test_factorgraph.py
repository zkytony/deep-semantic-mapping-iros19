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
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.experiments.factor_graph.run_factor_graph_tests import FactorGraphTest
from deepsm.experiments.common import COLD_ROOT, TOPO_MAP_DB_ROOT, BP_EXEC_PATH, BP_RESULT_DIR, GRAPHSPN_RESULTS_ROOT

# Very hard coded. Bad code.
def parse_likelihood(path):
    needed = ""
    with open(path) as f:
        content = f.readlines()
        lh_start = False
        # Get the part in the test_instance.log file that actually belongs to likelihoods
        for line in content:
            if not lh_start:
                if re.search('likelihoods', line) is None:
                    continue
                else:
                    lh_start = True
            else:
                if re.search('query', line) is not None:
                    lh_start = False
            
            if lh_start:
                needed += line.strip()
    needed = needed.replace('array(', '')
    needed = needed.replace(')', '')
    return eval("{" + needed + "}")
            


def same_building(args):

    building_configs = {
        "Stockholm": ["456-7", "457-6", "467-5", "567-4"],
        "Freiburg": ["12-3", "13-2", "23-1"],
        "Saarbrucken": ["123-4", "124-3", "134-2", "234-1"],
    }

    result_dir = os.path.join(BP_RESULT_DIR, args.exp_name, args.test_case + "_" + args.test_name)
    os.makedirs(result_dir, exist_ok=True)

    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    fg_testers = {}
    print("Loading factors...")
    for db_name in building_configs:
        for case in building_configs[db_name]:
            train_fl = case.split("-")[0]
            test_fl = case.split("-")[1]
            topo_dataset.load(db_name + train_fl, skip_unknown=True)
            topo_dataset.load(db_name + test_fl, skip_unknown=True)
            fg_tester = FactorGraphTest(BP_EXEC_PATH, topo_dataset,
                                        [db_name + train_fl],
                                        db_name + test_fl,
                                        result_dir=result_dir)
            fg_testers[db_name + test_fl] = fg_tester

    graphspn_results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, "results")
    for case_name in os.listdir(graphspn_results_dir):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            # In the result directory for case_name, there is a test_instance.log file
            # that contains the likelihoods for this test case.
            print("    " + case_name)
            timestamp = case_name.split("_")[1]
            likelihoods = parse_likelihood(os.path.join(graphspn_results_dir, case_name, 'test_instance.log'))
            import pdb; pdb.set_trace()



def main():
    parser = argparse.ArgumentParser(description="Use factor graphs to run the same test cases for graphspn;" \
                                     "Test cases are located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")
    parser.add_argument("what", type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')

    args = parser.parse_args()
    if args.what == "DGSM_SAME_BUILDING":
        same_building(args)

if __name__ == "__main__":
    main()
