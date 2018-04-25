#!/usr/bin/env python3
#
# Entry point to run graphspn experiment on a lot of sequences

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import os
import argparse
import subprocess
import time
import deepsm.util as util
from deepsm.experiments.common import TOPO_MAP_DB_ROOT


def main():
    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('-s', '--seed', type=int, help="Seed of randomly generating SPN structure. Default 100",
                        default=100)
    parser.add_argument('-e', '--exp-name', type=str, help="Name to label this experiment. Default: GraphSPNToyExperiment",
                        default="GraphSPNToyExperiment")
    parser.add_argument('-r', '--relax-level', type=float, help="Adds this value to every likelihood value and then re-normalize all likelihoods (for each node)",
                        default=0.0)
    parser.add_argument('-N', '--num-test-seqs', type=int, help="Total number of sequences to test on",
                        default=-1)
    args = parser.parse_args()
    
    # Run all experiments for the entire Stockholm building
    db_name = "Stockholm"
    floors = {4, 5, 6, 7}
    num_seqs_tested = 0
    for test_floor in sorted(floors):
        for seq_id in sorted(os.listdir(os.path.join(TOPO_MAP_DB_ROOT, "%s%d" % (db_name, test_floor)))):

            print("...%s..." % seq_id)
            
            train_floors_str = "".join(sorted(map(str, floors - {test_floor})))
            proc = subprocess.Popen(['./train_test_graphspn_classification.py',
                                     db_name,
                                     seq_id,
                                     str(test_floor),
                                     train_floors_str,
                                     '-s', str(args.seed),
                                     '-e', args.exp_name,
                                     '-r', str(args.relax_level)])
            proc.wait()
            num_seqs_tested += 1
            if args.num_test_seqs >= 0 and num_seqs_tested >= args.num_test_seqs:
                print("Test sequence limit of %d is reached" % num_seqs_tested)
                return


if __name__ == "__main__":
    main()
