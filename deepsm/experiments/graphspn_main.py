#!/usr/bin/env python3
#
# Entry point to run graphspn experiment on a lot of sequences

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import os, re
import argparse
import subprocess
import time
import deepsm.util as util
import deepsm.experiments.paths as paths
from deepsm.experiments.common import TOPO_MAP_DB_ROOT

def experiment_proc(what,
                    db_name,
                    seq_id,
                    seed,
                    exp_name,
                    test_name,
                    relax_level,
                    test_floor=None,
                    train_floors_str=None,
                    skip_placeholders=True):
    skip_placeholder_arg = ['--skip-placeholders'] if skip_placeholders else []
    if what == "DGSM_SAME_BUILDING":
        proc = subprocess.Popen(['./train_test_graphspn_classification.py',
                                 "samebuilding",
                                 db_name,
                                 seq_id,
                                 str(test_floor),
                                 train_floors_str,
                                 '-s', str(seed),
                                 '-e', exp_name,
                                 '-r', str(relax_level),
                                 '-t', test_name] + skip_placeholder_arg)
    elif what == "DGSM_ACROSS_BUILDINGS":
        proc = subprocess.Popen(['./train_test_graphspn_classification.py',
                                 "acrossbuildings",
                                 db_name,
                                 seq_id,
                                 '-s', str(seed),
                                 '-e', exp_name,
                                 '-r', str(relax_level),
                                 '-t', test_name] + skip_placeholder_arg)
    return proc


def same_buliding(args):
    # Run all experiments for the entire Stockholm building
    db_name = args.db_name
    if db_name == "Stockholm":
        floors = {4, 5, 6, 7}
    elif db_name == "Freiburg":
        floors = {1, 2, 3}
    elif db_name == "Saarbrucken":
        floors = {1, 2, 3, 4}

    if args.seq_id is None:    
        num_seqs_tested = 0
        for test_floor in sorted(floors):
            for seq_id in sorted(os.listdir(os.path.join(TOPO_MAP_DB_ROOT, "%s%d" % (db_name, test_floor)))):
                train_floors_str = "".join(sorted(map(str, floors - {test_floor})))
                dirpath_to_dgsm_graphs_result = paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                                                        db_name, "graphs",
                                                                                        train_floors_str, str(test_floor))
                if not os.path.exists(os.path.join(dirpath_to_dgsm_graphs_result, "%s%d_%s_likelihoods.json" % (db_name.lower(), test_floor, seq_id))):
                    print("Skipping %s. No DGSM result found." % seq_id)
                    continue

                print("...%s..." % seq_id)
                proc = experiment_proc("DGSM_SAME_BUILDING", db_name, seq_id,
                                       args.seed, args.exp_name, args.test_name, args.relax_level,
                                       test_floor=test_floor, train_floors_str=train_floors_str)
                proc.wait()
                num_seqs_tested += 1
                if args.num_test_seqs >= 0 and num_seqs_tested >= args.num_test_seqs:
                    print("Test sequence limit of %d is reached" % num_seqs_tested)
                    return
    else:
        seq_id = args.seq_id
        print("...%s..." % seq_id)
        test_floor = int(re.search("[0-9]+", seq_id).group())
        train_floors_str = "".join(sorted(map(str, floors - {test_floor})))
        proc = experiment_proc("DGSM_SAME_BUILDING", db_name, seq_id,
                               args.seed, args.exp_name, args.test_name, args.relax_level,
                               test_floor=test_floor, train_floors_str=train_floors_str)
        proc.wait()


def across_buildings(args):
    # Because tensorflow doesn't free the graph and ops in RAM memory, we have to
    # run experiment one sequence at a time
    test_db = args.db_name
    num_seqs_tested = 0
    for seq_id in sorted(os.listdir(os.path.join(TOPO_MAP_DB_ROOT, "%s" % test_db))):
        proc = experiment_proc("DGSM_ACROSS_BUILDINGS", test_db, seq_id,
                               args.seed, args.exp_name, args.test_name, args.relax_level)
        proc.wait()
        num_seqs_tested += 1
        if args.num_test_seqs >= 0 and num_seqs_tested >= args.num_test_seqs:
            print("Test sequence limit of %d is reached" % num_seqs_tested)
            return
        

def main():
    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('db_name', type=str, help="e.g. Stockholm")
    parser.add_argument('what', type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')
    parser.add_argument('-s', '--seed', type=int, help="Seed of randomly generating SPN structure. Default 100",
                        default=100)
    parser.add_argument('-e', '--exp-name', type=str, help="Name to label this experiment. Default: GraphSPNToyExperiment",
                        default="GraphSPNToyExperiment")
    parser.add_argument('-t', '--test-name', type=str, help="Name for grouping the experiment result. Default: mytest",
                        default="mytest")
    parser.add_argument('-r', '--relax-level', type=float, help="Adds this value to every likelihood value and then re-normalize all likelihoods (for each node)",
                        default=0.0)
    parser.add_argument('-N', '--num-test-seqs', type=int, help="Total number of sequences to test on",
                        default=-1)
    parser.add_argument('--seq-id', type=str, help="Sequence ID to run on; If supplied, -N is suppressed.")
    parser.add_argument("--skip-placeholders", help='e.g. Freiburg', action='store_true')
    args = parser.parse_args()

    if args.what == "DGSM_SAME_BUILDING":
        same_buliding(args)
    elif args.what == "DGSM_ACROSS_BUILDINGS":
        across_buildings(args)
            

if __name__ == "__main__":
    main()
