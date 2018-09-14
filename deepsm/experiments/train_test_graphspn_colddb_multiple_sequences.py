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
                    trial,
                    relax_level,
                    template,
                    num_partitions,
                    num_sampling_rounds,
                    test_floor=None,
                    train_floors_str=None,
                    skip_placeholders=True,
                    random_sampling=False,
                    category_type="SIMPLE",
                    similarity_coeff=-3.0,
                    complexity_coeff=7.0,
                    straight_template_coeff=8.0,
                    dom_coeff=4.85,
                    separable_coeff=2.15,
                    train_with_likelihoods=False,
                    investigate=False,
                    epochs_training=None,
                    likelihood_thres=0.05):
    nullable_args = []
    if skip_placeholders:
        nullable_args.append('--skip-placeholders')
    if random_sampling:
        nullable_args.append('--random-sampling')
    if train_with_likelihoods:
        nullable_args.append('--train-with-likelihoods')
    if investigate:
        nullable_args.append('--investigate')
    if epochs_training is not None:
        nullable_args.extend(['-E', str(epochs_training)])

    if what == "DGSM_SAME_BUILDING":
        coeffs = ['--similarity-coeff', similarity_coeff,
                  '--complexity-coeff', complexity_coeff,
                  '--straight_template-coeff', straight_template_coeff,
                  '--dom-coeff', dom_coeff,
                  '--separable-coeff', separable_coeff]
        proc = subprocess.Popen(['./train_test_graphspn.py',
                                 "samebuilding",
                                 db_name,
                                 seq_id,
                                 str(test_floor),
                                 train_floors_str,
                                 '-s', str(seed),
                                 '-e', exp_name,
                                 '-r', str(relax_level),
                                 '-t', test_name,
                                 '-l', str(trial),
                                 '-P', str(num_partitions),
                                 '-R', str(num_sampling_rounds),
                                 '-L', str(likelihood_thres),
                                 '--category-type', category_type,
                                 '--template', template] + nullable_args)
    elif what == "DGSM_ACROSS_BUILDINGS":
        proc = subprocess.Popen(['./train_test_graphspn.py',
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


    num_seqs_tested = 0
    for test_floor in sorted(floors):

        if args.test_floor is not None and test_floor != args.test_floor:
            continue

        print("Testing on floor %d" % test_floor)
        for seq_id in sorted(os.listdir(os.path.join(TOPO_MAP_DB_ROOT, "%s%d" % (db_name, test_floor)))):
            train_floors_str = "".join(sorted(map(str, floors - {test_floor})))
            dirpath_to_dgsm_graphs_result = paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                                                    db_name,
                                                                                    "graphs",
                                                                                    args.trial,
                                                                                    train_floors_str,
                                                                                    str(test_floor))
            if not os.path.exists(os.path.join(dirpath_to_dgsm_graphs_result, "%s%d_%s_likelihoods.json" % (db_name.lower(), test_floor, seq_id))):
                print("Skipping %s. No DGSM result found." % seq_id)
                continue

            print("...%s..." % seq_id)
            proc = experiment_proc("DGSM_SAME_BUILDING", db_name, seq_id,
                                   args.seed, args.exp_name, args.test_name, args.trial,
                                   args.relax_level, args.template, args.num_partitions, args.num_sampling_rounds,
                                   test_floor=test_floor, train_floors_str=train_floors_str, random_sampling=args.random_sampling,
                                   category_type=args.category_type, similarity_coeff=args.similarity_coeff, complexity_coeff=args.complexity_coeff,
                                   dom_coeff=args.dom_coeff, separable_coeff=args.separable_coeff, straight_template_coeff=args.straight_template_coeff,
                                   train_with_likelihoods=args.train_with_likelihoods, investigate=args.investigate, epochs_training=args.epochs_training,
                                   likelihood_thres=args.likelihood_thres)
            proc.wait()
            num_seqs_tested += 1
            if args.num_test_seqs >= 0 and num_seqs_tested >= args.num_test_seqs:
                print("Test sequence limit of %d is reached" % num_seqs_tested)
                return


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
    parser.add_argument('what', type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')
    parser.add_argument('-d', '--db_name', type=str, help="e.g. Stockholm")
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
    parser.add_argument('-P', '--num-partitions', type=int, help="Number of times the graph is partitioned (i.e. number of children for the root sum in GraphPSN)", default=5)
    parser.add_argument('-R', '--num-sampling-rounds', type=int, help="Number of rounds to sample partition sets before picking the best one.", default=100)
    parser.add_argument('-E', '--epochs-training', type=int, help="Number of epochs to train models.", default=100)
    parser.add_argument('-L', '--likelihood-thres', type=float, help="Likelihood update threshold for training.", default=0.05)
    parser.add_argument('-l', '--trial', type=int, help="Trial number to identify DGSM experiment result", default=0)
    parser.add_argument('--test-floor', type=int, help="Floor number that will be used for testing. Other floors are then for training.")
    parser.add_argument("--skip-placeholders", help='e.g. Freiburg', action='store_true')
    parser.add_argument("--category-type", type=str, help="either SIMPLE, FULL, or BINARY", default="SIMPLE")
    parser.add_argument("--template", type=str, help="either VIEW, THREE, or STAR", default="THREE")
    parser.add_argument("--random-sampling", action="store_true", help='Sample partitions randomly (but with higher complexity first). Not using a sampler.')
    parser.add_argument("--similarity-coeff", type=float, default=-3.0)
    parser.add_argument("--complexity-coeff", type=float, default=7.0)
    parser.add_argument("--straight-template-coeff", type=float, default=8.0)
    parser.add_argument("--dom-coeff", type=float, default=4.85)
    parser.add_argument("--separable-coeff", type=float, default=2.15)
    parser.add_argument("--train-with-likelihoods", action="store_true")
    parser.add_argument("--investigate", action="store_true", help="If set, loss plots during training will be saved to the same directory as the models.")
    args = parser.parse_args()

    util.CategoryManager.TYPE = args.category_type
    util.CategoryManager.init()

    if args.what == "DGSM_SAME_BUILDING":
        same_buliding(args)
    elif args.what == "DGSM_ACROSS_BUILDINGS":
        across_buildings(args)

if __name__ == "__main__":
    main()
