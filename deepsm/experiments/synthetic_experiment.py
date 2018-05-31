#!/usr/bin/env python

# This script runs map instance tests and collects statistics. The reason to have
# a separate script like this is to relase memory. tf.Session() does not
# release memory even if it is closed

import sys, os
import subprocess
import numpy as np

import argparse
import glob
import yaml
import time
from pprint import pprint

from deepsm.util import CategoryManager, print_banner, print_in_box, ColdDatabaseManager
from deepsm.experiments.common import COLD_ROOT, DGSM_RESULTS_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT


def get_stats(exp_name, test_name, db_name):
    # Because we might terminate the test in the middle, we need to get stats
    # as we go. Get all stats at the end
    total_instances = 0
    overall_stats = {}
    overall_accs = {}
    for fname in glob.glob(os.path.join(GRAPHSPN_RESULTS_ROOT, exp_name, 'results', 'overall_%s*.log' % test_name)):
        with open(fname) as f:
            stats = yaml.load(f)
        for subname in stats:  # The subname here refers to the {General} part of the full subname
            if subname not in overall_stats:
                overall_stats[subname] = {'_overall_':0, '_total_correct_':0, '_total_inferred_':0,
                                          '_total_instances_': 0, '_test_db_': db_name}
                overall_accs[subname] = []
            total, correct = 0, 0
            for key in stats[subname]:
                if not key.startswith("_"):
                    if key not in overall_stats[subname]:
                        overall_stats[subname][key] = [0, 0, 0]
                    overall_stats[subname][key][0] += stats[subname][key][0]
                    overall_stats[subname][key][1] += stats[subname][key][1]
                    overall_stats[subname][key][2] = overall_stats[subname][key][0] / max(1, overall_stats[subname][key][1])
                    correct += stats[subname][key][0]
                    total += stats[subname][key][1]
            overall_stats[subname]['_total_correct_'] += correct
            overall_stats[subname]['_total_inferred_'] += total
            overall_stats[subname]['_overall_'] = overall_stats[subname]['_total_correct_'] / max(1, overall_stats[subname]['_total_inferred_'])            
            overall_stats[subname]['_total_instances_'] += stats[subname]['_total_instances_']
            for case_name in stats[subname]['_completed_cases_']:
                # record the overall accuracy for every instance, by reading the report
                with open(os.path.join(os.path.dirname(fname), case_name, 'report.log')) as repf:
                    instance_report = yaml.load(repf)
                    overall_accs[subname].append(instance_report['_overall_'])
            overall_stats[subname]['_std_'] = np.std(overall_accs[subname])
            overall_stats[subname]['_mean_'] = np.mean(overall_accs[subname])
    return overall_stats

def synthetic():
    parser = argparse.ArgumentParser(description='Run instance spn tests and collect statistics.')
    parser.add_argument('db_abrv', type=str, help='Database name. sa=Saarbrucken, fr=Freiburg, st=Stockholm. More options see test_instance_spn.py -h for help.')
    parser.add_argument('train_kwargs_file', type=str, help='Path to YAML file for training parameters. Set to `-` if want to only check stats.')
    parser.add_argument('test_kwargs_file', type=str, help='Path to YAML file for training parameters. Set to `-` if want to only check stats.')
    parser.add_argument('-N', '--num-runs', type=int, help="number of sequences to test on. Default: 0 (nothing will be run; only show stats)", default=0)
    parser.add_argument('-n', '--num-rounds', type=int, help="number of inference tasks per sequence. Only used for INF_RAND and INF_FIXED_RAND. Default: 3.", default=3)
    parser.add_argument('-e', '--exp-name', type=str, help='Name of this experiment as a whole', default="InstanceSpnExperiment")
    parser.add_argument('-t', '--test-name', type=str, help="Name of this test. An experiment may have multiple tests. A test name is used to identify tests" \
                        "that should belong to the same parameter setting.")
    parser.add_argument('-s', '--seq-id', type=str, help='sequence id to do the experiments on.', default=None)
    parser.add_argument('-lo-c', '--low-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-c', '--high-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-lo-ic', '--low-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-ic', '--high-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-unif', '--uniform-for-incorrect', action="store_true", help="Set likelihoods for nodes labeled incorrect to be uniform. If supplied, override what is in the test_kwargs file")
    parser.add_argument('-seg', '--segment', action="store_true", help="Segment the graph such that each node is a room, instead of a place. If supplied, override what is in the test_kwargs file")
    parser.add_argument('-p', '--num-partitions', type=int, help="Number of partitions of the graph. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('--no-save', action='store_true', help="Do not save the stats to a file. Useful if you only want to see the stats")

    args = parser.parse_args(sys.argv[2:])

    if args.db_abrv == 'fr':
        db_name = "Freiburg"
    elif args.db_abrv == 'st':
        db_name = "Stockholm"
    elif args.db_abrv == 'sa':
        db_name = "Saarbrucken"
        
    try:
        # Deal with all optinal parameters
        optional_params = []
        if args.seq_id:
            optional_params.extend(['-i', args.seq_id])
        if args.test_name:
            optional_params.extend(['-t', args.test_name])
        if args.low_likelihood_correct:
            optional_params.extend(['-lo-c', args.low_likelihood_correct])
        if args.high_likelihood_correct:
            optional_params.extend(['-hi-c', args.high_likelihood_correct])
        if args.low_likelihood_incorrect:
            optional_params.extend(['-lo-ic', args.low_likelihood_incorrect])
        if args.high_likelihood_incorrect:
            optional_params.extend(['-hi-ic', args.high_likelihood_incorrect])
        if args.uniform_for_incorrect:
            optional_params.extend(['-unif'])
        if args.segment:
            optional_params.extend(['-seg'])
        if args.num_partitions:
            optional_params.extend(['-p', args.num_partitions])

            
        for i in range(args.num_runs):
            print_banner('Case #%d' % (i+1), ch='-')
            proc = subprocess.Popen(['./synthetic_experiment_subprocess.py',
                                     args.train_kwargs_file,
                                     args.test_kwargs_file,
                                     '-d', args.db_abrv,
                                     '-N', str(1),
                                     '-n', str(args.num_rounds),
                                     '-e', args.exp_name] + optional_params, stderr=subprocess.STDOUT)
            proc.wait()
    except KeyboardInterrupt as ex:
        print("Terminating...\n")
        time.sleep(2)
    finally:
        if not args.test_name:
            with open(args.test_kwargs_file) as f:
                test_kwargs = yaml.load(f)
            test_name = test_kwargs['test_name']
        else:
            test_name = args.test_name
        overall_stats = get_stats(args.exp_name, test_name, db_name)
        save_path = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, 'results', 'collected_overall_%s_%s_%s.log' % (test_name, time.strftime("%Y%m%d-%H%M"), db_name))
        with open(save_path, 'w') as f:
            yaml.dump(overall_stats, stream=f)
        print_banner("All summed up", ch='-')
        pprint(overall_stats)


if __name__ == "__main__":
    synthetic()
