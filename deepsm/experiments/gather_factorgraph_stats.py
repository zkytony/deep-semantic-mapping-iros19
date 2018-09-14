#!/usr/bin/env python3

import argparse
import yaml
import os
import numpy as np
from deepsm.experiments.common import BP_RESULTS_ROOT
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="Gather stats for graphspn experiments; A test case" \
                                     "is located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")

    args = parser.parse_args()

    results_dir = os.path.join(BP_RESULTS_ROOT, args.exp_name, args.test_case + "_" + args.test_name)

    stats = {'factor_graph':{'total':0, 'correct':0, 'overall':0.0}}
    
    acc_by_class = []
    acc_overall = []
    
    print("Gathering")
    for case_name in sorted(os.listdir(results_dir)):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            print("    " + case_name)

            # Open report.log
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            stats['factor_graph'][case_name] = report

            acc_by_class.append(report['_overall_by_class_'])
            acc_overall.append(report['_overall_'])

            fg_total = report['_total_inferred_']
            fg_correct = report['_total_correct_']

            
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            # For now, just compute the overall
            fg_total = report['_total_inferred_']
            fg_correct = report['_total_correct_']
            stats['factor_graph']['total'] += fg_total
            stats['factor_graph']['correct'] += fg_correct
            
    stats['factor_graph']['overall'] = np.mean(acc_overall)
    stats['factor_graph']['stdev'] = np.std(acc_overall)
    stats['factor_graph']['overall_by_class'] = np.mean(acc_by_class)
    stats['factor_graph']['stdev_by_class'] = np.std(acc_by_class)

    pprint(stats)

if __name__ == "__main__":
    main()

