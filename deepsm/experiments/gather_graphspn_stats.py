#!/usr/bin/env python3

import yaml
import os
import argparse
from deepsm.experiments.common import GRAPHSPN_RESULTS_ROOT
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="Gather stats for graphspn experiments; A test case" \
                                     "is located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")

    args = parser.parse_args()

    results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, "results")

    stats = {'graphspn':{'total':0, 'correct':0, 'overall':0.0},
             'dgsm':{'total':0, 'correct':0, 'overall':0.0}}

    print("Gathering")
    for case_name in os.listdir(results_dir):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            print("    " + case_name)
            # Open report.log
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            # For now, just compute the overall
            graphspn_total = report['_total_inferred_']
            graphspn_correct = report['_total_correct_']
            stats['graphspn']['total'] += graphspn_total
            stats['graphspn']['correct'] += graphspn_correct
            
            if '_dgsm_results_' in report:
                dgsm_total = report['_dgsm_results_']['_total_cases_']
                dgsm_correct = report['_dgsm_results_']['_total_correct_']
                stats['dgsm']['total'] += dgsm_total
                stats['dgsm']['correct'] += dgsm_correct

    if stats['graphspn']['total'] > 0:
        stats['graphspn']['overall'] = stats['graphspn']['correct'] / stats['graphspn']['total']
    if stats['dgsm']['total'] > 0:
        stats['dgsm']['overall'] = stats['dgsm']['correct'] / stats['dgsm']['total']

    pprint(stats)

if __name__ == "__main__":
    main()
