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

    acc_by_class = []
    acc_overall = []
    dgsm_acc_by_class = []
    dgsm_acc_overall = []

    print("Gathering")
    for case_name in sorted(os.listdir(results_dir)):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            print("    " + case_name)
            # Open report.log
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            stats['graphspn'][case_name] = report

            acc_by_class.append(report['_overall_by_class_'])
            acc_overall.append(report['_overall_'])

            graphspn_total = report['_total_inferred_']
            graphspn_correct = report['_total_correct_']
            
            if '_dgsm_results_' in report:
                dgsm_total = report['_dgsm_results_']['_total_cases_']
                dgsm_correct = report['_dgsm_results_']['_total_correct_']
                dgsm_acc_by_class.append(report['_dgsm_results_']['_overall_by_class_'])
                dgsm_acc_overall.append(report['_dgsm_results_']['_overall_'])

    stats['graphspn']['overall'] = np.mean(acc_overall)
    stats['graphspn']['stdev'] = np.std(acc_overall)
    stats['graphspn']['overall_by_class'] = np.mean(acc_by_class)
    stats['graphspn']['stdev_by_class'] = np.std(acc_by_class)

    stats['dgsm']['overall'] = np.mean(dgsm_acc_overall)
    stats['dgsm']['stdev'] = np.std(dgsm_acc_overall)
    stats['dgsm']['overall_by_class'] = np.mean(dgsm_acc_by_class)
    stats['dgsm']['stdev_by_class'] = np.std(dgsm_acc_by_class)

    pprint(stats)

if __name__ == "__main__":
    main()
