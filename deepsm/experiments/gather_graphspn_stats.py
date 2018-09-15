#!/usr/bin/env python3

import numpy as np
import yaml
import os
import argparse
from deepsm.experiments.common import GRAPHSPN_RESULTS_ROOT
from pprint import pprint

def get_seq_id(case_name):
    return case_name.split("-")[-1]

def main():
    parser = argparse.ArgumentParser(description="Gather stats for graphspn experiments; A test case" \
                                     "is located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")

    args = parser.parse_args()

    results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, "results")

    stats = {'graphspn':{'_all_':{'_overall_':0.0}},
             'dgsm':{'_all_':{'_overall_':0.0}}}

    acc_by_class = []
    acc_overall = []
    dgsm_acc_by_class = []
    dgsm_acc_overall = []

    print("Gathering")
    for case_name in sorted(os.listdir(results_dir)):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            print("    " + case_name)

            seq_id = get_seq_id(case_name)
            floor = seq_id.split("_")[0]
            if floor not in stats['graphspn']:
                stats['graphspn'][floor] = {}
            
            # Open report.log
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            stats['graphspn'][floor][seq_id] = report

            acc_by_class.append(report['_overall_by_class_'])
            acc_overall.append(report['_overall_'])

            if '_dgsm_results_' in report:
                dgsm_acc_by_class.append(report['_dgsm_results_']['_overall_by_class_'])
                dgsm_acc_overall.append(report['_dgsm_results_']['_overall_'])

    stats['graphspn']['_per_floor_'] = {}
    stats['dgsm']['_per_floor_'] = {}
    for floor in stats['graphspn']:
        if floor.startswith("_"):
            continue
        fl_acc_by_class = []
        fl_acc_overall = []
        fl_dgsm_acc_by_class = []
        fl_dgsm_acc_overall = []
        for seq_id in stats['graphspn'][floor]:
            fl_acc_by_class.append(stats['graphspn'][floor][seq_id]['_overall_by_class_'])
            fl_acc_overall.append(stats['graphspn'][floor][seq_id]['_overall_'])

            fl_dgsm_acc_by_class.append(stats['graphspn'][floor][seq_id]['_dgsm_results_']['_overall_by_class_'])
            fl_dgsm_acc_overall.append(stats['graphspn'][floor][seq_id]['_dgsm_results_']['_overall_'])
            
        stats['graphspn']['_per_floor_'][floor] = {
            'overall': np.mean(fl_acc_overall),
            'stdev':  np.std(fl_acc_overall),
            'overall_by_class': np.mean(fl_acc_by_class),
            'stdev_by_class': np.std(fl_acc_by_class),
        }

        stats['dgsm']['_per_floor_'][floor] = {
            'overall': np.mean(fl_dgsm_acc_overall),
            'stdev':  np.std(fl_dgsm_acc_overall),
            'overall_by_class': np.mean(fl_dgsm_acc_by_class),
            'stdev_by_class': np.std(fl_dgsm_acc_by_class),
        }
        
    stats['graphspn']['_all_']['_overall_'] = np.mean(acc_overall)
    stats['graphspn']['_all_']['_stdev_'] = np.std(acc_overall)
    stats['graphspn']['_all_']['_overall_by_class_'] = np.mean(acc_by_class)
    stats['graphspn']['_all_']['_stdev_by_class_'] = np.std(acc_by_class)

    stats['dgsm']['_all_']['_overall_'] = np.mean(dgsm_acc_overall)
    stats['dgsm']['_all_']['_stdev_'] = np.std(dgsm_acc_overall)
    stats['dgsm']['_all_']['_overall_by_class_'] = np.mean(dgsm_acc_by_class)
    stats['dgsm']['_all_']['_stdev_by_class_'] = np.std(dgsm_acc_by_class)

    pprint(stats)
    print("GraphSPN per floor:")
    pprint(stats['graphspn']['_per_floor_'])
    print("GraphSPN overall:")
    pprint(stats['graphspn']['_all_'])
    
    print("DGSM per floor:")
    pprint(stats['dgsm']['_per_floor_'])
    print("DGSM overall:")
    pprint(stats['dgsm']['_all_'])

if __name__ == "__main__":
    main()
