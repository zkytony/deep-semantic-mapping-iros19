#!/usr/bin/env python3

import argparse
import yaml
import os
import numpy as np
from deepsm.experiments.common import BP_RESULTS_ROOT
from deepsm.util import CategoryManager
from gather_graphspn_stats import gather_novelty_results
from pprint import pprint

def get_seq_id(case_name):
    return case_name.split("-")[-1]


def complete_the_report(report):
    """Used for older report.log format"""
    
    # Compute accuracy by class
    accuracy_per_catg = []
    for catg in report:
        if not catg.startswith("_"):
            accuracy_per_catg.append(report[catg][2])
    report['_overall_by_class_'] = float(np.mean(accuracy_per_catg))
    report['_stdev_by_class_'] = float(np.std(accuracy_per_catg))

def main():
    parser = argparse.ArgumentParser(description="Gather stats for graphspn experiments; A test case" \
                                     "is located at [ExperimentName]/[TestCase]_timestamp_[test-name]_seq_id")
    parser.add_argument("exp_name", type=str, help="Name of experiment. e.g. GraphSPNToyExperiment")
    parser.add_argument("test_case", type=str, help="Name of test. e.g. Classification")
    parser.add_argument("test_name", type=str, help="Name of test. e.g. mytest")
    parser.add_argument("-k", "--category-type", type=str, default="SEVEN") 

    args = parser.parse_args()

    CategoryManager.TYPE = args.category_type
    CategoryManager.init()


    results_dir = os.path.join(BP_RESULTS_ROOT, args.exp_name, args.test_case + "_" + args.test_name)

    # IF test_case is "Novelty", we gather novelty results and plot roc curve.
    if args.test_case.lower() == "novelty":
        gather_novelty_results(args.exp_name, args.test_case, args.test_name,
                               results_dir=results_dir, report_fname='report.log')
        return

    stats = {'factor_graph':{'_all_':{'_overall_':0.0}}}

    acc_by_class = []
    acc_overall = []
    
    print("Gathering")
    for case_name in sorted(os.listdir(results_dir)):
        if case_name.startswith(args.test_case) \
           and args.test_name in case_name:
            print("    " + case_name)

            seq_id = get_seq_id(case_name)
            floor = seq_id.split("_")[0]
            if floor not in stats['factor_graph']:
                stats['factor_graph'][floor] = {}
            
            # Open report.log
            with open(os.path.join(results_dir, case_name, 'report.log')) as f:
                report = yaml.load(f)

            if "_overall_by_class_" not in report:
                complete_the_report(report)

            stats['factor_graph'][floor][seq_id] = report

            acc_by_class.append(report['_overall_by_class_'])
            acc_overall.append(report['_overall_'])

    stats['factor_graph']['_per_floor_'] = {}
    for floor in stats['factor_graph']:
        if floor.startswith("_"):
            continue
        fl_acc_by_class = []
        fl_acc_overall = []
        for seq_id in stats['factor_graph'][floor]:
            fl_acc_by_class.append(stats['factor_graph'][floor][seq_id]['_overall_by_class_'])
            fl_acc_overall.append(stats['factor_graph'][floor][seq_id]['_overall_'])

        stats['factor_graph']['_per_floor_'][floor] = {
            'overall': np.mean(fl_acc_overall),
            'stdev':  np.std(fl_acc_overall),
            'overall_by_class': np.mean(fl_acc_by_class),
            'stdev_by_class': np.std(fl_acc_by_class),
        }
        
    stats['factor_graph']['_all_']['_overall_'] = np.mean(acc_overall)
    stats['factor_graph']['_all_']['_stdev_'] = np.std(acc_overall)
    stats['factor_graph']['_all_']['_overall_by_class_'] = np.mean(acc_by_class)
    stats['factor_graph']['_all_']['_stdev_by_class_'] = np.std(acc_by_class)

    pprint(stats)
    print("Factor_Graph per floor:")
    pprint(stats['factor_graph']['_per_floor_'])
    print("Factor_Graph overall:")
    pprint(stats['factor_graph']['_all_'])

if __name__ == "__main__":
    main()

