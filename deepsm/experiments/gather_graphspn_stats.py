#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import re
import argparse
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT
from deepsm.util import plot_roc
from pprint import pprint
from sklearn import metrics
from deepsm.util import CategoryManager

def get_seq_id(case_name):
    return case_name.split("-")[-1]

def get_split_string(building, test_floor):
    if building == "Stockholm":
        floors = {4, 5, 6, 7}
    elif building == "Freiburg":
        floors = {1, 2, 3}
    elif building == "Saarbrucken":
        floors = {1, 2, 3, 4}
    train_floors = "".join(map(str, sorted(floors - {int(test_floor)})))
    return train_floors + "-" + str(test_floor)

def complete_the_report(report):
    """Used for older report.log format"""
    
    # Compute accuracy by class
    accuracy_per_catg = []
    for catg in report:
        if not catg.startswith("_"):
            accuracy_per_catg.append(report[catg][2])
    report['_overall_by_class_'] = float(np.mean(accuracy_per_catg))
    report['_stdev_by_class_'] = float(np.std(accuracy_per_catg))

    if '_dgsm_results_' in report:
        accuracy_per_catg = []
        for catg in report['_dgsm_results_']:
            if not catg.startswith("_"):
                accuracy_per_catg.append(report['_dgsm_results_'][catg][2])
        report['_dgsm_results_']['_overall_by_class_'] = float(np.mean(accuracy_per_catg))
        report['_dgsm_results_']['_stdev_by_class_'] = float(np.std(accuracy_per_catg))


def gather_novelty_results(exp_name, test_case, test_name,
                           results_dir=None, report_fname='novelty.log'):
    if results_dir is None:
        results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, exp_name, "results")

    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load("Stockholm", skip_unknown=True, skip_placeholders=True, single_component=True)

    print("Gathering")
    results = {}
    seqs_floors = {}
    min_no_swap = float('inf')
    for case_name in os.listdir(results_dir):
        if case_name.startswith(test_case) \
           and test_name in case_name:
            print("    " + case_name)

            seq_id = get_seq_id(case_name)
            floor = seq_id.split("_")[0]
            if floor not in results:
                results[floor] = []
                seqs_floors[floor] = []
            seqs_floors[floor].append(seq_id)

            topo_map = dataset.get("Stockholm", seq_id)
            
            # Open novelty.log
            with open(os.path.join(results_dir, case_name, report_fname)) as f:
                novelty = yaml.load(f)
                
            if 'results' in novelty:
                novelty = novelty['results']

            # Normalize
            # for case in novelty:
            #     if type(novelty[case]) == dict:
            #         novelty[case] = novelty[case]['_raw_']
            #     novelty[case] = novelty[case] / len(topo_map.nodes)
            
            no_swap = novelty['_no_swap_']['_normalized_']
            for case in novelty:
                if type(case) == str and case.startswith("_"):
                    novelty[case] = novelty[case]['_normalized_']
                if type(novelty[case]) == dict:
                    novelty[case] = novelty[case]['_normalized_']

                if type(case) != tuple and case.startswith("_"):
                    results[floor].append((1, novelty[case], len(topo_map.nodes), case)) # 1 means not noval
                else:
                    results[floor].append((0, novelty[case], len(topo_map.nodes), case)) # 0 means noval

            pprint(novelty)
            print(len(topo_map.nodes))

    floor = list(sorted(results.keys()))[0]
    print(floor)
    sorted_samples = reversed(sorted(results[floor], key=lambda k: k[1]))
    for s in sorted_samples:
        print(s)
    pprint(seqs_floors)

    pairs = []
    names = []
    for floor in results:
        y_true = [k[0] for k in results[floor]]
        y_score = [k[1] for k in results[floor]]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        pairs.append((fpr,tpr))
        names.append(get_split_string("Stockholm", int(re.search('[0-9]+', floor).group())))
        
    save_path = os.path.join(results_dir, "gathered", "%s-roc.png" % test_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_roc(pairs, savepath=save_path, names=names)
    print("Plotted roc for %s to %s" % (test_name, save_path))


            
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

    results_dir = os.path.join(GRAPHSPN_RESULTS_ROOT, args.exp_name, "results")

    # IF test_case is "Novelty", we gather novelty results and plot roc curve.
    if args.test_case.lower() == "novelty":
        gather_novelty_results(args.exp_name, args.test_case, args.test_name)
        return
    

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

            if "_overall_by_class_" not in report:
                complete_the_report(report)

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
