#!/usr/bin/env python3

import numpy as np
import os
import argparse
import deepsm.util as util
import deepsm.experiments.paths as paths
import yaml
from pprint import pprint

def same_building_results(db_name, config):
    """
    Configurations:
    "test_case": (str) e.g. '456-7'
    "trial": (int) e.g. 4    (if left out, compute average and standard deviation over all trials)

    Returns: A dictionary that contains "avg", "std", and "class_results"
    as its keys. If 'trial' is present in config, then the `std` represents
    standard deviation across graphs in that trial. Otherwise, it represents
    standard deviation across trials.
    """
    trial = 0 if "trial" not in config else config['trial']
    
    results_dir = paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                          db_name,
                                                          "graphs",
                                                          trial,
                                                          config['test_case'].split("-")[0],
                                                          config['test_case'].split("-")[1])


    if "trial" in config:
        # Read the results.yaml in the specified trial
        with open(os.path.join(results_dir, "results.yaml")) as f:
           stats = yaml.load(f)

        # Get standard deviation across graphs
        accuracies = []
        for key in stats:
            if key.startswith(db_name.lower()):
                accuracies.append(stats[key]['accuracy'])
        std = np.std(accuracies)
        return {'avg': stats['accuracy'],
                'std': std,
                'class_results': stats['class_results']}

    else:
        # Read results.yaml in all trials.
        accuracies = []
        class_results = {}
        for root, dirs, files in os.walk(os.path.dirname(results_dir)):
            for name in files:
                if name.endswith("results.yaml"):
                    with open(os.path.join(root, name)) as f:
                        stats = yaml.load(f)
                    accuracies.append(stats['accuracy'])
                    for catg in stats['class_results']:
                        if catg not in class_results:
                            class_results[catg] = []
                        class_results[catg].append(stats['class_results'][catg][-1])
        for catg in class_results:
            class_results[catg] = {'avg': np.mean(class_results[catg]),
                                   'std': np.std(class_results[catg])}
        return {'avg': np.mean(accuracies),
                'std': np.std(accuracies),
                'class_results': class_results}
                    
 
def main():
    parser = argparse.ArgumentParser(description="Get DGSM test results for graphs")
    parser.add_argument('what', type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')
    parser.add_argument('-d', '--db_name', type=str, help='e.g. Freiburg (case-sensitive)')
    parser.add_argument('--config', type=str, help='Quoted string in the form of a Python dictionary,'\
                        'that provides key-value pair for configurations (e.g. "{\'test_case\': \'456-7\'}"',
                        default="{}")
    args = parser.parse_args()

    config = eval(args.config)
    what = args.what
    
    # We need the real_data and set_defs to be in the ./tmp_experiment_dgsm" directory
    tmp_data_dir = ".tmp_experiment_dgsm"
    os.makedirs(tmp_data_dir, exist_ok=True)

    if "category_type" in config:
        util.CategoryManager.TYPE = config['category_type']
    else:
        util.CategoryManager.TYPE = "SIMPLE"
    util.CategoryManager.init()

    if what == "DGSM_SAME_BUILDING":
        stats = same_building_results(args.db_name, config)
        pprint(stats)


if __name__ == "__main__":
    main()
