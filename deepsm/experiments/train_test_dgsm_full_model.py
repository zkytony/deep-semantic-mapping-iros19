#!/usr/bin/env python3

import tensorflow as tf
import os
import argparse
import deepsm.dgsm.train_test_model as dgsm_runner
import deepsm.experiments.paths as paths
from deepsm.util import CategoryManager
from pprint import pprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DGSM and produce test results")
    parser.add_argument('what', type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING, DGSM_ACROSS_BUILDINGS)')
    parser.add_argument('-d', '--db_name', type=str, help='e.g. Freiburg (case-sensitive)')
    parser.add_argument('--config', type=str, help='Quoted string in the form of a Python dictionary,'\
                        'that provides key-value pair for configurations (e.g. "{\'test_case\': \'456-7\'}"',
                        default="{}")
    parser.add_argument('-n', '--num-trials', type=int, help='number of trials', default=1)
    args = parser.parse_args()

    what = args.what
    config = eval(args.config)

    if "category_type" in config:
        CategoryManager.TYPE = config['category_type']
    else:
        CategoryManager.TYPE = "SIMPLE"
    CategoryManager.init()

    dgsm_args = None
    if what == "DGSM_SAME_BUILDING":
        """
        Configurations:
        "test_case": (str) e.g. '456-7'
        "training_params: (str) e.g. '#batch-size 10 #learning-rate 0.01'
             note: all `#` will be replaced as `--`
        """
        original_real_data_path = os.path.join(paths.path_to_dgsm_dataset_same_building(CategoryManager.NUM_CATEGORIES, args.db_name),
                                               'real_data')
        original_set_defs_path = os.path.join(os.path.dirname(original_real_data_path),
                                              config['test_case'], "set_defs")

        for trial_number in range(args.num_trials):
            print("**********Trial %d**********" % trial_number)
            
            results_dir = paths.path_to_dgsm_result_same_building(CategoryManager.NUM_CATEGORIES,
                                                                  args.db_name,
                                                                  "graphs",
                                                                  trial_number,
                                                                  config['test_case'].split("-")[0],
                                                                  config['test_case'].split("-")[1])

            # Print arguments:
            print("============")
            print(" Arguments  ")
            print("============")
            print("Original real_data_path: %s" % original_real_data_path)
            print("Original set_defs_path: %s" % original_set_defs_path)
            print("")
            print("")
            print("Configs:")
            pprint(config)

            training_params = []
            if "training_params" in config:
                training_params = config['training_params'].split()

            dgsm_args_parser = dgsm_runner.create_parser()
            dsgm_args = dgsm_runner.parse_args(parser=dgsm_args_parser,
                                               args_list=[original_set_defs_path,
                                                          original_real_data_path,
                                                          results_dir,
                                                          '1',
                                                          '--graph-test',
                                                          '--building', args.db_name + "_" + config['test_case']] + training_params)
            dgsm_runner.main(args=dsgm_args)
