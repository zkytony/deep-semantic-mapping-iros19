#!/usr/bin/env python3
#
# Generate result for a DGSM that combines several submodels;
# Specifically, we would like to produce arrays of normalized
# likelihood values.
#
# In this experiment framework, the virtual scans of each graph
# is grouped into the same room. Therefore, when saving the results
# of likelihoods, we save one file per room (i.e. per graph). These
# will later act as input to GraphSPN.
#
# We report the DGSM's classification result per node, and accuracy
# 1) per graph per class,   2) per graph all classes in total,
# 3) all graphs per class,  4) all graphs all classes in total.
#
# Refer to dgsm/result.py:Result:get_classification_results_for_graphs()

import os
import argparse
import deepsm.dgsm.model_results as dgsm_results
import deepsm.util as util
import deepsm.experiments.paths as paths


if __name__ == "__main__":

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

    if what == "DGSM_SAME_BUILDING":
        """
        Configurations:
        "test_case": (str) e.g. '456-7'
        """
        original_real_data_path = os.path.join(paths.path_to_dgsm_dataset_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                                                        args.db_name), 'real_data')
        original_set_defs_path = os.path.join(os.path.dirname(original_real_data_path),
                                              config['test_case'], "set_defs")
        symlink_real_data_path = os.path.join(tmp_data_dir, "real_data")
        symlink_set_defs_path = os.path.join(tmp_data_dir, "set_defs")

        if os.path.exists(symlink_real_data_path):
            os.remove(symlink_real_data_path)
        if os.path.exists(symlink_set_defs_path):
            os.remove(symlink_set_defs_path)
        os.symlink(original_real_data_path, symlink_real_data_path)
        os.symlink(original_set_defs_path, symlink_set_defs_path)

        results_dir = os.path.dirname(paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                                              args.db_name,
                                                                              "CR",
                                                                              config['test_case'].split("-")[0],
                                                                              config['test_case'].split("-")[1]))

    elif what == "DGSM_ACROSS_BUILDINGS":
        """
        Configurations:
        "test_case": (str) e.g. 'Stockholm_Freiburg-Saarbrucken'
        """
        test_building = config['test_case'].split("_")[0]
        train_buildings = sorted(config['test_case'].split("_")[1].split("-"))
        
        original_real_data_path = os.path.join(paths.path_to_dgsm_dataset_across_buildings(util.CategoryManager.NUM_CATEGORIES),
                                               'real_data')
        original_set_defs_path = os.path.join(os.path.dirname(original_real_data_path),
                                              config['test_case'], "set_defs")
        symlink_real_data_path = os.path.join(tmp_data_dir, "real_data")
        symlink_set_defs_path = os.path.join(tmp_data_dir, "set_defs")

        if os.path.exists(symlink_real_data_path):
            os.remove(symlink_real_data_path)
        if os.path.exists(symlink_set_defs_path):
            os.remove(symlink_set_defs_path)
        os.symlink(original_real_data_path, symlink_real_data_path)
        os.symlink(original_set_defs_path, symlink_set_defs_path)

        results_dir = os.path.dirname(paths.path_to_dgsm_result_across_buildings(util.CategoryManager.NUM_CATEGORIES,
                                                                                 "CR",
                                                                                 train_buildings,
                                                                                 test_building))
    classes = []
    for i in range(util.CategoryManager.NUM_CATEGORIES):
        classes.append(util.CategoryManager.category_map(i, rev=True))


    dgsm_args_parser = dgsm_results.create_parser()
    dgsm_args = dgsm_results.parse_args(parser=dgsm_args_parser,
                                        args_list=[tmp_data_dir,
                                                   results_dir,
                                                   '1'])
    dgsm_results.main(args=dgsm_args, trained_classes=classes)
