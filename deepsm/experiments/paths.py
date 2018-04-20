# Manages all paths

import os
from deepsm.experiments.common import DGSM_DB_ROOT, DGSM_RESULTS_ROOT

"""
Dataset path:
dataset_root/
    experiment_data/
        Nclasses/
            samebuilding/
                stockholm/
                    real_data    [[{building}_{room_id} ... ], ... [{building#}_{seq_id} ... ] ...]
                    456-7/
                        set_defs
                    ...
                ... (freiburg, saarbrucken)

Result path
results_root/
    Nclasses/
        samebuilding/
            stockholm/
                456-7/
                    1PO/
                    CR/
                    ...
                    graphs/
                        {graph_id}_likelihoods.json   (graph_id == {building#}_{seq_id})
                ...
            ...
"""


def path_to_polar_scans(building, dim="56x21", seq_id=None):
    if seq_id is not None:
        return os.path.join(DGSM_DB_ROOT,
                            "polar_scans", "polar_scans_%s" % building.lower(),
                            dim, "%s.pkl" % seq_id)
    else:
        return os.path.join(DGSM_DB_ROOT,
                            "polar_scans", "polar_scans_%s" % building.lower(), dim)

    
def path_to_dgsm_dataset_same_building(num_categories,
                                       building_name):
    return os.path.join(DGSM_DB_ROOT,
                        "experiment_data",
                        "%dclasses" % num_categories,
                        "same_building",
                        building_name)

def path_to_dgsm_set_defs_same_building(dgsm_dataset_same_building_path,
                                        train_floors,
                                        test_floor):
    """
    Args:
       train_floors (str): For example, 456 means floor4, floor5, floor6 (or seq#)
       test_floor (str): For example 7 means floor7
    """
    return os.path.join(dgsm_dataset_same_building_path,
                        "%s-%s" % (train_floors, test_floor))


def path_to_dgsm_result_same_building(num_categories,
                                      building_name,
                                      submodel_class,
                                      train_floors,
                                      test_floor):
    return os.path.join(DGSM_RESULTS_ROOT,
                        "%dclasses" % num_categories,
                        "same_building",
                        building_name,
                        "%s-%s" % (train_floors, test_floor),
                        submodel_class)
