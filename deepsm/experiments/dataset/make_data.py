# Create datasets for DGSM training and testing on a single floor

import re
import pickle
import os
import argparse
from deepsm.experiments.common import DGSM_DB_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT, COLD_ROOT
from deepsm.experiments.dataset.dgsm_dataset import DGSMDataset
import deepsm.experiments.paths as paths
from deepsm.util import CategoryManager

def get_floor_plans(rooms):
    """rooms, a dictionary loaded from labels.json in groundtruth"""
    # Process rooms info
    floor_plans = {}
    for room_id in rooms:
        floor = int(room_id.split("-")[0])
        if floor not in floor_plans:
            floor_plans[floor] = []
        floor_plans[floor].append(room_id)
    return floor_plans

def get_db_info(db_name):
    db_info = {'floors':{}, 'rooms':{}}
    # floors
    for floor in os.listdir(os.path.join(GROUNDTRUTH_ROOT, db_name, "groundtruth")):
        floor_prefix = re.search("(seq|floor)", floor).group()
        floor_number = int(re.search("[1-9]+", floor).group())
        if "floor_prefix" not in db_info:
            db_info['floor_prefix'] = floor_prefix
        if db_info['floor_prefix'] != floor_prefix:
            raise ValueError("Got floor prefix: %s, Expecting %s" % (floor_prefix, db_info['floor_prefix']))
        db_info['floors'][floor_number] = []
        #seqs
        for seq_id in os.listdir(os.path.join(COLD_ROOT, db_name, "data")):
            db_info['floors'][floor_number].append(seq_id)
        #rooms
        with open(os.path.join(GROUNDTRUTH_ROOT, db_name, floor, "labels.json")) as f:
            rooms = json.load(f)
        db_info['rooms'][floor_number] = rooms
    return db_info
    

def create_datasets_same_building(db_name, db_info, dim="56x21"):
    """
    Given DB name, produce
    (1) data for training DGSM (scans of N-1 floors) (N is total number of floors)
    (2) data for testing DGSM (scans of remaining floor)
    (3) topo maps for testing GraphSPN on the same sequences of the remaining floor

    Args:

    db_name (str): e.g. Stockholm (case-sensitive)
    db_info (dict):
       'floors': {floor# -> [..., seq_id, ...]}  (Note: floor# is a number)
       'floor_prefix': (str) (e.g. "seq" or "floor")
       'rooms': {floor# -> dictionary (loaded from labels.json in groundtruth)}
    """
    # Initialize dataset objects
    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dgsm_dataset = DGSMDataset()

    # Load data
    topo_dataset.load(db_name, skip_unknown=True)
    dgsm_dataset.add_datapath(paths.path_to_polar_scans(db_name, dim=dim), db_name)
    scans = dgsm_dataset.load_sequences([db_name]) # list of scans

    # Load graph scans and create set_defs for each floor combination
    all_set_defs = {}

    # Path to where the data will be stored for this db_name
    db_data_path = paths.path_to_dgsm_dataset_same_building(CategoryManager.NUM_CATEGORIES, db_name)
    os.makedirs(db_data_path, exist_ok=True)

    floors = set(db_info['floors'])
    for fl in floors:
        # Print info
        floor_testing = fl
        floors_training = sorted(list(floors - {floor_testing}))
        print("Test floor: %d" % floor_testing)
        print("Train floors: %s" % floors_training)
        
        for seq_id in db_info['floors'][fl]:
            print("    Adding topo map scans (%s)" % seq_id)
            seq_scans = dgsm_dataset.load_one_sequence(db_name, seq_id)
            topo_map = topo_dataset.get(db_name, seq_id)            
            topo_map_scans = dgsm_dataset.polar_scans_from_graph(db_name + str(fl),  # e.g. stockholm7
                                                                 seq_id,
                                                                 seq_scans,
                                                                 topo_map)
            scans.extend(topo_map_scans)
        # Now create set_defs where fl is the test floor
        print("    Creating set_defs")
        seqs_testing = db_info['floors'][fl]
        floor_plan = get_floor_plans(db_info['rooms'][fl])
        set_defs = DGSMDataset.make_set_defs({db_name: floor_plan},
                                             {db_name: floors_training},
                                             {db_name: seqs_testing})
        # save set_defs
        path_to_set_defs = paths.path_to_dgsm_set_defs_same_building(db_data_path,
                                                                     "".join(floors_training),
                                                                     floor_testing)
        os.makedirs(path_to_set_defs, exist_ok=True)
        with open(os.path.join(path_to_set_defs, "set_defs"), 'wb') as f:
            pickle.dump(set_defs, f)
            print("    set_defs saved to %s/set_defs" % path_to_set_defs)
    # Now save real_data
    with open(os.path.join(db_data_path, "real_data"), 'wb') as f:
        pickle.dump(scans, f)
        print("real_data saved to %s/real_data" % db_data_path)
    print("Done!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data for experiments")
    parser.add_argument("db_name", type=str, help='e.g. Freiburg')
    parser.add_argument("what", type=str, help='what data you want to make available constants: (DGSM_SAME_BUILDING)')
    parser.add_argument("--dim", type=str, help='Dimension of polar scans. AxB where A and B are numbers.', default="56x21")
    args = parser.parse_args()

    what = args.what
    if what == "DGSM_SAME_BUILDING":
        db_info = get_db_info(args.db_name)
        create_datasets_same_building(db_name, db_info, dim=args.dim)