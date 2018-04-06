# Dataset to support DGSM experiments (training and testing).
# This is a dataset of

import pickle
import os
from pprint import pprint

from deepsm.graphspn.tbm.dataset import TopoMapDataset

class DGSMDataset:

    """
    The dataset consists of sequences of polar scans.
    """
    def __init__(self):
        # db_name -> datapath
        self.datapaths = {}

    def add_datapath(self, polar_datapath, db_name):
        """
        polar_datapath (str): path to directory that contains polar scans
            data. The directory should be structured like:
            
            polar_datapath:
                {seq_id}_scans.pkl
        
            The pickle file should be list of scans for that sequence.
        db_name (str): A name for the data in the given path (e.g. Freiburg)
        """
        db_name = db_name.lower()
        if db_name in self.data:
            raise ValueError("%s already added." % db_name)
        self.datapaths[db_name] = polar_datapath

    def seq_path(self, db_name, seq_id):
        return os.path.join(self.datapaths[db_name], seq_id + "_scans.pkl")

    def load_sequences(self, db_names):
        """
        Saves a single pickle file that combines `num_seqs_each` number of
        sequences in each db_name.
        """
        seqdata = []
        for db_name in db_names:
            db_name = db_name.lower()
            for s in sorted(os.listdir(self.datapaths[db_name])):
                fname = os.path.join(self.datapaths[db_name], s)
                with open(fname, 'rb') as f:
                    d = pickle.load(f)
                    seqdata.extend(d)
        return seq_data

    @staticmethod
    def make_dataset(data, outpath):
        with open(os.path.join(outpath, "real_data")) as f:
            pickle.dump(seqdata, f)
        

    def polar_scans_from_graph(self, db_name, seq_id, topo_map):
        """
        Create a sequence of polar scans where each corresponds to a node in the
        topological map graph.
        """
        db_name = db_name.lower()
        with open(self.seq_path(db_name, seq_id)) as f:
            seq_data = pickle.load(f) # seq_data should be list of individual scans with some additional info
        graph_scans = []
        for nid in topo_map.nodes:
            # Find the closest scan; scan is of the format [room_id, room_class, [scan], (x,y)]
            x, y = topo_map.nodes[nid].pose
            closest_scan = min(seq_data, key=lambda s: (s[3][0]-x)**2 + (s[3][1]-y)**2)
            # To comply with the DGSM framework, which groups polar scans by rooms, because we
            # want to test the whole graph together, we should name all scans in the graph using
            # the same room. We will just use the {db_name}_{seq_id} as the room name. Additionally, we
            # add node id at the end of the scan for later result analysis.
            graph_scan = "%s_%s" % (db_name, seq_id) + closest_scan[1:] + [nid]
            graph_scans.append(graph_scan)
        return graph_scans


    @staticmethod
    def make_set_defs(db_floor_plans, db_floors_training, db_seqs_testing):
        """Create a set_defs dictionary.

        `db_floor_plans`: map from db_name to {floor -> room_id}; The room_id here is not prefixed by db_name
        `db_floors_training`: map from db_name to list of floors, where
              the rooms will be used for trianing;
        `db_seqs_testing`: map from db_name to list of sequence_ids, which is
              the sequence whose graph will be used in testing."""
        set_defs = {'train_rooms_1': [], 'test_rooms_1': []}
        for db_name in db_floors_training:
            for floor in db_floors_training[db_name]:
                for room_id in db_floor_plans[floor]:
                    set_defs['train_rooms_1'].append(db_name.lower() + "_" + room_id)
        for db_name in db_seqs_testing:
            for seq_id in db_seqs_testing[db_name]:
                set_defs['test_rooms_1'].append("%s_%s" % (db_name.lower(), seq_id))
        return set_defs

if __name__ == "__main__":
    # Some testing
    TOPO_MAP_DB_ROOT = "/home/zkytony/Documents/thesis/experiments/spn_topo/experiments/data/topo_map"
    datapath1 = "/home/zkytony/Documents/thesis/experiments/Data/polar_scans"
    outpath = "./"
        
    dgsm_dataset = DGSMDataset()
    dgsm_dataset.add_datapath(datapath1, "small")
    seq_data = dgsm_dataset.load_sequences(["small"])

    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    topo_dataset.load("Stockholm", skip_unknown=True)
    topo_map = topo_dataset.get("Stockholm", "floor7_cloudy_b")

    graph_scans = dgsm_dataset.polar_scans_from_graph("small", "floor7_cloudy_1", topo_map)
    seq_data.extend(graph_scans)
    dgsm_dataset.make_dataset(seq_data, outpath)
    set_defs = dgsm_dataset.make_set_defs({"small": {"floor4":["4-CR-1", "4-1PO-2"],
                                                     "floor7":["4-CR-1", "4-1PO-2"]}},
                                          {"small": ["floor4"]},
                                          {"small": ["floor7_cloudy_1"]}}
    pprint(set_defs)
