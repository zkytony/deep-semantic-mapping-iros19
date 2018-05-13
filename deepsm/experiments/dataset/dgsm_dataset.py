# Dataset to support DGSM experiments (training and testing).
# This is a dataset of

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
import pickle
import os
import re
import yaml
from pprint import pprint
import numpy as np

from deepsm.graphspn.tbm.dataset import TopoMapDataset
import deepsm.util as util
from deepsm.util import ColdDatabaseManager
from deepsm.experiments.common import DGSM_DB_ROOT, TOPO_MAP_DB_ROOT, COLD_ROOT

class DGSMDataset:

    """
    The dataset consists of sequences of polar scans.
    """
    def __init__(self):
        """Initialize DGSMDataset."""
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
        if db_name in self.datapaths:
            raise ValueError("%s already added." % db_name)
        self.datapaths[db_name] = polar_datapath

    def seq_path(self, db_name, seq_id):
        db_name = db_name.lower()
        return os.path.join(self.datapaths[db_name], seq_id + "_scans.pkl")

    def load_sequences(self, db_names, max_seqs_per_floor=-1):
        """
        Returns a single list that combines scans of all sequences in each db_name
        """
        seq_data = []
        for db_name in db_names:
            db_name = db_name.lower()
            seqs_per_floor = {}
            for s in sorted(os.listdir(self.datapaths[db_name])):
                floor = re.search("(seq|floor)[1-9]+", s).group()
                if floor not in seqs_per_floor:
                    seqs_per_floor[floor] = 0
                if max_seqs_per_floor > 0 and seqs_per_floor[floor] >= max_seqs_per_floor:
                    continue
                fname = os.path.join(self.datapaths[db_name], s)
                with open(fname, 'rb') as f:
                    d = pickle.load(f)
                    seq_data.extend(d)
                    seqs_per_floor[floor] += 1
        return seq_data

    def load_one_sequence(self, db_name, seq_id):
        db_name = db_name.lower()
        with open(self.seq_path(db_name, seq_id), 'rb') as f:
            seq_data = pickle.load(f) # seq_data should be list of individual scans with some additional info
        return seq_data

    
    def polar_scans_from_graph(self, db_name, seq_id, seq_data, topo_map):
        """
        Create a sequence of polar scans where each corresponds to a node in the
        topological map graph. NOTE: Placeholders do not have matched scans.
        """
        db_name = db_name.lower()
        graph_scans = []
        # First, group the scans into room classes
        seq_data_grouped = {}
        for vscan in seq_data:
            room_class = vscan[1]
            if room_class not in seq_data_grouped:
                seq_data_grouped[room_class] = []
            seq_data_grouped[room_class].append(vscan)
        
        for nid in topo_map.nodes:
            # Find the closest scan with THE SAME CLASS;
            #   scan is of the format [room_id, room_class, [scan], (x,y)]
            if not topo_map.nodes[nid].placeholder:
                x, y = topo_map.nodes[nid].pose
                node_class = topo_map.nodes[nid].label
                if node_class not in seq_data_grouped:
                    raise ValueError("%s is not an expected class in %s! Expected classes %s" % (node_class, seq_id, list(seq_data_grouped.keys())))
                closest_scan = min(seq_data_grouped[node_class], key=lambda s: (s[3][0]-x)**2 + (s[3][1]-y)**2)
                # To comply with the DGSM framework, which groups polar scans by rooms, because we
                # want to test the whole graph together, we should name all scans in the graph using
                # the same room. We will just use the {db_name}_{seq_id} as the room name. Additionally, we
                # add node id {nid} at the end of the scan for later result analysis.
                graph_scan = ["%s_%s" % (db_name, seq_id)] + closest_scan[1:] + [nid]
                assert graph_scan[1] == node_class, "Room class does not match! %s != %s" % (graph_scan[1], room_class)
                graph_scans.append(graph_scan)
        return graph_scans


    def visualize_graph_scans(self,
                              ax,
                              graph_scans,
                              topo_map,
                              canonical_map_yaml_path,
                              img=None,
                              **kwargs):
        """Draw topological map and mark the virtual scans that mapped to each node
        
        kwargs are the arguments for TopologicalMap:visualize() method.
        """
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
        topo_map.visualize(ax, canonical_map_yaml_path=canonical_map_yaml_path, consider_placeholders=True,
                           img=img, **kwargs)
        # For each vscan in graph_scan, use the stored node id to reference the "true location"
        for vscan in graph_scans:
            nid = vscan[-1]
            vscan_pose = vscan[-2]
            util.plot_dot(ax,
                          vscan_pose[0],
                          vscan_pose[1],
                          map_spec, img,
                          dotsize=10,
                          color='orange',
                          zorder=2, linewidth=5.0, edgecolor='black')
            place = topo_map.nodes[nid]
            util.plot_line(ax,
                           place.pose,
                           vscan_pose,
                           map_spec,
                           img,
                           linewidth=5,
                           color='black',
                           zorder=1)


    def plot_scans_on_map(self, ax, scans, canonical_map_yaml_path, img=None, **kwargs):
        """Plot the locations of each scan on the map. Assume tha map has already been plotted"""
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
        prev_vscan_pose = None
        for vscan in scans:
            vscan_pose = vscan[3]
            util.plot_dot(ax,
                          vscan_pose[0],
                          vscan_pose[1],
                          map_spec, img,
                          dotsize=2,
                          color='blue',
                          zorder=4, linewidth=0.0)
            if prev_vscan_pose is not None:
                util.plot_line(ax,
                               prev_vscan_pose,
                               vscan_pose,
                               map_spec,
                               img,
                               linewidth=2,
                               color='blue',
                               zorder=4)
            prev_vscan_pose = vscan_pose
        

    @staticmethod
    def make_set_defs(db_floor_plans, db_floors_training, db_seqs_testing):
        """Create a set_defs dictionary. We will only have one subset (1)

        `db_floor_plans`: map from db_name to {floor -> [room_id]}; The room_id here is NOT prefixed by db_name
        `db_floors_training`: map from db_name to list of floors, where
              the rooms will be used for trianing;
        `db_seqs_testing`: map from db_name to list of sequence_ids, which is
              the sequence whose graph will be used in testing."""
        set_defs = {'train_rooms_1': [], 'test_rooms_1': []}
        for db_name in db_floors_training:
            for floor in db_floors_training[db_name]:
                for room_id in db_floor_plans[db_name][floor]:
                    set_defs['train_rooms_1'].append(db_name.lower() + "_" + room_id)
        for db_name in db_seqs_testing:
            for seq_id in db_seqs_testing[db_name]:
                set_defs['test_rooms_1'].append("%s_%s" % (db_name.lower(), seq_id))
        # Known categories are everything but 'UN' class, which represents novel.
        set_defs['known_categories'] = util.CategoryManager.known_categories()
        set_defs['novel_categories'] = util.CategoryManager.novel_categories()
        return set_defs

    @staticmethod
    def make_dataset(scans):
        """
        Dump the data as a single pickle file, with classes mapped to the canonical
        category under the room class scheme defined in deepsm.util.CategoryManager

        Args:
            scans (list): list of scans of format [room_id, room_class, np.array(...), ?(x, y), ?nid]
        """
        # Even if canonical_class is unknown, it shouldn't be a problem because
        # DGSM submodel will only pick the samples that have the same class for training.
        for vscan in scans:
            canonical_class = util.CategoryManager.canonical_category(vscan[1], checking=True)  # data[i][1] is the room class
            vscan[1] = canonical_class
        return scans

    @staticmethod
    def filter_scans_by_distance(scans, distance=0.5):
        """
        Reduces scan frequency by filtering out scans that are too close to the previous
        one (within `distance`); scans should be ordered by order of occurrence, and scans
        in the same room should be next to each other.

        distance (float): minimum distance (in meter) between two scans that remain after filtering
        """
        prev_scan = scans[0]        
        final_scans = [prev_scan]
        i = 1
        while i < len(scans):
            cur_scan = scans[i]
            # If room class are different, we reassign prev_scan
            if cur_scan[0] != prev_scan[0]:
                prev_scan = cur_scan
                continue
            scans_dist = np.linalg.norm(np.array(prev_scan[3])-np.array(cur_scan[3]))
            if scans_dist > distance:
                final_scans.append(cur_scan)
                prev_scan = cur_scan
            i += 1
        return final_scans


#---------------------------------------------------------------------------------
def test(VISUALIZE=False):
    # Testing. DO NOT DELETE.
    outpath = "tmp_tests"
    datapath1 = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/dataset/dgsm/polar_scans/polar_scans_small"
    
    ColdMgr = ColdDatabaseManager("Stockholm", COLD_ROOT)

    dgsm_dataset = DGSMDataset()
    dgsm_dataset.add_datapath(datapath1, "small")
    
    topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    topo_dataset.load("Stockholm", skip_unknown=True)
    topo_map = topo_dataset.get("Stockholm", "floor7_cloudy_b")
 
    # Test creating a dataset of vscans corresponding to topological graph nodes
    floor7cb_scans = dgsm_dataset.load_one_sequence("small", "floor7_cloudy_b")
    graph_scans = dgsm_dataset.polar_scans_from_graph("small", "floor7_cloudy_b", floor7cb_scans, topo_map)
    if VISUALIZE:
        ax = plt.gca()
        # Visualize the graph scans
        dgsm_dataset.visualize_graph_scans(ax,
                                           graph_scans,
                                           topo_map,
                                           ColdMgr.groundtruth_file("floor7", 'map.yaml'))
        dgsm_dataset.plot_scans_on_map(ax, floor7cb_scans, ColdMgr.groundtruth_file("floor7", 'map.yaml'))
        plt.show()

    seq_data = dgsm_dataset.load_sequences(["small"])
    seq_data.extend(graph_scans)
    
    real_data = dgsm_dataset.make_dataset(seq_data)
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "real_data"), 'wb') as f:
        pickle.dump(real_data, f)
    
    set_defs = dgsm_dataset.make_set_defs({"small": {"floor4":["4-CR-1", "4-1PO-2"],
                                                     "floor7":["4-CR-1", "4-1PO-2"]}},
                                          {"small": ["floor4"]},
                                          {"small": ["floor7_cloudy_b"]})
    pprint(set_defs)
    # Save set_defs to the same output directory
    with open(os.path.join(outpath, "set_defs"), 'wb') as f:
        pickle.dump(set_defs, f)
        print("set_defs saved to %s" % os.path.join(outpath, "set_defs"))
    
if __name__ == "__main__":
    test(VISUALIZE=True)
