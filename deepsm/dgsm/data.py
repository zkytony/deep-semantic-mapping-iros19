import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pprint
from enum import Enum
from deepsm.util import CategoryManager
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.experiments.common import TOPO_MAP_DB_ROOT

pp = pprint.PrettyPrinter(indent=4)


class Data:

    class OccupancyVals(Enum):
        """How many occupancy values to consider."""
        TWO = 1  # 2 values and unknown
        THREE = 2  # 3 values, unknown is a value

    def __init__(self, num_angle_cells,
                 radius_min, radius_max, radius_factor):
        self._num_angle_cells = num_angle_cells
        self._radius_min = radius_min
        self._radius_max = radius_max
        self._radius_factor = radius_factor
        self._calculate_polar_cells()

    @property
    def num_radius_cells(self):
        return self._num_radius_cells

    @property
    def num_angle_cells(self):
        return self._num_angle_cells

    @property
    def occupancy_vals(self):
        return self._occupancy_vals

    @property
    def training_scans(self):
        return self._training_scans
    
    @property
    def training_labels(self):
        return self._training_labels

    @property
    def testing_scans(self):
        return self._testing_scans

    @property
    def testing_labels(self):
        return self._testing_labels

    @property
    def testing_footprint(self):
        """testing_footprint stores the room id, room category, and pose of
        each scan sample in the testing data."""
        return self._testing_footprint

    @property
    def all_scans(self):
        return self._all_scans

    @property
    def masked_scans(self):
        return self._masked_scans

    @property
    def submodel_class(self):
        return self._submodel_class

    @property
    def data(self):
        return self._data

    @property
    def train_rooms(self):
        return self._train_rooms

    @property
    def test_rooms(self):
        return self._test_rooms

    @property
    def novel_classes(self):
        return self._novel_classes

    @property
    def subset(self):
        return self._subset

    def _calculate_polar_cells(self):
        self._angles = np.linspace(-180, 180, self._num_angle_cells + 1)
        r = self._radius_min
        self._radiuses = [r]
        v = 0.04
        while r < self._radius_max:
            r = r + v
            self._radiuses.append(r)
            v *= self._radius_factor
        self._radiuses = np.array(self._radiuses)
        self._num_radius_cells = len(self._radiuses) - 1

    def load(self, set_defs_path, data_path):
        try:
            with open(data_path, 'rb') as f:
                self._data = pickle.load(f)
            with open(set_defs_path, 'rb') as f:
                self._set_defs = pickle.load(f)
        except Exception as e:
            print("\nERROR: Cannot load data: %s" % e)
            sys.exit(1)
        # Check if sample size matches
        if not (self._data[0][2].size ==
                self._num_angle_cells * self._num_radius_cells):
            print("\nERROR: Data parameters do not match!")
            sys.exit(1)

    def process(self, subset, occupancy_vals):
        self._subset = subset
        self._occupancy_vals = occupancy_vals

        # Get relevant data samples
        self._train_rooms = self._set_defs['train_rooms_' + str(self._subset)]
        self._test_rooms = self._set_defs['test_rooms_' + str(self._subset)]
        self._novel_classes = self._set_defs['novel_categories']

        graph_ids = set()
        training_scans = []
        training_labels = []
        training_scans_graph = []  # used to produce likelihoods matching with graph nodes
        training_labels_graph = []
        training_footprint_graph = []
        testing_scans = []
        testing_labels = []
        testing_footprint = []
        for i in self._data:
            rid = i[0]
            rcat = i[1]
            rnum = CategoryManager.category_map(rcat, checking=True)
            scan = i[2]
            if "#" in rid:
                room_id, seq_id = rid.split("#")[0], rid.split("#")[1]
            
                if room_id in self._train_rooms and rnum != CategoryManager.CAT_MAP['UN']:
                    if type(i[-2]) == str and i[-2] == "graph_node":
                        training_scans_graph.append(scan)
                        training_labels_graph.append(rnum)
                        db_name = room_id.split("_")[0]
                        graph_id = db_name + "_" + seq_id # according to conevntion specified in experiments/paths.py
                        graph_ids.add(graph_id)
                        training_footprint_graph.append([graph_id, rcat, i[-1]])
                    else:
                        training_scans.append(scan)
                        training_labels.append(rnum)
            else:
                if rid in self._test_rooms and rnum != CategoryManager.CAT_MAP['UN']:
                    testing_scans.append(scan)
                    testing_labels.append(rnum)
                    testing_footprint.append([rid, rcat, i[-1]])

        pprint.pprint(graph_ids)

        training_scans = np.vstack(training_scans)
        training_scans_graph = np.vstack(training_scans_graph)
        training_labels = np.vstack(training_labels)
        training_labels_graph = np.vstack(training_labels_graph)
        testing_scans = np.vstack(testing_scans)
        testing_labels = np.vstack(testing_labels)
        all_scans = np.vstack([i[2] for i in self._data])

        self._training_labels = training_labels
        self._training_labels_graph = training_labels_graph
        self._training_footprint_graph = training_footprint_graph
        self._testing_labels = testing_labels
        self._testing_footprint = testing_footprint

        # Modify to 3 occupancy vals
        if self._occupancy_vals == Data.OccupancyVals.THREE:
            self._training_scans = np.copy(training_scans)
            self._training_scans[training_scans == -1] = 0
            self._training_scans[training_scans == 0] = 1
            self._training_scans[training_scans == 1] = 2
            self._training_scans_graph = np.copy(training_scans_graph)
            self._training_scans_graph[training_scans_graph == -1] = 0
            self._training_scans_graph[training_scans_graph == 0] = 1
            self._training_scans_graph[training_scans_graph == 1] = 2
            self._testing_scans = np.copy(testing_scans)
            self._testing_scans[testing_scans == -1] = 0
            self._testing_scans[testing_scans == 0] = 1
            self._testing_scans[testing_scans == 1] = 2
            self._all_scans = np.copy(all_scans)
            self._all_scans[all_scans == -1] = 0
            self._all_scans[all_scans == 0] = 1
            self._all_scans[all_scans == 1] = 2
        elif self._occupancy_vals == Data.OccupancyVals.TWO:
            self._training_scans = training_scans
            self._testing_scans = testing_scans
            self._all_scans = all_scans
        else:
            raise Exception()

    def plot_polar_scan_setup(self, masked=False):
        a, r = np.meshgrid(np.radians(self._angles), self._radiuses)
        plt.rc('ytick', labelsize=20)
        fig = plt.figure(figsize=(8, 8))
        if masked:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'mycmap', [(0, 'red'),
                           (0.333, 'gray'),
                           (0.666, 'white'),
                           (1, 'black')])
        else:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'mycmap', [(0, 'gray'),
                           (0.5, 'white'),
                           (1, 'black')])
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                          projection='polar', facecolor='#880000')
        ax.get_xaxis().set_ticks([])
        ax.tick_params(axis='y', colors='#3333ff')
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(-1)
        # Make same fake data to properly sale the colors on the plot
        fake_scan = np.zeros((self._num_radius_cells, self._num_angle_cells))
        fake_scan[0, 0] = 0
        fake_scan[1, 0] = 1
        fake_scan[2, 0] = 2
        if masked:
            fake_scan[3, 0] = -1
        plot = ax.pcolormesh(a, r, fake_scan, cmap=cmap)
        return fig, plot

    def plot_polar_scan_plot(self, plot_data, scan, img_path):
        fig, plot = plot_data
        plot.set_array(scan)
        fig.savefig(img_path, dpi=100, bbox_inches='tight')

    def plot_polar_scan(self, scan, img_path, masked=False):
        plot_data = self.plot_polar_scan_setup(masked)
        self.plot_polar_scan_plot(plot_data, scan, img_path)
        plt.close()

    def visualize_data(self, vis_dir):
        print("\nPlotting data...", )
        plot_data = self.plot_polar_scan_setup()
        for i, d in enumerate(self._data):
            rid = d[0]
            rcat = d[1]
            scan = d[2]
            img_path = os.path.join(vis_dir, "%04d-%s-%s.png" %
                                    (i, rid, rcat))
            self.plot_polar_scan_plot(plot_data, scan + 1, img_path)
            # Print progress
            perc = (i / len(self._data)) * 100.0
            sys.stdout.write("%.2f \r" % (perc))
            sys.stdout.flush()
        plt.close()
        print("Done!")

    def print_info(self):
        print("\nData info:")
        print("* Num angle cells: %s" % self._num_angle_cells)
        print("* Num radius cells: %s" % self._num_radius_cells)
        print("* Angles: %s" % ', '.join('%.2f' % a for a in self._angles))
        print("* Radiuses: %s" % ', '.join('%.2f' % a for a in self._radiuses))
        print("* Num training samples: %s" % len(self._training_scans))
        print("* Num data samples: %s" % len(self._data))
        print("* Data sample size: %s" % (self._training_scans.shape[1],))
        print("* Train rooms: %s" % ', '.join(self._train_rooms))
        print("* Test rooms: %s" % ', '.join(self._test_rooms))

    def _mask_scan(self, scan, mask_pos):
        scopy = np.copy(scan)
        scopy = scopy.reshape(self._num_radius_cells, self._num_angle_cells)
        scopy[:, mask_pos:(mask_pos + self._num_angle_cells // 4)] = -1
        return scopy.ravel()

    def generate_masked_scans(self, mask_seed):
        mask_pos = np.random.RandomState(mask_seed).randint(
            self._num_angle_cells - self._num_angle_cells // 4,
            size=len(self._all_scans))
        self._masked_scans = []
        for i, s in enumerate(self._all_scans):
            self._masked_scans.append(
                self._mask_scan(s, mask_pos[i]))
        self._masked_scans = np.vstack(self._masked_scans)

    def visualize_masked_scans(self, vis_dir):
        print("\nPlotting masked scans...", )
        plot_data = self.plot_polar_scan_setup(masked=True)
        for i, s in enumerate(self._masked_scans):
            img_path = os.path.join(vis_dir, "%04d.png" % i)
            self.plot_polar_scan_plot(plot_data, s, img_path)
            # Print progress
            perc = (i / len(self._masked_scans)) * 100.0
            sys.stdout.write("%.2f \r" % (perc))
            sys.stdout.flush()
        plt.close()
        print("Done!")

    def save_masked_scans(self, res_dir):
        print("\nSaving masked scans...", )
        np.save(os.path.join(res_dir, 'masked_scans'),
                self._masked_scans)
        print("Done!")


    def verify_integrity(self):
        print("\nVerifying data integrity...")
        self._verify_training_footprint()
        print("Done!")


    def _verify_training_footprint(self):
        """Verify that what's in training_footprint covers each topological map
        fully. Each element in `training_footprint` has the format:

        graph_id, groundtruth_class_str, nid

        graph_id = {building#}_{seq_id} (see experiments/paths.py)
        """
        maps_counts = {}   # map from graph_id to a set of tuples (nid, groundtruth_class)

        topo_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
        training_footprint = self._training_footprint_graph

        for d in training_footprint:
            graph_id = d[0]
            groundtruth_class = d[1]
            nid = d[2]
            if graph_id not in maps_counts:
                maps_counts[graph_id] = set()
            maps_counts[graph_id].add((nid, groundtruth_class))

        db_loaded = set()
        for graph_id in sorted(maps_counts):
            db = graph_id.split("_")[0].capitalize()
            seq_id = "_".join(graph_id.split("_")[1:])
            if db not in db_loaded:
                topo_dataset.load(db, skip_unknown=True, skip_placeholders=True, single_component=False)
                db_loaded.add(db)
            topo_map = topo_dataset.get(db, seq_id)
            for nid, groundtruth_class in maps_counts[graph_id]:
                if nid not in topo_map.nodes:
                    raise Exception("Something wrong.\n"
                                    + "graph_id: %s\n" % graph_id
                                    + "Node %s does not exist in the topo map" % str(nid))
                if topo_map.nodes[nid].label != groundtruth_class:
                    raise Exception("Something wrong.\n"
                                    + "graph_id: %s\n" % graph_id
                                    + "Expected: node %d class %s" % (nid, groundtruth_class)
                                    + "  Actual: node %d class %s" % (nid, topo_map.nodes[nid].label))
            if len(maps_counts[graph_id]) != len(topo_map.nodes):
                raise Exception("Incorrect number of nodes with likelihoods."
                                + "graph_id: %s\n" % graph_id
                                + "Expected: %s" % (len(topo_map.nodes))
                                + "  Actual: %s" % (len(maps_counts[graph_id])))
        print("-  Verified training footprint.")
