# Manager for processing topological map dataset
#
# author: Kaiyu Zheng

import matplotlib
import matplotlib.pyplot as plt

from deepsm.graphspn.tbm.topo_map import PlaceNode, TopologicalMap
from deepsm.graphspn.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate
from deepsm.util import CategoryManager, compute_view_number
import os, sys
import numpy as np
import csv
import math
import random
from collections import deque


class TopoMapDataset:


    def __init__(self, db_root):
        """
        db_root: path to root db. Database should be in 'db_root/db_name' directory.
        """
        self.db_root = db_root
        self._topo_maps_data = {}  # {db_name: {seq_id: topo_map}}
        self._using_tiny_graphs = {}
        


    def create_template_dataset(self, *templates, num_partitions=10,
                                db_names=None, seq_ids=None, seqs_limit=-1):
        """
        Return a dataset of samples that can be used to train template SPNs. This
        dataset contains symmetrical data, i.e. for every pair of semantics, its
        reverse is also present in the dataset.

        If `db_names` and `seq_ids` are both None, load from all db_names. If both are not None,
        will treat `seq_ids` as None. `seq_ids` should be a list of "{db}-{seq_id}" strings.

        Return format:

           {K:V}, K is database name, V is -->
            --> NxM list, where N is the number of data samples, and M is the number of
           nodes in the templates provided. For example, with 3-node template, each
           data sample would be a list [a, b, c] where a, b, c are the category numbers
           for the nodes on the template.
        """
        samples = {}
        total_seqs_count = 0
        if db_names is None and seq_ids is None:
            db_names = self._topo_maps_data.keys()

        topo_maps = {}  # map from "{db}_{seq_id}" to topo map
        if db_names is not None:
            for db_name in db_names:
                samples[db_name] = []
                for seq_id in self._topo_maps_data[db_name]:
                    topo_maps[db_name+"-"+seq_id] = self._topo_maps_data[db_name][seq_id]

        else:  # seq_ids must not be None
            for db_seq_id in seq_ids:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = self._topo_maps_data[db_name][seq_id]
                if db_name not in samples:
                    samples[db_name] = []
                
        for kk in range(num_partitions):
            for db_seq_id in topo_maps:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                supergraph = topo_maps[db_seq_id]

                for tmpl in templates:
                    supergraph = supergraph.partition(tmpl)

                for n in supergraph.nodes:
                    template_sample = supergraph.nodes[n].to_catg_list()
                    samples[db_name].append(template_sample)
                    samples[db_name].append(list(reversed(template_sample)))

                total_seqs_count += 1

                if seqs_limit > 0 and total_seqs_count >= seqs_limit:
                    return samples
        return samples
        

    def create_edge_template_dataset(self, edge_template, num_partitions=10,
                                     db_names=None, seqs_limit=-1, return_stats=False):
        """
        WARNING: IMPLEMENTATION OUT OF DATE
        Returns samples used to train edge template SPNs. Each sample is also
        a tuple, where the first half of elements are semantics, and
        the second half of elements are view numbers.

        Samples are stored in a dictionary. {db_name: [...samples...]}

        Return format:

           {K:V}, K is database name, V is -->
            --> NxM list, where N is the number of data samples, and M is the number of
           ELEMENTS in a edge_template sample (refer to EdgeTemplate.to_sample() for
           details).
        """
        samples = {}
        total_seqs_count = 0
        if db_names is None:
            db_names = self._topo_maps_data.keys()

        __stats = {}
            
        for db_name in db_names:
            samples[db_name] = []
            __stats[db_name] = {'coverage':[0,0,0]}
            __seqs_in_db = 0
            __edges_covered, __total_edges = 0, 0
            for kk in range(num_partitions):
                for seq_id in self._topo_maps_data[db_name]:
                    topo_map = self._topo_maps_data[db_name][seq_id]
                    edge_tmpls, edge_singles = topo_map.partition_by_edge(edge_template)
                    # if the given edge template is a SingleEdgeTemplate, then edge_singles
                    # should be empty. So we only need to deal with edge_tmpls
                    for etmpl in edge_tmpls:
                        samples[db_name].extend(list(etmpl.to_sample()))
                        
                    total_seqs_count += 1
                    __edges_covered += len(edge_tmpls)*edge_template.num_edges()
                    __total_edges += len(topo_map.edges)
                    __seqs_in_db += 1
                    __stats[db_name]['coverage'][0] = __edges_covered / __seqs_in_db
                    __stats[db_name]['coverage'][1] = __total_edges / __seqs_in_db
                    __stats[db_name]['coverage'][2] = __edges_covered / __total_edges
                    
                    if seqs_limit > 0 and total_seqs_count >= seqs_limit:
                        if return_stats:
                            return samples, __stats
                        else:
                            return samples
            if return_stats:
                return samples, __stats
            else:
                return samples


    def create_edge_relation_template_dataset(self, template, num_partitions=10,
                                              db_names=None, seqs_limit=-1, return_stats=False):
        """
        WARNING: IMPLEMENTATION OUT OF DATE
        template (tuple): a tuple (num_nodes, num_edges) indicating the type of EdgeRelationTemplate
                          being modeled.

        Returns a dictionary from database name to a list of samples (each sample is also a list
        """
        samples = {}
        total_seqs_count = 0
        if db_names is None:
            db_names = self._topo_maps_data.keys()

        __stats = {'num_samples':0}
        for db_name in db_names:
            samples[db_name] = []
            __stats[db_name] = {'num_seqs': 0, 'num_samples': 0}
            __seqs_in_db = 0
            for kk in range(num_partitions):
                for seq_id in self._topo_maps_data[db_name]:
                    topo_map = self._topo_maps_data[db_name][seq_id]
                    ert_map = topo_map.partition_by_edge_relations()
                    erts_matched = ert_map[template]
                    for ert in erts_matched:
                        if ert.nodes is None:  # no nodes.
                            _, sample = ert.to_sample()  # sample is a number here.
                            sample = [sample]  # We make a list ourselves.
                        else:
                            sample, vdist = ert.to_sample()
                            if vdist is not None:  # If vdist is None, then this template is just a single node.
                                sample.append(vdist)
                        samples[db_name].append(sample)
                    total_seqs_count += 1
                    __stats['total_seqs'] = total_seqs_count
                    __stats[db_name]['num_samples'] = len(samples[db_name])
                    __stats[db_name]['num_seqs'] += 1
                    __stats['num_samples'] += 1
                
                    if seqs_limit > 0 and total_seqs_count >= seqs_limit:
                        if return_stats:
                            return samples, __stats
                        else:
                            return samples
        if return_stats:
            return samples, __stats
        else:
            return samples
                            
    
    def get_topo_maps(self, db_name=None, seq_id=None, amount=1):
        """
        Returns a dictionary of seq_id to topo map.
        
        If `amount` is -1, get all with db_name.
        """
        if db_name is None:
            db_name = random.sample(self._topo_maps_data.keys(), 1)[0]
        if seq_id is None:
            if amount == -1:
                return self._topo_maps_data[db_name]
            topo_maps = {}
            for _ in range(amount):
                seq_id = random.sample(self._topo_maps_data[db_name].keys(), 1)[0]
                sample = self._topo_maps_data[db_name][seq_id]
                topo_maps[seq_id] = sample
            return topo_maps
        else:
            if self._using_tiny_graphs[db_name]:
                if '~' in seq_id:
                    return {seq_id: self._topo_maps_data[db_name][seq_id]}
                # Get all sequences with seq_id as prefix.
                all_tiny_maps = {k:self._topo_maps_data[db_name][k] for k in self._topo_maps_data[db_name]
                                 if k.startswith(seq_id)}
                # Return `amount` number of topo_maps
                topo_maps = {}
                for _ in range(amount):
                    seq_id = random.sample(all_tiny_maps.keys(), 1)[0]
                    sample = all_tiny_maps[seq_id]
                    topo_maps[seq_id] = sample
                return topo_maps
            else:
                return {seq_id: self._topo_maps_data[db_name][seq_id]}


    def load_tiny_graphs(self, db_name, seq_ids=None, skip_unknown=False, skip_placeholders=False, limit=None, num_nodes=2):
        """
        Load and break graphs into samples of only `num_nodes`.

        If `seq_ids` is provided, then the tiny graphs will be loaded from these sequences. Note
        that each element in `seq_ids` should be of format "{db}-{seq_id}"
        """
        # Get db names from seq_ids, if not None
        if seq_ids is not None:
            for db_seq_id in seq_ids:
                db_name = db_seq_id.split("-")[0]
                if db_name not in self._topo_maps_data:
                    self.load(db_name, skip_unknown=skip_unknown, skip_placeholders=skip_placeholders, limit=limit)
                self._using_tiny_graphs[db_name] = True
        else:
            self._using_tiny_graphs[db_name] = True
            if db_name not in self._topo_maps_data:
                self.load(db_name, skip_unknown=skip_unknown, skip_placeholders=skip_placeholders, limit=limit)
        # Break each topological map into multiple tiny ones
        tiny_graphs = {}
        topo_maps = {}
        if seq_ids is not None:
            for db_seq_id in seq_ids:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = self._topo_maps_data[db_name][seq_id]
        else:
            topo_maps = self._topo_maps_data[db_name]

        for seq_id in topo_maps:
            if seq_ids is not None:  # seq_id now has db_name prefix
                db_name = seq_id.split("-")[0]
                seq_id = seq_id.split("-")[1]
            if db_name not in tiny_graphs:
                tiny_graphs[db_name] = {}
            topo_map = self._topo_maps_data[db_name][seq_id]

            if num_nodes <= 20:
                max_num = 20  # maximum number of tiny graphs
            else:
                max_num = 5
            count = 0
            for _ in range(max_num):
                BFS = True
                start_nid = random.sample(topo_map.nodes.keys(), 1)[0]
                q = deque()
                q.append(start_nid)
                nodes, conns = {start_nid:topo_map.nodes[start_nid]}, {}
                while len(q) > 0:
                    if BFS:
                        cur_nid = q.popleft()
                    else:
                        cur_nid = q.pop()
                    neighbors = topo_map.neighbors(cur_nid)
                    for nnid in random.sample(neighbors, len(neighbors)):
                        if nnid not in nodes:
                            nodes[nnid] = topo_map.nodes[nnid]
                            if cur_nid not in conns:
                                conns[cur_nid] = set({})
                            if nnid not in conns:
                                conns[nnid] = set({})
                            conns[cur_nid].add((nnid, compute_view_number(topo_map.nodes[cur_nid], topo_map.nodes[nnid])))
                            conns[nnid].add((cur_nid, compute_view_number(topo_map.nodes[nnid], topo_map.nodes[cur_nid])))
                            if len(nodes) == num_nodes:
                                # We can quit BFS/DFS. Set the queue to be empty
                                q.clear()
                                break
                            q.append(nnid)
                if len(nodes) < num_nodes:
                    continue # This one has invalid size
                # Add edges that weren't covered by the BFS but are present in the tiny graph
                for nid in nodes:
                    for mid in nodes:
                        if nid != mid:
                            if topo_map.edge_between(nid, mid) is not None:
                                conns[nid].add((mid, compute_view_number(nodes[nid], nodes[mid])))
                tiny = TopologicalMap(nodes, conns)
                tiny_graphs[db_name][seq_id+"~"+str(count)] = tiny
                count += 1
                BFS = not BFS
        for db_name in tiny_graphs:
            self._topo_maps_data[db_name] = tiny_graphs[db_name]


    def load(self, db_name, skip_unknown=False, skip_placeholders=False, limit=None, segment=False, single_component=True):
        """
        loads data. The loaded data is stored in self.topo_maps_data, a dictionary
        where keys are database names and values are data.

        If 'skip_unknown' is set to True, then we skip nodes whose labels map to unknown category.

        If 'skip_placeholders' is set to True, then we skip nodes that are placeholders

        'limit' is the max number of topo maps to be loaded.

        If `segment` is True, the topological map will be segmented such that each node in the graph
           is a whole room, and there is no doorway node.

        If `single_component` is True, then will make sure the loaded graph is a single, connected graph. If the
           raw graph contains multiple components, will select the largest component as the topological map, and
           discard smaller ones.

        CSV format should be:
        "
        node['id'](int), node['placeholder'](bool), node_pose_x(float), node_pose_y(float), 
            match_pose[1](float), match_pose[2](float), node_anchor_pose[0](float), node_anchor_pose[1](float),
            match_label(string), match_vscan_time(timestamp), ... 8 views (see comments below) ...
        "

        """
        topo_maps = {}
        db_path = os.path.join(self.db_root, db_name)
        for seq_id in sorted(os.listdir(db_path)):

            # check if over limit
            if limit is not None and len(topo_maps) >= limit:
                break

            # not over limit. Keep loading.
            with open(os.path.join(db_path, seq_id, "nodes.dat")) as f:
                nodes_data_raw = csv.reader(f, delimiter=' ')

                nodes = {}
                conn = {}
                skipped = set({})

                # We may want to skip nodes with unknown classes. This means we need to first
                # pick out those nodes that are skipped. So we iterate twice.
                for row in nodes_data_raw:
                    nid = int(row[0])
                    label = row[8]
                    placeholder = bool(int(row[1]))
                    if skip_unknown:
                        if CategoryManager.category_map(label, checking=True) == CategoryManager.category_map('UN', checking=True):
                            skipped.add(nid)
                    if skip_placeholders:
                        if placeholder:
                            skipped.add(nid)
                f.seek(0)

                for row in nodes_data_raw:
                    node = {
                        'id': int(row[0]),
                        'placeholder': bool(int(row[1])),
                        'pose': tuple(map(float, row[2:4])),
                        'anchor_pose': tuple(map(float, row[6:8])),
                        'label': row[8],
                        'vscan_timestamp': float(row[9])
                    }

                    # Skip it?
                    if skip_unknown and node['id'] in skipped:
                        continue

                    # Add connections
                    edges = set({})
                    i = 10
                    while i < len(row):
                        neighbor_id = int(row[i])
                        # Check if we should skip this
                        if skip_unknown and neighbor_id in skipped:
                            i += 3 # still need to advance the index
                            continue
                        
                        affordance = float(row[i+1])
                        view_number = float(row[i+2])
                        if neighbor_id != -1:
                            edges.add((neighbor_id, view_number))
                        i += 3

                    # If there is no edge for this node, we just skip it.
                    if len(edges) == 0:
                        continue

                    # No, we don't skip this.
                    pnode = PlaceNode(
                        node['id'], node['placeholder'], node['pose'], node['anchor_pose'], node['label']
                    )
                    nodes[pnode.id] = pnode
                    conn[pnode.id] = edges
                topo_map = TopologicalMap(nodes, conn)
                if segment:
                    topo_map = topo_map.segment(remove_doorway=True)
                # There may be multiple connected components in the topological map (due to skipping nodes).
                # If single_component is True, we only keep the largest component.
                if single_component:
                    components = topo_map.connected_components()
                    topo_maps[seq_id] = max(components, key=lambda c:len(c.nodes))
                else:
                    topo_maps[seq_id] = topo_map

        # Save the loaded topo maps
        self._topo_maps_data[db_name] = topo_maps
        self._using_tiny_graphs[db_name] = False


    def split(self, mix_ratio, *db_names):
        """
        Mix up the sequences in dbs (from `db_names`), and split them up into two groups of sequences.
        The first group takes `mix_ratio` percentage, used for training. The second group takes 1 - `mix_ratio`
        percentage, used for testing.

        Returns two lists of sequence ids, each for one group. Note that to distinguish Freiburg sequences and
        Saarbrucken sequences, each sequence id is prepended a string for its database (format "{db}-{seq_id}")
        """
        all_seqs = []
        for db_name in db_names:
            seqs = [db_name+"-"+seq_id for seq_id in self.get_topo_maps(db_name=db_name, amount=-1).keys()]
            all_seqs.extend(seqs)
        # Shuffle and split
        random.shuffle(all_seqs)
        split_indx = round(mix_ratio*len(all_seqs))
        return all_seqs[:split_indx], all_seqs[split_indx:]
        

    @property
    def topo_maps(self):
        return self._topo_maps_data

    def get(self, db_name, seq_id):
        return self._topo_maps_data[db_name][seq_id]
    

    def generate_visualization(self, coldmgr, db_names=None):
        """
        Generates visualizations of topological maps and save them
        to {db_root}/{db_name}/{seq_id}/{topo_map.png}

        coldmgr (ColdDatabaseManager): Cold database manager instance.
        """
        matplotlib.use('Agg')
        if db_names is None:
            db_names = self._topo_maps_data.keys()
        for db_name in db_names:
            coldmgr.db_name = db_name
            for seq_id in self._topo_maps_data[db_name]:
                topo_map = self._topo_maps_data[db_name][seq_id]
                topo_map.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'))
                plt.savefig(os.path.join(self.db_root, db_name, seq_id, "topo_map.png"))
                plt.clf()
                
                sys.stdout.write(".")
                sys.stdout.flush()
        sys.stdout.write('\n')

                
