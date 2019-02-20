# Manager for processing topological map dataset
#
# author: Kaiyu Zheng

import matplotlib
import matplotlib.pyplot as plt

from deepsm.graphspn.tbm.topo_map import PlaceNode, TopologicalMap
from deepsm.graphspn.tbm.template import Template, SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate, EdgeRelationTemplateInstance, NodeTemplate
from deepsm.graphspn.tbm.algorithm import NodeTemplatePartitionSampler, EdgeRelationPartitionSampler
from deepsm.util import CategoryManager, compute_view_number
import deepsm.experiments.paths as paths
import os, sys, re
import numpy as np
import csv
import math
import random
from collections import deque
import json
import copy

DEBUG = False

class TopoMapDataset:


    def __init__(self, db_root):
        """
        db_root: path to root db. Database should be in 'db_root/db_name' directory.
        """
        self.db_root = db_root
        self._topo_maps_data = {}  # {db_name: {seq_id: topo_map}}
        self._using_tiny_graphs = {}


    # TODO: This shouldn't really be here.  It's a more general function for the experiments.
    def get_dbname_info(self, db_name, train=True):
        """Given a db_name (str), e.g. Stockholm456, return a tuple ('Stockholm', '456', 7}.
        If `train` is False, then regard the floor information (i.e. 456 in this example) as
        testing floors and thus will return ('Stockholm', 7, '456')"""
        # Example: Stockholm456
        building = re.search("(stockholm|freiburg|saarbrucken)", db_name, re.IGNORECASE).group().capitalize()
        given_floors = db_name[len(building):]
        if building == "Stockholm":
            remaining_floors = {4, 5, 6, 7}
        elif building == "Freiburg":
            remaining_floors = {1, 2, 3}
        elif building == "Saarbrucken":
            remaining_floors = {1, 2, 3, 4}
        for f in given_floors:
            remaining_floors = remaining_floors - {int(f)}
        remaining_floors = "".join(map(str, (sorted(remaining_floors))))
        if train:
            return building, given_floors, remaining_floors
        else:
            return building, remaining_floors, given_floors

    def _load_dgsm_likelihoods(self, db_names, db_test=None, **kwargs):
        """Load all dgsm_likelihoods for the databases provided in db_names.

        db_info: a dictionary that maps from db_name to a list of tuples
                 (train_floors, test_floor). Exmaple for db_name: Stockholm456

        kwargs:
            db_test (list): If provided, we will load likelihoods in a different way for these dbs.
                            Example: ['Stockholm7']. `db_names` should not include db_test.
                            WARNING: It's actually not practical at the moment to have multiple elements
                            in db_test.

        Note: This function is only planned to be used for DGSM_SAME_BUILDING experiments.
        """
        result = {}
        trial_number = kwargs.get("trial_number", 0)
        result_paths = {}
        for db_name in db_names:
            building, train_floors, test_floor = self.get_dbname_info(db_name)
            path_to_results = paths.path_to_dgsm_result_same_building(CategoryManager.NUM_CATEGORIES,
                                                                      building,
                                                                      "graphs",
                                                                      trial_number,
                                                                      train_floors,
                                                                      test_floor)
            result_paths[test_floor] = path_to_results
            with open(os.path.join(path_to_results, "%s%s_likelihoods_training.json" % (building.lower(), train_floors))) as f:
                d = json.load(f)
                for graph_id in d:
                    if graph_id not in result:
                        result[graph_id] = {}
                    for nid_str in d[graph_id]:
                        result[graph_id][int(nid_str)] = d[graph_id][nid_str]

        if db_test is not None:
            for db_name in db_test:
                building, train_floors, test_floor = self.get_dbname_info(db_name, train=False)
                path_to_results = result_paths[test_floor]
                for fname in os.listdir(path_to_results):
                    if fname.endswith("_likelihoods.json"):
                        graph_id = building.lower() + "_" + "_".join(fname.split("_likelihoods.json")[0].split("_")[1:])
                        with open(os.path.join(path_to_results, fname)) as f:
                            d = json.load(f)
                            result[graph_id] = {}
                            for nid_str in d:
                                result[graph_id][int(nid_str)] = d[nid_str][2]
        return result


    def load_template_dataset(self, template_type, db_names=None, seq_ids=None,
                              seqs_limit=-1, random=True, db_test=None, **kwargs):
        """
        loads all samples for template_type. `template_type` can either be 'three' or 'view'.
        
        If `use_dgsm_likelihoods` is set to True, the returned also contains likelihoods.

        Both samples and likelihoods are of format:
            db_name -> template.
        
        If random is False, loads samples of node template from file. The file is in path 
        
          TOPO_DB_PATH/
              building/
                  seq_id/
                      samples/
                          samples_{template}-{num_classes}.csv
                          likelihoods_{template}-{num_classes}.csv

        If `db_names` and `seq_ids` are both None, load from all db_names. If both are not None,
        will treat `seq_ids` as None. `seq_ids` should be a list of "{db}-{seq_id}" strings.

        If `db_test` is not None, that means we want to load samples and likelihoods from the
        test sequences as well. Then `db_names` is ASSUMED to not be None and does not contain
        `db_test`. Example: Stockholm7.

        """
        get_likelihoods = kwargs.get("use_dgsm_likelihoods", False)  # For now, this only works for random partitioning
        templates = Template.templates_for(template_type)
        db_train = copy.deepcopy(db_names)
        
        if random:
            rs = self.create_template_dataset_with_sampler(template_type, db_names=db_names, seq_ids=seq_ids, seqs_limit=seqs_limit,
                                                           num_partitions=kwargs.get('num_partitions', 5),
                                                           random=random, get_likelihoods=get_likelihoods, db_test=db_test)

            likelihoods = None
            if get_likelihoods:
                samples, likelihoods = rs
            else:
                samples = rs

            # Note: iterate in sorted order to guarantee match-up between samples and likelihoods.
                
            result_samples = {}
            for db_name in sorted(samples):
                result_samples[db_name] = {}
                for seq_id in sorted(samples[db_name]):
                    for template in samples[db_name][seq_id]:
                        if template not in result_samples[db_name] and len(samples[db_name][seq_id][template]) > 0:
                            result_samples[db_name][template] = samples[db_name][seq_id][template]
                        elif len(samples[db_name][seq_id][template]) > 0:
                            result_samples[db_name][template] = np.vstack((result_samples[db_name][template],
                                                                           samples[db_name][seq_id][template]))

            if likelihoods is not None:
                result_likelihoods = {}
                for db_name in sorted(likelihoods):
                    result_likelihoods[db_name] = {}
                    for seq_id in sorted(likelihoods[db_name]):
                        for template in likelihoods[db_name][seq_id]:
                            if template not in result_likelihoods[db_name] and len(likelihoods[db_name][seq_id][template]) > 0:
                                result_likelihoods[db_name][template] = likelihoods[db_name][seq_id][template]
                            elif len(likelihoods[db_name][seq_id][template]) > 0:
                                result_likelihoods[db_name][template] = np.vstack((result_likelihoods[db_name][template],
                                                                                   likelihoods[db_name][seq_id][template]))

            # Split train db and test db. Then return.
            if db_test is not None:
                result_samples_train, result_samples_test = {}, {}
                for db_name in sorted(db_train):
                    result_samples_train[db_name] = result_samples[db_name]
                for db_name in sorted(db_test):
                    result_samples_test[db_name] = result_samples[db_name]

                if get_likelihoods:
                    result_likelihoods_train, result_likelihoods_test = {}, {}
                    for db_name in sorted(db_train):
                        result_likelihoods_train[db_name] = result_likelihoods[db_name]
                    for db_name in sorted(db_test):
                        result_likelihoods_test[db_name] = result_likelihoods[db_name]

                    return result_samples_train, result_likelihoods_train, result_samples_test, result_likelihoods_test
                else:
                    return result_samples_train, result_likelihoods_test
            else:
                if get_likelihoods:
                    return result_samples, result_likelihoods
                else:
                    return result_samples
                                                                  
        else:
            if db_names is None and seq_ids is None:
                db_names = self._topo_maps_data.keys()

            topo_maps = {}  # map from "{db}_{seq_id}" to topo map
            if db_names is not None:
                for db_name in db_names:
                    for seq_id in self._topo_maps_data[db_name]:
                        topo_maps[db_name+"-"+seq_id] = self._topo_maps_data[db_name][seq_id]
            else:  # seq_ids must not be None
                for db_seq_id in seq_ids:
                    db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                    topo_maps[db_seq_id] = self._topo_maps_data[db_name][seq_id]

            samples = {}
            for db_seq_id in topo_maps:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]

                if db_name not in samples:
                    samples[db_name] = None

                for template in tepmlates:
                    if template not in samples[db_name][templates]:
                        samples[db_name][template] = None
                        
                    samples_path = os.path.join(self.db_root, db_name, seq_id, "samples",
                                                "samples_%s-%s.csv" % (template.__name__, CategoryManager.NUM_CATEGORIES))
                    loaded_samples = np.loadtxt(samples_path, dtype=int, delimiter=",")
                    if loaded_samples.ndim == 1:
                        loaded_samples = loaded_samples.reshape(-1, 1)

                    if samples[db_name][template] is None:
                        samples[db_name][template] = loaded_samples
                    else:
                        samples[db_name][template] = np.vstack((samples[db_name][template], loaded_samples))

            return samples


    def create_template_dataset_with_sampler(self, template_type, db_names=None, seq_ids=None, seqs_limit=-1,
                                             num_partitions=5, num_rounds=100, repeat=1,
                                             save=True, random=False, get_likelihoods=False, db_test=None):
        """
        Return a dataset of samples that can be used to train template SPNs.

        This dataset contains symmetrical data, i.e. for every pair of semantics, its
        reverse is also present in the dataset.

        If `db_names` and `seq_ids` are both None, load from all db_names. If both are not None,
        will treat `seq_ids` as None. `seq_ids` should be a list of "{db}-{seq_id}" strings.

        `num_partitions` number of partitions are created per topo map among `num_rounds`
        of partition sampling. This process is executed for `repeat` number of times.

        `template_type` determines the sampler to be used. Its value can be 'VIEW' or 'THREE'

        If `random` is True, simply sample 5 random partitions. `num_rounds` is overridden,
        and save is set to False.
        """
        def get_lh(lh, *nids):
            return np.array([lh[nid] for nid in nids])

        if random is True:
            save = False

        if db_names is None and seq_ids is None:
            db_names = list(self._topo_maps_data.keys())

        if get_likelihoods:
            dgsm_likelihoods = self._load_dgsm_likelihoods(db_names, db_test=db_test)

        if db_test is not None:
            db_names.extend(db_test)

        samples, likelihoods = {}, {}
        topo_maps = {}  # map from "{db}_{seq_id}" to topo map
        if db_names is not None:
            for db_name in db_names:
                samples[db_name], likelihoods[db_name] = {}, {}
                for seq_id in self._topo_maps_data[db_name]:
                    topo_maps[db_name+"-"+seq_id] = self._topo_maps_data[db_name][seq_id]
                    samples[db_name][seq_id], likelihoods[db_name][seq_id] = {}, {}
        else:  # seq_ids must not be None
            for db_seq_id in seq_ids:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = self._topo_maps_data[db_name][seq_id]
                if db_name not in samples:
                    samples[db_name], likelihoods[db_name] = {}, {}
                samples[db_name][seq_id], likelihoods[db_name][seq_id] = {}, {}

        while repeat >= 1:
            repeat -= 1
            for db_seq_id in topo_maps:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                building, _, _ = self.get_dbname_info(db_name)
                graph_id = building.lower() + "_" + seq_id  # Here, graph_id needs to comply with the format used in load_dgsm_likelihoods()
                topo_map = topo_maps[db_seq_id]
                print("Partitioning topo map %s" % (db_seq_id))

                if template_type.lower() == "three":
                    sampler = NodeTemplatePartitionSampler(topo_map)
                elif template_type.lower() == "view":
                    sampler = EdgeRelationPartitionSampler(topo_map)
                elif template_type.lower() == "star":
                    sampler = NodeTemplatePartitionSampler(topo_map, templates=Template.templates_for("star"))
                else:
                    raise ValueError("Unrecognized template type %s" % template_type)

                if random:
                    chosen_pset, _ = sampler.sample_partitions(num_partitions)
                else:
                    partition_sets, _, best_index = sampler.sample_partition_sets(num_rounds,
                                                                                  num_partitions,
                                                                                  pick_best=True)
                    chosen_pset = partition_sets[best_index]
                for p in chosen_pset:
                    if template_type.lower() == "three" or template_type.lower() == "star":
                        for template in p:
                            if template not in samples[db_name][seq_id]:
                                samples[db_name][seq_id][template] = []
                                likelihoods[db_name][seq_id][template] = []
                            supergraph = p[template]
                            for snid in supergraph.nodes:
                                samples[db_name][seq_id][template].append(supergraph.nodes[snid].to_catg_list())
                                if template.num_nodes() >= 2 and template.num_nodes() <= 3:
                                    samples[db_name][seq_id][template].append(list(reversed(supergraph.nodes[snid].to_catg_list())))

                                # If we want likelihoods
                                if get_likelihoods:
                                    template_lh = get_lh(dgsm_likelihoods[graph_id], *(n.id for n in supergraph.nodes[snid].nodes))
                                    likelihoods[db_name][seq_id][template].append(template_lh)
                                    if template.num_nodes() >= 2:
                                        reversed_template_lh = get_lh(dgsm_likelihoods[graph_id], *(n.id for n in reversed(supergraph.nodes[snid].nodes)))
                                        likelihoods[db_name][seq_id][template].append(reversed_template_lh)

                    elif template_type.lower() == "view":
                        for template in p:  # template here is a tuple (num_nodes, num_edge_pair)
                            template_class = EdgeRelationTemplateInstance.get_class(template)
                            if template not in samples[db_name][seq_id]:
                                samples[db_name][seq_id][template_class] = []
                                likelihoods[db_name][seq_id][template_class] = []
                            for i in p[template]:
                                catg_list, vdist = i.to_sample()
                                if catg_list is not None:
                                    if vdist is None:
                                        samples[db_name][seq_id][template_class].append(catg_list)
                                        if get_likelihoods:
                                            likelihoods[db_name][seq_id][template_class].append(get_lh(dgsm_likelihoods[graph_id], *(n.id for n in i.nodes)))
                                    else:
                                        vdist -= 1 # -1 is because vdist ranges from 1-4 but we want 0-3 as input to the network
                                        samples[db_name][seq_id][template_class].append(catg_list + [vdist])
                                        samples[db_name][seq_id][template_class].append(list(reversed(catg_list)) + [vdist])
                                        if get_likelihoods:
                                            likelihoods[db_name][seq_id][template_class].append(get_lh(dgsm_likelihoods[graph_id], *(n.id for n in i.nodes)))
                                            likelihoods[db_name][seq_id][template_class].append(get_lh(dgsm_likelihoods[graph_id], *(n.id for n in reversed(i.nodes))))
                                else:
                                    vdist -= 1 # same reason as above
                                    samples[db_name][seq_id][template_class].append([vdist])

                    else:
                        raise ValueError("Unrecognized template type %s" % template_type)

        # Save
        if save:
            for db_name in samples:
                for seq_id in samples[db_name]:
                    for template in samples[db_name][seq_id]:
                        save_dirpath = os.path.join(self.db_root, db_name, seq_id, "samples")
                        os.makedirs(save_dirpath, exist_ok=True)
                        samples_for_template = np.array(samples[db_name][seq_id][template], dtype=int)
                        likelihoods_for_template = np.array(likelihoods[db_name][seq_id][template], dtype=int)
                        
                        if template_type.lower() == "three":
                            np.savetxt(os.path.join(save_dirpath, 'samples_%s-%s.csv' % (template.__name__, CategoryManager.NUM_CATEGORIES)),
                                       samples_for_template, delimiter=",", fmt='%i')
                            if get_likelihoods:
                                np.savetxt(os.path.join(save_dirpath, 'likelihoods_%s-%s.csv' % (template.__name__, CategoryManager.NUM_CATEGORIES)),
                                           likelihoods_for_template, delimiter=",", fmt='%i')
                        elif template_type.lower() == "view":
                            np.savetxt(os.path.join(save_dirpath, 'samples_%s-%s.csv' % (EdgeRelationTemplateInstance.get_class(template).__name__,
                                                                                         CategoryManager.NUM_CATEGORIES)),
                                       samples_for_template, delimiter=",", fmt='%i')
                            if get_likelihoods:
                                np.savetxt(os.path.join(save_dirpath, 'likelihoods_%s-%s.csv' % (EdgeRelationTemplateInstance.get_class(template).__name__,
                                                                                                 CategoryManager.NUM_CATEGORIES)),
                                           likelihoods_for_template, delimiter=",", fmt='%i')                            
                        else:
                            raise ValueError("Unrecognized template type %s" % template_type)
                        
                        print("Saved samples for %s %s %s" % (db_name, seq_id, template))
                        if get_likelihoods:
                            print("Saved likelihoods for %s %s %s" % (db_name, seq_id, template))
        print("Done!")
        if get_likelihoods:
            return samples, likelihoods
        else:
            return samples
            
        

    def create_node_template_dataset(self, template, num_partitions=10,
                                     db_names=None, seq_ids=None, seqs_limit=-1, **kwargs):
        """
        Return a dataset of samples that can be used to train template SPNs. This
        dataset contains symmetrical data, i.e. for every pair of semantics, its
        reverse is also present in the dataset.

        If `db_names` and `seq_ids` are both None, load from all db_names. If both are not None,
        will treat `seq_ids` as None. `seq_ids` should be a list of "{db}-{seq_id}" strings.

        Return format:

           {K:V}, K is database name, V is -->
            --> NxM list, where N is the number of data samples, and M is the number of
           nodes in the template provided. For example, with 3-node template, each
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

                # for size in range(most_complex_template.size(), template.size(), -1):
                #     t = NodeTemplate.size_to_class(size)
                #     _, unused_graph = supergraph.partition(t, return_unused=True)
                #     supergraph = unused_graph

                supergraph = supergraph.partition(template)

                for n in supergraph.nodes:
                    template_sample = supergraph.nodes[n].to_catg_list()
                    samples[db_name].append(template_sample)
                    samples[db_name].append(list(reversed(template_sample)))

                total_seqs_count += 1

                if seqs_limit > 0 and total_seqs_count >= seqs_limit:
                    return samples
        return samples
        


    def create_edge_relation_template_dataset(self, template, num_partitions=10,
                                              db_names=None, seqs_limit=-1, return_stats=False,
                                              balance_views=False):
        """
        template (AbsEdgeRelationTemplate):
                   a class name which is a subclass of AbsEdgeRelationTemplate

        balance_views (bool): If true, will make sure that for each view distance value, the total
                   number of samples that have that value is the same.

        Returns a dictionary from database name to a list of samples (each sample is also a list
        """
        template = template.to_tuple() # we only need to work with the tuple representation of the template
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
                            _, vdist = ert.to_sample()  # sample is a number here.
                            sample = [vdist-1]  # We make a list ourselves. -1 is because vdist ranges from 1-4 but we want 0-3 as input to the network
                            samples[db_name].append(sample)
                        else:
                            sample, vdist = ert.to_sample()
                            samples[db_name].append(sample)
                            samples[db_name].append(list(reversed(sample)))
                            if vdist is not None:  # If vdist is None, then this template is just a single node.
                                samples[db_name][-1].append(vdist-1)  # -1 is because vdist ranges from 1-4 but we want 0-3 as input to the network
                                samples[db_name][-2].append(vdist-1)

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


    # def create_edge_template_dataset(self, edge_template, num_partitions=10,
    #                                  db_names=None, seqs_limit=-1, return_stats=False):
    #     """
    #     DEPRECATED.

    #     WARNING: IMPLEMENTATION OUT OF DATE
    #     Returns samples used to train edge template SPNs. Each sample is also
    #     a tuple, where the first half of elements are semantics, and
    #     the second half of elements are view numbers.

    #     Samples are stored in a dictionary. {db_name: [...samples...]}

    #     Return format:

    #        {K:V}, K is database name, V is -->
    #         --> NxM list, where N is the number of data samples, and M is the number of
    #        ELEMENTS in a edge_template sample (refer to EdgeTemplate.to_sample() for
    #        details).
    #     """
    #     samples = {}
    #     total_seqs_count = 0
    #     if db_names is None:
    #         db_names = self._topo_maps_data.keys()

    #     __stats = {}
            
    #     for db_name in db_names:
    #         samples[db_name] = []
    #         __stats[db_name] = {'coverage':[0,0,0]}
    #         __seqs_in_db = 0
    #         __edges_covered, __total_edges = 0, 0
    #         for kk in range(num_partitions):
    #             for seq_id in self._topo_maps_data[db_name]:
    #                 topo_map = self._topo_maps_data[db_name][seq_id]
    #                 edge_tmpls, edge_singles = topo_map.partition_by_edge(edge_template)
    #                 # if the given edge template is a SingleEdgeTemplate, then edge_singles
    #                 # should be empty. So we only need to deal with edge_tmpls
    #                 for etmpl in edge_tmpls:
    #                     samples[db_name].extend(list(etmpl.to_sample()))
                        
    #                 total_seqs_count += 1
    #                 __edges_covered += len(edge_tmpls)*edge_template.num_edges()
    #                 __total_edges += len(topo_map.edges)
    #                 __seqs_in_db += 1
    #                 __stats[db_name]['coverage'][0] = __edges_covered / __seqs_in_db
    #                 __stats[db_name]['coverage'][1] = __total_edges / __seqs_in_db
    #                 __stats[db_name]['coverage'][2] = __edges_covered / __total_edges
                    
    #                 if seqs_limit > 0 and total_seqs_count >= seqs_limit:
    #                     if return_stats:
    #                         return samples, __stats
    #                     else:
    #                         return samples
    #         if return_stats:
    #             return samples, __stats
    #         else:
    #             return samples



    # def _balance_views(self, template, samples, divisions=8):
    #     """First, count the number of samples for each view. Then, go through each possible
    #     combination of classes and view distances. For each combination where the view distance
    #     with largest number of samples is not the same as that for the entire set of samples,
    #     double the number of samples for that combination, and update the counts."""
    #     if template.num_edge_pair == 0:
    #         return samples  # No need to balance. No view distance variable.
        
    #     for db_name in samples_dict:
            
    #         total_counts = {}  # total number of samples for each view distance value
    #         case_counts = {} # total number of samples for each combination (i.e. case)
    #         for sample in samples_dict[db_name]:
    #             if tuple(sample) not in case_counts:
    #                 cases_counts[tuple(sample)] = {}
    #             cases_counts[tuple(sample[:template.num_nodes()])][sample[-1]] = cases_counts[tuple(sample[:template.num_nodes()])].get(sample[-1],0) + 1
    #             total_counts[sample[-1]] = counts.get(sample[-1],0) + 1

    #         max_count_view = max(total_counts, key=lambda k: total_counts[k])

    #         for case in cases_counts:
    #             max_count_view_case = max(case_counts[case], key=lambda k: case_counts[case][k])
    #             if max_count_view_case != max_count_view:
    #                 # double the number of samples for this case.
    #                 num_case_samples = sum(cases_counts[case].values())
    #                 for i in range(num_case_samples):
    
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


    def load(self, db_name, skip_unknown=False, skip_placeholders=False,
             limit=None, segment=False, single_component=True, room_remap={'7-2PO-1': '7-1PO-3'},
             skipped_seq_pattern={"floor6_base*"},
             skipped_rooms={'Stockholm':{}}):#{'4-1PO-1'}}):
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
           discard smaller ones. This is by default True.

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

            # Previously the sequence skipping does not explicitly exist. GraphSPN relied on the fact
            # that there is no polar scans for floor6_base sequences so no DGSM results for them in order
            # to skip the floor6_base sequences. Now we can make this explicit through the skipped_seq_pattern.
            skip_seq = False
            for pattern in skipped_seq_pattern:
                if re.search(pattern, seq_id) is not None:
                    if DEBUG:
                        print("Skipping %s (matched pattern %s)" % (seq_id, pattern))
                    skip_seq = True
            if skip_seq:
                continue

            # check if over limit
            if limit is not None and len(topo_maps) >= limit:
                break

            node_room_mapping = {}
            with open(os.path.join(db_path, seq_id, "rooms.dat")) as f:
                rows = csv.reader(f, delimiter=' ')
                for row in rows:
                    nid = int(row[0])
                    room_id = row[1]
                    node_room_mapping[nid] = room_id

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

                    # Also, skip nodes in rooms that we want to skip
                    building = self.get_dbname_info(db_name)[0]  # we don't care whether db_name is test or train here.
                    if building in skipped_rooms and node_room_mapping[nid] in skipped_rooms[building]:
                        if DEBUG:
                            print("Skipping node %d in room %s" % (nid, node_room_mapping[nid]))
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

                    # Room mapping
                    room_id = node_room_mapping[node['id']]
                    if room_id in room_remap:
                        room_using = room_remap[room_id]
                        node['label'] = room_using.split("-")[1]  # since we change the room, label should also change
                        # print("Remap node %d from room %s to room %s" % (node['id'], room_id, room_using))
                    node['room'] = room_id
                    
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
                        node['id'], node['placeholder'], node['pose'], node['anchor_pose'], node['label'], node['room']
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
                    if len(components) > 1:
                        if DEBUG:
                            print("-- %s is broken into %d components" % (seq_id, len(components)))
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

        Returns two lists of sequoence ids, each for one group. Note that to distinguish Freiburg sequences and
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

                
