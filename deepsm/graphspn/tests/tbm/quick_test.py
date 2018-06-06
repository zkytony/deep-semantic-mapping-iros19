#!/usr/bin/env python

import sys, os
import json
import random
import time
import numpy as np
from numpy import float32
from abc import abstractmethod
from pprint import pprint
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, PairTemplate, SingletonTemplate, \
    ThreeNodeTemplate, PairEdgeTemplate, SingleEdgeTemplate
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn, InstanceSpn
from spn_topo.tests.tbm.runner import TbmExperiment
from spn_topo.tests.runner import TestCase
from spn_topo.util import CategoryManager, print_banner, print_in_box, ColdDatabaseManager
from spn_topo.tests.tbm.test_instance_spn import InstanceSpnExperiment
from pprint import pprint


# Policy functions
def doorway_policy(topo_map, node, **kwargs):
    return node.label == "DW"
        
def random_policy(topo_map, node, rand_rate=0.2):
    return random.uniform(0, 1.0) <= rand_rate


if __name__ == "__main__":

    seed = 100
    
    results_dir = "/home/zkytony/Documents/thesis/experiments/spn_topo/experiments/results"
    topo_map_db_root = "/home/zkytony/Documents/thesis/experiments/spn_topo/experiments/data/topo_map"
    # Config
    train_kwargs = {
        'num_partitions': 20,
        'num_batches': 10,
        'save': True,
        'load_if_exists': True,
        'likelihood_thres': 0.5,
        'save_training_info': True,
        'skip_unknown': CategoryManager.SKIP_UNKNOWN,

        # spn_structure
        'num_decomps': 1,
        'num_subsets': 3,
        'num_mixtures': 5,
        'num_input_mixtures': 5
    }

    spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}
    three_node_spn = NodeTemplateSpn(ThreeNodeTemplate, seed=seed, **spn_params)
    pair_node_spn = NodeTemplateSpn(PairTemplate, seed=seed, **spn_params)
    single_node_spn = NodeTemplateSpn(SingletonTemplate, seed=seed, **spn_params)
    spns = [three_node_spn, pair_node_spn, single_node_spn]
    name = "Quick_Tests"
    exp = InstanceSpnExperiment(topo_map_db_root, *spns,
                                root_dir=results_dir, name=name)
    print_in_box(["Experiment %s" % name])
    print_banner("Start", ch='v')

    train_dbs = ['Freiburg']
    test_db = 'Saarbrucken'
    exp.load_training_data(*train_dbs, skip_unknown=CategoryManager.SKIP_UNKNOWN)
    with tf.Session() as sess:
        train_info = exp.train_models(sess, **train_kwargs)
        spn_paths = {model.template.__name__:exp.model_save_path(model) for model in spns}
        
        exp.load_testing_data(test_db, skip_unknown=CategoryManager.SKIP_UNKNOWN)
        test_instances = exp.dataset.get_topo_maps(db_name=test_db, amount=1)
        for seq_id in test_instances:
            topo_map = test_instances[seq_id]
            spns_tmpls = [(spns[i], spns[i].template) for i in range(len(spns))]
            instance_spn = InstanceSpn(topo_map, sess, *spns_tmpls, num_partitions=1,
                                       seq_id = seq_id, spn_paths=spn_paths, divisions = 8)

            # Init ops, and try inferences
            instance_spn.init_ops()

            # Evaluate
            print(instance_spn.evaluate(sess, topo_map.current_category_map()))

            # Marginal
            groundtruth = topo_map.current_category_map()
            topo_map.mask_by_policy(random_policy)
            query = topo_map.current_category_map()
            query_nids = [nid for nid in query if query[nid] == -1]
            marginals = instance_spn.marginal_inference(sess, query_nids, query)
            pprint(marginals)
            topo_map.reset_categories()
            
            result_catg_map = {}
            for nid in marginals:
                result_catg_map[nid] = marginals[nid].index(max(marginals[nid]))
            total, correct = 0, 0
            ok = True
            for nid in query:
                if query[nid] == -1:
                    if result_catg_map[nid] == groundtruth[nid]:
                        correct += 1
                    total += 1
                else:
                    if nid in result_catg_map:
                        print("Oops.")
                        ok = False
            if not ok:
                print("Something wrong.")

            print(correct)
            print(total)
            
            # MPE
            catg_map = instance_spn.mpe_inference(sess, query)

            
            # Expand
            instance_spn.expand()
            instance_spn.init_ops()

            # Evaluate
            print(instance_spn.evaluate(sess, topo_map.current_category_map()))

            # MPE
            topo_map.mask_by_policy(random_policy)
            query = topo_map.current_category_map()
            query_lh = TbmExperiment.create_instance_spn_likelihoods(0, topo_map, true_catg_map,high_likelihood, low_likelihood)
            pprint(query)
            catg_map = instance_spn.mpe_inference(sess, query, query_lh=query_lh)
            pprint(catg_map)

            # Marginal
            query_nids = [nid for nid in query if query[nid] == -1]
            marginals = instance_spn.marginal_inference(sess, query_nids, query, query_lh=query_lh)
            pprint(marginals)
            
            result_catg_map = {}
            for nid in marginals:
                result_catg_map[nid] = marginals[nid].index(max(marginals[nid]))
            total, correct = 0, 0
            ok = True
            for nid in query:
                if query[nid] == -1:
                    if result_catg_map[nid] == groundtruth[nid]:
                        correct += 1
                    total += 1
                else:
                    if nid in result_catg_map:
                        print("Oops.")
                        ok = False
            if not ok:
                print("Something wrong.")

            print(correct)
            print(total)
            
            print("HEY")
