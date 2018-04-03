#!/usr/bin/env python

import tensorflow as tf
import csv
import numpy as np

from spn_topo.spn_model import SpnModel
from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, PairTemplate, SingletonTemplate, \
    ThreeNodeTemplate, PairEdgeTemplate, SingleEdgeTemplate, StarTemplate
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn
from spn_topo.tests.tbm.runner import TbmExperiment
from spn_topo.tests.runner import TestCase
from spn_topo.util import CategoryManager, print_banner, print_in_box

from spn_topo.tests.constants import COLD_ROOT, RESULTS_DIR, TOPO_MAP_DB_ROOT

if __name__ == "__main__":
    seed = 100
    train_kwargs = {
        'num_partitions': 10,
        'num_batches': 10,
        'likelihood_thres': 0.5,
    }
    spn_params = {
        'num_decomps': 1,
        'num_subsets': 3,
        'num_mixtures': 5,
        'num_input_mixtures': 5
    }
    pair_node_spn = NodeTemplateSpn(PairTemplate, seed=seed, **spn_params)
    single_node_spn = NodeTemplateSpn(SingletonTemplate, seed=seed, **spn_params)

    # Dataset
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load("Stockholm", skip_unknown=True)
    dataset.load("Freiburg", skip_unknown=True)
    dataset.load("Saarbrucken", skip_unknown=True)

    with tf.Session() as sess:

        # Load trained parameters from file
        train_db = ["Freiburg", "Saarbrucken"]
        for model in [pair_node_spn, single_node_spn]:
            CUR = "/home/zkytony/Documents/thesis/experiments/spn_topo/spn_topo/tests/tbm"
            saved_path = CUR + "/trained_spns/%s_%d_%s.spn" % (model.template.__name__,
                                                              CategoryManager.NUM_CATEGORIES,
                                                              "-".join(sorted(train_db)))
            model.init_weights_ops()
            model.initialize_weights(sess)
            model.load(saved_path, sess)
            model.init_learning_ops()

        # Single node template probability table
        single_prob = {}
        for i in range(CategoryManager.NUM_CATEGORIES):
            lh = single_node_spn.evaluate(sess, np.array([i], dtype=int))[0][0]
            single_prob[i] = lh

            
        # Pair node template probability table
        pair_prob = {}
        for i in range(CategoryManager.NUM_CATEGORIES):
            for j in range(CategoryManager.NUM_CATEGORIES):
                lh = pair_node_spn.evaluate(sess, np.array([i, j], dtype=int))[0][0]
                pair_prob[(i,j)] = lh

        # Save results as csv
        with open('single_node_prob.csv', 'w') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"')
            for i in sorted(single_prob):
                csvwriter.writerow([i, single_prob[i]])
            print("Saved single node template probability table.")
        with open('pair_nodes_prob.csv', 'w') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"')
            for i, j in sorted(pair_prob):
                csvwriter.writerow([i, j, pair_prob[(i,j)]])
            print("Saved pair nodes template probability table.")
        print("Done.")
                

