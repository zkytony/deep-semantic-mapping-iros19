#!/usr/bin/env python
#
# Experiments for view template alone
#
# author: Kaiyu Zheng

import csv
import sys, os
import json
import random
import time
import numpy as np
from numpy import float32
from abc import abstractmethod

import tensorflow as tf

from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.graphspn.tbm.template import EdgeRelationTemplate, ThreeRelTemplate, SingleTemplate, SingleRelTemplate
from deepsm.graphspn.tbm.spn_template import TemplateSpn, EdgeRelationTemplateSpn
from deepsm.graphspn.tests.tbm.runner import TbmExperiment
from deepsm.graphspn.tests.runner import TestCase
from deepsm.util import CategoryManager, print_banner, print_in_box

from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT

MPE = 'mpe'
MARGINAL = 'marginal'

class EdgeRelationTemplateExperiment(TbmExperiment):

    def __init__(self, db_root, *spns, **kwargs):
        """
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.
        """
        super().__init__(db_root, *spns, **kwargs)

    class TestCase_CompletionTask(TestCase):

        def run(self, sess, **kwargs):
            pass


def run_edge_relation_template_experiment(train_kwargs, test_kwargs, seed=None, semantic=False):
    # EdgeRelation template experiments

    timestamp = time.strftime("%Y%m%d-%H%M")
    train_kwargs['timestamp'] = timestamp
    test_kwargs['timestamp'] = timestamp
    
    print_in_box(["NodeTemplate experiments"], ho="+", vr="+")
    
    spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}

    template_spn = EdgeRelationTemplateSpn(train_kwargs['template'], seed=seed, **spn_params)

    with tf.Session() as sess:
        name = "EdgeRelationTemplateExperiment"
        print_in_box(["Experiment %s" % name])
        print_banner("Start", ch='v')

        exp = EdgeRelationTemplateExperiment(TOPO_MAP_DB_ROOT,
                                             template_spn,
                                             root_dir=GRAPHSPN_RESULTS_ROOT, name=name)
        exp.load_training_data(*train_kwargs['train_db'],
                               skip_unknown=CategoryManager.SKIP_UNKNOWN)
        train_info = exp.train_models(sess, **train_kwargs)
        import pdb; pdb.set_trace()
        model = exp.get_model(test_kwargs['template'])

        if test_kwargs['inference_type'] == MPE:
            model.init_mpe_states()  # init MPE ops. (after expanding)
        else:
            print("Not initializing MPE states. Using marginal inference")
            model.init_learning_ops()

        exp.load_testing_data(test_kwargs['test_db'], skip_unknown=CategoryManager.SKIP_UNKNOWN)

        # test_kwargs['subname'] = 'random_lh_%s' % model.template.__name__
        # report_rnd = exp.test(TemplateSpnExperiment.TestCase_RandomCompletionTask, sess, **test_kwargs)
        # print_banner("Finish", ch='^')


if __name__ == "__main__":
    
    seed = random.randint(200,1000)

    # Config
    train_kwargs = {
        'num_partitions': 7,#10,
        'num_batches': 10,
        'save': True,
        'load_if_exists': False,
        'likelihood_thres': 0.1,
        'save_training_info': True,

        # spn_structure
        'num_decomps': 3,#1,
        'num_subsets': 3,#2,
        'num_mixtures': 2,#2,
        'num_input_mixtures': 2,#2

        'template': ThreeRelTemplate,

        'train_db': ['Stockholm456']
    }

    test_kwargs = {
        'template': ThreeRelTemplate,
        'limit': -1,
        'num_partitions': 2,
        'inference_type': MARGINAL,
        'test_db': 'Stockholm7',
        'expand': False
    }
    
    run_edge_relation_template_experiment(train_kwargs, test_kwargs, semantic=False, seed=seed)
