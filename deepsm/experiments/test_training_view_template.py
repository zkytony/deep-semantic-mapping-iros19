#!/usr/bin/env pytohn
#
# Verification of view template SPN training
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
from pprint import pprint
import csv

import tensorflow as tf

from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.graphspn.tbm.template import EdgeRelationTemplate, ThreeRelTemplate, SingleTemplate, SingleRelTemplate, RelTemplate
from deepsm.graphspn.tbm.spn_template import TemplateSpn, EdgeRelationTemplateSpn
from deepsm.graphspn.tests.tbm.runner import TbmExperiment
from deepsm.graphspn.tests.runner import TestCase
from deepsm.util import CategoryManager, print_banner, print_in_box
from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT


def save_samples_to_csv(template, samples_dict, save_path, save_stats=True):
    """
    Prints a csv file for the given sames to the `save_path`. Also
    saves a .log file that contains some stats (e.g. number
    of instances for each combination)

    template (EdgeRelationTemplate)
    samples_dict (dict) map from db_name t o a list of data samples.

    The CSV file has the format:

    DB_NAME NODE_1 NODE_2 ... VIEW_DIST_1 ...

    Store the abbreviation of the categories
    """
    cases = {}
    
    with open(save_path, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',', quotechar='"')
        for db_name in samples_dict:
            for sample in samples_dict[db_name]:
                res = []
                for i in range(template.num_nodes()):
                    res.append(CategoryManager.category_map(sample[i], rev=True))
                sample_abrv = res + sample[template.num_nodes():]
                csvwriter.writerow([db_name] + sample_abrv)
                cases[tuple(sample_abrv)] = cases.get(tuple(sample_abrv),0) + 1

    if save_stats:
        with open(save_path + "_stats.log", 'w') as f:
            pprint(cases, f)
        
    

if __name__ == "__main__":

    seed = random.randint(200,1000)

    template = ThreeRelTemplate

    # Config
    train_kwargs = {
        'num_partitions': 7,#10,
        'num_batches': 10,
        'save': True,
        'load_if_exists': False,
        'likelihood_thres': 0.1,
        'save_training_info': True,

        # spn_structure
        'num_decomps': 2,
        'num_subsets': 3,
        'num_mixtures': 2,
        'num_input_mixtures': 2,

        'template': template,

        'train_db': ['Freiburg', 'Saarbrucken']
    }

    spn_params = TbmExperiment.strip_spn_params(train_kwargs, train_kwargs['learning_algorithm'])
    template_spn = EdgeRelationTemplateSpn(train_kwargs['template'], seed=seed, **spn_params)
    
    exp = TbmExperiment(TOPO_MAP_DB_ROOT,
                        template_spn,
                        root_dir=GRAPHSPN_RESULTS_ROOT, name="TMP_VIEW_TMPL")

    exp.load_training_data(*train_kwargs['train_db'],
                           skip_unknown=CategoryManager.SKIP_UNKNOWN)
    
    samples_dict = exp._dataset.create_edge_relation_template_dataset(template_spn.template,
                                                                      num_partitions=10,
                                                                      **{'db_names': train_kwargs['train_db']})

    os.makedirs("./analysis/training", exist_ok=True)
    save_samples_to_csv(template, samples_dict, "./analysis/training/%s_samples.csv" % template.__name__, save_stats=True)
