#!/usr/bin/env python3
#
# Takes DGSM output marginals as input, produce
# GraphSPN marginals.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import re
import argparse
import time
import os, sys
import json, yaml
from pprint import pprint
import tensorflow as tf
from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.spn_template import NodeTemplateSpn, InstanceSpn
from deepsm.graphspn.tbm.template import NodeTemplate, PairTemplate, SingletonTemplate, ThreeNodeTemplate, StarTemplate
from deepsm.graphspn.tests.tbm.runner import TbmExperiment
from deepsm.graphspn.tests.runner import TestCase
import deepsm.util as util
import deepsm.experiments.paths as paths

from deepsm.experiments.common import COLD_ROOT, DGSM_RESULTS_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT

def normalize(a):
    return a / np.sum(a)

def load_likelihoods(results_dir, graph_id, categories, relax_level=None, return_dgsm_result=False):
    """
    Load likelihoods which are outputs from DGSM when feeding scans related to a graph
    identified by `graph_id`. The likelihoods should be normalized and sum to 1.

    Args:
        results_dir (str) directory for DGSM output for graph scan results
        graph_id (str) {building}_{seq_id}
        categories (list) list of string categories
        relax_level (None or float): If not none, adds this value to every likelihood value
                                     and then re-normalize all likelihoods (for each node).
    """
    with open(os.path.join(results_dir, "%s_likelihoods.json" % graph_id.lower())) as f:
        # lh is formatted as {node id -> [groundtruth, prediction, likelihoods(normalized)]}
        # We only need the likelihoods
        lh = json.load(f)

    total_cases = 0
    total_correct = 0
    dgsm_result = {}
        
    lh_out = {}
    for nid in lh:
        lh_out[int(nid)] = np.zeros((util.CategoryManager.NUM_CATEGORIES,))
        for i in range(len(lh[nid][2])):
            catg = categories[i]
            indx = util.CategoryManager.category_map(catg)
            lh_out[int(nid)][indx] = lh[nid][2][i]
        groundtruth = lh[nid][0]
        prediction = lh[nid][1]
        if groundtruth not in dgsm_result:
            dgsm_result[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
        if groundtruth == prediction:
            total_correct += 1
            dgsm_result[groundtruth][1] += 1
        total_cases += 1
        dgsm_result[groundtruth][0] += 1
        dgsm_result[groundtruth][2] = dgsm_result[groundtruth][1] / dgsm_result[groundtruth][0]
    dgsm_result['_total_cases_'] = total_cases
    dgsm_result['_total_correct_'] = total_correct
    dgsm_result['_overall_'] = total_cases / total_correct

    # Apply relaxation
    if relax_level is not None:
        for nid in lh_out:
            lh_out[nid] += relax_level
            lh_out[nid] = normalize(lh_out[nid])

    # Return
    if return_dgsm_result:
        return lh_out, dgsm_result
    else:
        return lh_out


def get_category_map_from_lh(lh):
    """lh is a dictionary { nid -> [ ... likelihood for class N... ]}"""
    category_map = {}   # nid -> numerical value of the category with highest likelihood
    for nid in lh:
        class_index = np.argmax(lh[nid])
        category_map[nid] = class_index
    return category_map


def normalize_marginals(marginals):
    result = {}
    for nid in marginals:
        likelihoods = np.array(marginals[nid]).flatten()
        normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                           (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
        result[nid] = normalized
    return result



class GraphSPNToyExperiment(TbmExperiment):
    
    def __init__(self, db_root, *spns, **kwargs):
        """
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.
        """
        super().__init__(db_root, *spns, **kwargs)

    class TestCase_Classification(TestCase):
        def __init__(self, experiment):
            super().__init__(experiment)

        def run(self, sess, *args, expand=False, **kwargs):
            """
            **kwargs:
                topo_map (TopologicalMap)
                graph_id (str)
                instance_spn (InstanceSpn)
                categories (list of strings)
                graph_results_dir (str)
                relax_level (float) optional
            """
            instance_spn = kwargs.get('instance_spn', None)
            topo_map = kwargs.get('topo_map', None)
            graph_id = kwargs.get('graph_id', None)
            categories = kwargs.get('categories', None)
            graph_results_dir = kwargs.get('graph_results_dir', None)
            relax_level = kwargs.get('relax_level', None)
            self._topo_map = topo_map
            self._graph_id = graph_id

            true_catg_map = topo_map.current_category_map()

            total_correct, total_cases = 0, 0
            __record = {
                'results':{
                    util.CategoryManager.category_map(k, rev=True):[0,0,0] for k in range(util.CategoryManager.NUM_CATEGORIES)
                },
                'instance':{}
            }
            
            # We query for all nodes using marginal inference
            print("Preparing inputs for marginal inference")
            query_lh, dgsm_result = load_likelihoods(graph_results_dir, graph_id, categories, relax_level=relax_level, return_dgsm_result=True)
            # Add uniform likelihoods for placeholders
            for nid in topo_map.nodes:
                if nid not in query_lh:
                    if topo_map.nodes[nid].placeholder:
                        query_lh[nid] = normalize(np.full((util.CategoryManager.NUM_CATEGORIES,), 1.0))
                    else:
                        raise ValueError("Node %d has no associated likelihoods. Something wrong in DGSM output?" % nid)
            query_nids = list(topo_map.nodes.keys())
            query = {k:-1 for k in query_nids}
            print("Performing marginal inference...")
            marginals = instance_spn.marginal_inference(sess, query_nids,
                                                        query, query_lh=query_lh)
            
            result_catg_map = {}
            for nid in query:
                result_catg_map[nid] = marginals[nid].index(max(marginals[nid]))
            
            for nid in true_catg_map:
                if topo_map.nodes[nid].placeholder:
                    continue  # skip placeholder when calculating accuracy
                true_catg = util.CategoryManager.category_map(true_catg_map[nid], rev=True) # (str)
                infrd_catg = util.CategoryManager.category_map(result_catg_map[nid], rev=True)

                if true_catg == infrd_catg:
                    __record['results'][true_catg][0] += 1
                    total_correct += 1
                __record['results'][true_catg][1] += 1
                __record['results'][true_catg][2] = __record['results'][true_catg][0] / __record['results'][true_catg][1]
                total_cases += 1
            __record['results']['_overall_'] = total_correct / max(total_cases,1)
            __record['results']['_total_correct_'] = total_correct
            __record['results']['_total_inferred_'] = total_cases

            # Record
            __record['instance']['_marginals_'] = marginals
            __record['instance']['_marginals_normalized_'] = normalize_marginals(marginals)
            __record['instance']['likelihoods'] = query_lh
            __record['instance']['true'] = true_catg_map
            __record['instance']['query'] = get_category_map_from_lh(query_lh)
            __record['instance']['result'] = result_catg_map
            __record['dgsm_results'] = dgsm_result
            
            self._record = __record

        def _report(self):
            return self._record

        def save_results(self, save_path):
            def save_vis(topo_map, category_map, graph_id, save_path, name, consider_placeholders):
                floor = re.search("(seq|floor)[1-9]+", graph_id).group()
                db_name = re.search("(stockholm|freiburg|saarbrucken)", graph_id, re.IGNORECASE).group().capitalize()
                ColdMgr = util.ColdDatabaseManager(db_name, COLD_ROOT, gt_root=GROUNDTRUTH_ROOT)
                topo_map.assign_categories(category_map)
                rcParams['figure.figsize'] = 22, 14
                topo_map.visualize(plt.gca(), ColdMgr.groundtruth_file(floor, 'map.yaml'), consider_placeholders=consider_placeholders)
                plt.savefig(os.path.join(save_path, '%s_%s.png' % (graph_id, name)))
                plt.clf()
                print("Saved %s visualization for %s." % (name, graph_id))
                
            report = self._report()
            with open(os.path.join(save_path, "report.log"), "w") as f:
                yaml.dump(report['results'], f)
            with open(os.path.join(save_path, "test_instance.log"), "w") as f:
                pprint(report['instance'], stream=f)

            # Save visualizations
            save_vis(self._topo_map, report['instance']['true'], self._graph_id, save_path, 'groundtruth',  True)
            save_vis(self._topo_map, report['instance']['query'], self._graph_id, save_path, 'query', True)
            save_vis(self._topo_map, report['instance']['result'], self._graph_id, save_path, 'result', False)  # All nodes are no
            save_vis(self._topo_map, report['instance']['result'], self._graph_id, save_path, 'exclude_ph_result', True)  # All nodes are no
                
            return report


def run_experiment(seed, train_kwargs, test_kwargs, templates, exp_name,
                   amount=1, seq_id=None):
    """Run experiment

    Arguments:

    amount (int): amount of testing topo maps to load.
    """
    spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}
    
    spns = []
    for template in templates:
        tspn = NodeTemplateSpn(template, seed=seed, **spn_params)
        spns.append(tspn)

    # Build experiment object
    exp = GraphSPNToyExperiment(TOPO_MAP_DB_ROOT, *spns,
                                root_dir=GRAPHSPN_RESULTS_ROOT, name=exp_name)
    
    util.print_in_box(["Experiment %s" % exp_name])
    util.print_banner("Start", ch='v')
    
    # Load training data
    exp.load_training_data(*train_kwargs['db_names'], skip_unknown=True)
    spn_paths = {model.template.__name__:exp.model_save_path(model) for model in spns}

    try:
        with tf.Session() as sess:
            # Train models
            train_info = exp.train_models(sess, **train_kwargs)

            # Relax the single template spn's prior    
            single_node_spn = exp.get_model(SingletonTemplate)
            print("Relaxing prior...")
            SpnModel.make_weights_same(sess, single_node_spn.root)

            # remove inputs; necessasry for constructing instance spn
            for template_spn in spns:                
                template_spn._conc_inputs.set_inputs()

            # Test
            exp.load_testing_data(test_kwargs['db_name'], skip_unknown=True)
            test_instances = exp.get_test_instances(db_name=test_kwargs['db_name'],
                                                    amount=amount,
                                                    seq_id=seq_id,
                                                    auto_load_splitted=seq_id is None)
            for db_seq_id in test_instances:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                util.print_banner(seq_id, length=40)
                topo_map = test_instances[db_seq_id]
                spns_tmpls = [(spns[i], spns[i].template) for i in range(len(spns))]
                visp_dirpath = os.path.join(exp.root_path,
                                            "results",
                                            "partitions_%s_%s_%s" % (db_name, test_kwargs['timestamp'], seq_id))
                os.makedirs(visp_dirpath, exist_ok=True)
                instance_spn = InstanceSpn(topo_map, sess, *spns_tmpls, num_partitions=test_kwargs['num_partitions'],
                                           seq_id=seq_id, spn_paths=spn_paths, divisions=8,
                                           visualize_partitions_dirpath=visp_dirpath, db_name=db_name)
                test_kwargs['instance_spn'] = instance_spn
                test_kwargs['graph_id'] = db_name + "_" + seq_id
                test_kwargs['topo_map'] = topo_map
                test_kwargs['categories'] = train_kwargs['trained_categories']
                test_kwargs['subname'] = '%s-%s' % (test_kwargs['test_name'], seq_id)

                instance_spn.expand()
                print("Initializing Ops. Will take a while...")
                instance_spn.init_ops(no_mpe=True)

                report = exp.test(GraphSPNToyExperiment.TestCase_Classification, sess, **test_kwargs)
    except KeyboardInterrupt as ex:
        print("Terminating...\n")

def init_experiment(templates, exp_name, seed, spn_params):
    return exp


def main():
    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('-s', '--seed', type=int, help="Seed of randomly generating SPN structure. Default 100",
                        default=100)
    parser.add_argument('-e', '--exp-name', type=str, help="Name to label this experiment. Default: GraphSPNToyExperiment",
                        default="GraphSPNToyExperiment")
    parser.add_argument('-r', '--relax-level', type=float, help="Adds this value to every likelihood value and then re-normalize all likelihoods (for each node)")
    parser.add_argument('-N', '--num-test-seqs', type=int, help="Total number of sequences to test on", default=-1)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    classes = []
    for i in range(util.CategoryManager.NUM_CATEGORIES):
        classes.append(util.CategoryManager.category_map(i, rev=True))


    # Global configs:
    # Configuration
    train_kwargs = {
        'trained_categories': classes,
        'load_if_exists': True,
        'shuffle': True,
        "save": True,

        # spn structure
        "num_decomps": 1,
        "num_subsets": 3,
        "num_mixtures": 5,
        "num_input_mixtures": 5
    }
    test_kwargs = {
        'test_name': 'fullstockholm',
        'num_partitions': 5,
        'timestamp': timestamp,
        'relax_level': args.relax_level if args.relax_level else None
    }

    templates = [SingletonTemplate, PairTemplate, ThreeNodeTemplate, StarTemplate]

    # Run all experiments for the entire Stockholm building
    db_name = "Stockholm"
    floors = {4, 5, 6, 7}
    num_seqs_tested = 0
    for test_floor in sorted(floors):
        for seq_id in os.listdir(os.path.join(TOPO_MAP_DB_ROOT, "%s%d" % (db_name, test_floor))):

            train_floors_str = "".join(sorted(map(str, floors - {test_floor})))
            train_kwargs['db_names'] = ["%s%s" % (db_name, train_floors_str)]
            test_kwargs['db_name'] = "%s%d" % (db_name, test_floor)
            
            test_kwargs['graph_results_dir'] \
                = paths.path_to_dgsm_result_same_building(util.CategoryManager.NUM_CATEGORIES,
                                                          db_name,
                                                          "graphs",
                                                          train_floors_str,
                                                          test_floor)
            graph_id = "%s%d_%s" % (db_name.lower(), test_floor, seq_id)
            query_lh, dgsm_result = load_likelihoods(test_kwargs['graph_results_dir'],
                                                     graph_id,
                                                     train_kwargs['trained_categories'],
                                                     relax_level=test_kwargs['relax_level'],
                                                     return_dgsm_result=True)
            exp_name = args.exp_name
            run_experiment(args.seed, train_kwargs, test_kwargs, templates, exp_name,
                           seq_id=seq_id)
            num_seqs_tested += 1
            if num_seqs_tested >= args.num_test_seqs:
                print("Test sequence limit of %d is reached" % num_seqs_tested)
                return


if __name__ == "__main__":
    main()
