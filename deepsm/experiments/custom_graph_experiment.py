# Run GraphSPN on custom graph specified by a .ug file.
# See graphspn/tbm/graph_builder.py:build_graph for file
# format specification.
#
# author: Kaiyu Zheng

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import tensorflow as tf
import argparse
import yaml
import time
import os, re, sys
from pprint import pprint
from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.spn_template import NodeTemplateSpn
from deepsm.graphspn.tbm.spn_instance import NodeTemplateInstanceSpn
from deepsm.graphspn.tbm.template import NodeTemplate, PairTemplate, SingletonTemplate, ThreeNodeTemplate, StarTemplate
from deepsm.graphspn.tests.tbm.runner import TbmExperiment, normalize_marginals, normalize_marginals_remain_log, get_category_map_from_lh
from deepsm.graphspn.tests.runner import TestCase
from deepsm.graphspn.tbm.graph_builder import build_graph
from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT
import deepsm.util as util

class CustomGraphExperiment(TbmExperiment):
    
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
                likelihoods (map from node id to likelihood array)
                graph_id (str)
                instance_spn (InstanceSpn)
                relax_level (float) optional
            """
            instance_spn = kwargs.get('instance_spn', None)
            topo_map = kwargs.get('topo_map', None)
            query_lh = kwargs.get('likelihoods', None)
            graph_id = kwargs.get('graph_id', None)
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
                    __record['results'][true_catg][1] += 1
                    total_correct += 1
                __record['results'][true_catg][0] += 1
                __record['results'][true_catg][2] = __record['results'][true_catg][1] / __record['results'][true_catg][0]
                total_cases += 1
            __record['results']['_overall_'] = total_correct / max(total_cases,1)
            __record['results']['_total_correct_'] = total_correct
            __record['results']['_total_inferred_'] = total_cases

            # Record
            __record['instance']['_marginals_'] = marginals
            __record['instance']['_marginals_normalized_'] = normalize_marginals(marginals)
            __record['instance']['_marginals_normalized_log_'] = normalize_marginals_remain_log(marginals)
            __record['instance']['likelihoods'] = query_lh
            __record['instance']['true'] = true_catg_map
            __record['instance']['query'] = get_category_map_from_lh(query_lh)
            __record['instance']['result'] = result_catg_map
            
            self._record = __record

        def _report(self):
            return self._record

        def save_results(self, save_path):
            def save_vis(topo_map, category_map, graph_id, save_path, name, consider_placeholders):
                floor = re.search("(seq|floor)[1-9]+", graph_id).group()
                db_name = re.search("(fake)", graph_id, re.IGNORECASE).group().capitalize()
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
                yaml.dump(report['instance'], f)

            # Save visualizations
            save_vis(self._topo_map, report['instance']['true'], self._graph_id, save_path, 'groundtruth',  True)
            save_vis(self._topo_map, report['instance']['query'], self._graph_id, save_path, 'query', True)
            save_vis(self._topo_map, report['instance']['result'], self._graph_id, save_path, 'result', False)  # All nodes are no
                
            return report



def run_experiment(seed, train_kwargs, test_kwargs, templates, exp_name):

    # specify path to a .ug file
    # build topological map
    # specify what templates to use
    # train template spns for those templates
    # use those templates to partition the graph.
    # feed likelihood in to graphspn
    # produce inference result
    # save the parameters for the graphspn

    spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}
    
    spns = []
    for template in templates:
        tspn = NodeTemplateSpn(template, seed=seed, **spn_params)
        spns.append(tspn)

    # Build experiment object
    exp = CustomGraphExperiment(TOPO_MAP_DB_ROOT, *spns,
                                root_dir=GRAPHSPN_RESULTS_ROOT, name=exp_name)

    util.print_in_box(["Experiment %s" % exp_name])
    util.print_banner("Start", ch='v')

    # Load training data
    exp.load_training_data(*train_kwargs['db_names'], skip_unknown=True, skip_placeholders=False)
    spn_paths = {model.template.__name__:exp.model_save_path(model) for model in spns}

    # Obtain test graph
    topo_map, likelihoods = build_graph(test_kwargs['graph_file'])
    db_name, seq_id = test_kwargs['db_seq_id'].split("-")[0], test_kwargs['db_seq_id'].split("-")[1]

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
            spns_tmpls = [(spns[i], spns[i].template) for i in range(len(spns))]
            visp_dirpath = os.path.join(exp.root_path,
                                        "results",
                                        "partitions_%s_%s" % (test_kwargs['db_seq_id'], test_kwargs['timestamp']))
            os.makedirs(visp_dirpath, exist_ok=True)
            instance_spn = NodeTemplateInstanceSpn(topo_map, sess, *spns_tmpls, num_partitions=test_kwargs['num_partitions'],
                                                   seq_id=seq_id, divisions=8,
                                                   visualize_partitions_dirpath=visp_dirpath, db_name=db_name)
            
            test_kwargs['instance_spn'] = instance_spn
            test_kwargs['graph_id'] = db_name + "_" + seq_id
            test_kwargs['topo_map'] = topo_map
            test_kwargs['subname'] = '%s-%s' % (test_kwargs['test_name'], seq_id)
            test_kwargs['likelihoods'] = likelihoods
            
            instance_spn.expand()
            
            print("Initializing Ops. Will take a while...")
            instance_spn.init_ops(no_mpe=True)

            report = exp.test(CustomGraphExperiment.TestCase_Classification, sess, **test_kwargs)
    except KeyboardInterrupt as ex:
        print("Terminating...\n")


def custom_graph():
    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('graph_file', type=str, help="Path to .ug file that specifies an undirected graph")
    parser.add_argument('-s', '--seed', type=int, help="Seed of randomly generating SPN structure. Default 100",
                        default=100)
    parser.add_argument('-e', '--exp-name', type=str, help="Name to label this experiment. Default: CustomGraphExperiment",
                        default="CustomGraphExperiment")
    parser.add_argument('-r', '--relax-level', type=float, help="Adds this value to every likelihood value and then re-normalize all likelihoods (for each node)")
    args = parser.parse_args(sys.argv[2:])

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Global configs:
    # Configuration
    train_kwargs = {
        'load_if_exists': True,
        'shuffle': True,
        "save": True,

        # spn structure
        "num_decomps": 1,
        "num_subsets": 2,
        "num_mixtures": 2,
        "num_input_mixtures": 5
    }
    test_kwargs = {
        'test_name': os.path.basename(args.graph_file),
        'num_partitions': 2,
        'timestamp': timestamp,
        'db_seq_id': 'Fake-floor1_happy',
        'graph_file': args.graph_file,
        'relax_level': args.relax_level if args.relax_level else None
    }

    templates = [SingletonTemplate, PairTemplate, ThreeNodeTemplate]#, StarTemplate]

    # Remember to change the databases for training as you want.
    all_db = {'Stockholm456'} #}#{'Freiburg', 'Saarbrucken', 'Stockholm'}
    train_kwargs['db_names'] = sorted(['Freiburg', 'Stockholm', 'Saarbrucken'])
    test_kwargs['db_name'] = 'Fake'

    exp_name = args.exp_name
    run_experiment(args.seed, train_kwargs, test_kwargs, templates, exp_name)
