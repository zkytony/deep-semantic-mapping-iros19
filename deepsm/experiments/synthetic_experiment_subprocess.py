#!/usr/bin/env python
#
# Big Question: “Does introducing view variables have any positive effect
#                on the result?”
#
# Test B: Test using the instance SPN
#   Training set = topo maps from two databases
#   Testing set = topo maps from the other database
#
#   During testing:
#     Basically do what is in test_template_spn.py for the scale of the
#     whole graph
#
# author: Kaiyu Zheng

import sys, os
import yaml
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
import argparse

import libspn as spn

from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.graphspn.tbm.spn_template import NodeTemplateSpn, EdgeRelationTemplateSpn
from deepsm.graphspn.tbm.spn_instance import NodeTemplateInstanceSpn, EdgeRelationTemplateInstanceSpn
from deepsm.graphspn.tbm.template import NodeTemplate, PairTemplate, SingletonTemplate, ThreeNodeTemplate, StarTemplate, \
    EdgeRelationTemplate, ThreeRelTemplate, RelTemplate, SingleRelTemplate, SingleTemplate
from deepsm.graphspn.tests.tbm.runner import TbmExperiment, doorway_policy, random_policy, random_fixed_policy, random_fixed_plus_placeholders, get_noisification_level
from deepsm.graphspn.tests.runner import TestCase
from deepsm.util import CategoryManager, print_banner, print_in_box, ColdDatabaseManager

########### Constants ###########
# Paths. Need to change if you work on a different machine.
from deepsm.experiments.common import COLD_ROOT, DGSM_RESULTS_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT
# Constants for tests
INF_PLACEHOLDER = 1
INF_RAND = 2
INF_FIXED_RAND = 3
INF_DW   = 4
LH  = 10
RAND_RATE_TEST   = 100
NOISE_LEVEL_TEST = 200
NOVELTY = 300
# Inference types
MARGINAL = 'marginal'
MPE = 'mpe'


class InstanceSpnExperiment(TbmExperiment):

    """
    As described above.
    """

    def __init__(self, db_root, *spns, **kwargs):
        """
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.
        """
        super().__init__(db_root, *spns, **kwargs)


    def get_stats(self, *args, **kwargs):
        overall_stats = {}
        for subname in self._completed:  # The subname here refers to the {General} part of the full subname.
            # There is no synthesis of stats across subnames.
            total_instances = 0
            stats = {'_overall_':0, '_total_correct_':0, '_total_inferred_':0,
                     '_total_instances_': 0, '_test_db_': self.test_db}
            for case_name in self._completed[subname]:
                # Read the report yaml file
                report = None
                try:
                    with open(os.path.join(self._root_path, 'results', case_name, 'report.log')) as f:
                        report = yaml.load(f)
                except Exception as ex:
                    # Somehow loading this one failed. Ignore it.
                    print("Failed to open/load report for %s. Exception: %s" % (case_name, ex))
                    print("  This is expected if you are running Novelty Detection task.")
                    continue
                total, correct = 0, 0
                for key in report:
                    if not key.startswith("_"):
                        if key not in stats:
                            stats[key] = [0,0,0]
                        stats[key][0] += report[key][0]
                        stats[key][1] += report[key][1]
                        stats[key][2] = stats[key][0] / max(1, stats[key][1])
                        total += report[key][1]
                        correct += report[key][0]
                stats['_total_inferred_'] += total
                stats['_total_correct_'] += correct
                total_instances += 1
            stats['_overall_'] = stats['_total_correct_'] / max(1, stats['_total_inferred_'])
            stats['_total_instances_'] = total_instances
            stats['_completed_cases_'] = list(self._completed[subname])
            overall_stats[subname] = stats
        return overall_stats

    # Infer placeholders
    # Local classification correction
    # Novelty detection


    ##################################################
    #                                                #
    #     TestCase_GraphInference                    #
    #                                                #
    ##################################################
    class TestCase_GraphInference(TestCase):
        
        def __init__(self, experiment):
            super().__init__(experiment)
            self._topo_map = None
            self._seq_id = None
            self._db_name = None
            self._record = None



        def run(self, sess, *args, expand=False, **kwargs):
            """
            sess (tf.Session): tensorflow session.

            *args:

            **kwargs:
               inference_type (string): Type of inference to perform on the instance spn. 'MPE' or 'Marginal'.
               expand (bool): Expand structure after initializing it?  Default False.
               func_mask_graph (function): Required. Creates a sample of masked node labels,
                                           in the format specified in spn_template:mpe_inference().
                                           (parameters: topo_map)
               high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
               low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                                       and the other (untrue) classes.

               topo_map (TopologicalMap)
               seq_id (str)
               db_name (str)
               instance_spn (InstanceSpn)
               uniform_for_incorrect (bool)
               consider_placeholders (bool): If True, will set the likelihoods for a placeholder node to be uniform, regardless
                                             of the high/low likelihoods setting.
            """
            topo_map = kwargs.get('topo_map', None)
            seq_id = kwargs.get('seq_id', None)
            db_name = kwargs.get('db_name', None)
            inference_type = kwargs.get('inference_type', None)
            instance_spn = kwargs.get('instance_spn', None)
            uniform_for_incorrect = kwargs.get('uniform_for_incorrect', False)
            consider_placeholders = kwargs.get('consider_placeholders', False)
            self._topo_map = topo_map
            self._seq_id = seq_id
            self._db_name = db_name
            self._consider_placeholders = consider_placeholders
            
            if expand:
                high_likelihood_correct = kwargs.get("high_likelihood_correct")
                low_likelihood_correct = kwargs.get("low_likelihood_correct")
                high_likelihood_incorrect = kwargs.get("high_likelihood_incorrect")
                low_likelihood_incorrect = kwargs.get("low_likelihood_incorrect")


            # groundtruth
            true_catg_map = topo_map.current_category_map()

            # Occlude nodes based on given function (FROM NOW ON --- topo map's node label may be -1)
            func_mask_graph = kwargs.get('func_mask_graph')
            func_mask_graph(topo_map, **kwargs.get('func_mask_params', {}))

            query_catg_map = topo_map.current_category_map()

            __record = {
                'results':{
                    CategoryManager.category_map(k, rev=True):[0,0,0] for k in range(CategoryManager.NUM_CATEGORIES)
                },
                'instance':{}
            }
            total_correct, total_cases = 0, 0
            query_lh = None
            if expand:
                # Create likelihoods. Note that now, topo_map contains masked classes '-1'.
                query_lh = TbmExperiment.create_instance_spn_likelihoods(NodeTemplate.code(), topo_map, true_catg_map,
                                                                         high_likelihood_correct, low_likelihood_correct,
                                                                         high_likelihood_incorrect, low_likelihood_incorrect,
                                                                         uniform_for_incorrect=uniform_for_incorrect,
                                                                         consider_placeholders=consider_placeholders)
            if inference_type == MPE:
                query = topo_map.current_category_map()
                if expand:
                    result_catg_map = instance_spn.mpe_inference(sess, query, query_lh)
                else:                    
                    result_catg_map = instance_spn.mpe_inference(sess, query)

            elif inference_type == MARGINAL:
                query = topo_map.current_category_map()
                query_nids = []
                for nid in query:
                    if query[nid] == -1:
                        query_nids.append(nid)
                if expand:
                    # Query should be all -1. And we query for all nodes.
                    query_nids = list(topo_map.nodes.keys())
                    query = {k:-1 for k in query_nids}#TbmExperiment.create_category_map_from_likelihoods(0, query_lh) #
                marginals = instance_spn.marginal_inference(sess, query_nids, query, query_lh=query_lh)
                result_catg_map = {}
                for nid in query:
                    if expand:
                        result_catg_map[nid] = marginals[nid].index(max(marginals[nid]))
                    else:
                        if query[nid] == -1:
                            result_catg_map[nid] = marginals[nid].index(max(marginals[nid]))
                        else:
                            result_catg_map[nid] = query[nid]
                __record['instance']['_marginals_'] = marginals
            else:
                raise ValueError("Unrecognized inference type %s" % inference_type)
                

            # Record
            for nid in true_catg_map:
                true_catg = CategoryManager.category_map(true_catg_map[nid], rev=True) # (str)
                infrd_catg = CategoryManager.category_map(result_catg_map[nid], rev=True)

                # Determine if we will consider the node-inference case for nid.
                count_case = False
                if consider_placeholders:
                    if topo_map.nodes[nid].placeholder:
                        assert query_catg_map[nid] == -1  # Must infer placeholder in this case.
                        count_case = True
                elif expand or query_catg_map[nid] == -1:
                    count_case = True
                if count_case:
                    if true_catg == infrd_catg:
                        __record['results'][true_catg][0] += 1
                        total_correct += 1
                    __record['results'][true_catg][1] += 1
                    __record['results'][true_catg][2] = __record['results'][true_catg][0] / __record['results'][true_catg][1]
                    total_cases += 1
                else:
                    # Otherwise, this is not a query node
                    if not (consider_placeholders and query_catg_map[nid] == -1):  # When consider placeholders, we keep the network's results on incorrect classes
                        result_catg_map[nid] = true_catg_map[nid]

            __record['instance']['true'] = true_catg_map
            __record['instance']['query'] = query_catg_map
            __record['instance']['result'] = result_catg_map

            # if self._experiment.template_mode == EdgeTemplate.code():
            #     __record['instance']['raw_result'] = counts
            if expand:
                __record['instance']['likelihoods'] = query_lh
            if 'func_mask_policy' in kwargs:
                __record['instance']['policy'] = kwargs.get('func_mask_policy').__name__     
            __record['results']['_overall_'] = total_correct / max(total_cases,1)
            __record['results']['_total_correct_'] = total_correct
            __record['results']['_total_inferred_'] = total_cases
            
            self._record = __record
        ## End run


        def _report(self):
            return self._record


        def save_results(self, save_path):

            def save_vis(topo_map, category_map, db_name, seq_id, save_path, name, consider_placeholders):
                ColdMgr = ColdDatabaseManager(db_name, COLD_ROOT)
                topo_map.assign_categories(category_map)
                rcParams['figure.figsize'] = 22, 14
                topo_map.visualize(plt.gca(), ColdMgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), consider_placeholders=consider_placeholders)
                plt.savefig(os.path.join(save_path, '%s_%s_%s.png' % (db_name, seq_id, name)))
                plt.clf()
                print("Saved %s visualization for %s:%s." % (name, db_name, seq_id))

            # report
            report = self._report()

            with open(os.path.join(save_path, "report.log"), "w") as f:
                yaml.dump(report['results'], f)
            with open(os.path.join(save_path, "test_instance.log"), "w") as f:
                pprint(report['instance'], stream=f)

            # Save visualizations
            save_vis(self._topo_map, report['instance']['true'], self._db_name, self._seq_id, save_path, 'groundtruth',  False)
            save_vis(self._topo_map, report['instance']['query'], self._db_name, self._seq_id, save_path, 'query', self._consider_placeholders)
            save_vis(self._topo_map, report['instance']['result'], self._db_name, self._seq_id, save_path, 'result', False)  # All nodes are not placeholders in result.
            return report



    ##################################################
    #                                                #
    #     TestCase_InferPlaceholders                 #
    #                                                #
    ##################################################
    class TestCase_InferPlaceholders(TestCase_GraphInference):
        
        def __init__(self, experiment):
            super().__init__(experiment)
            self._record = None

            
        def run(self, sess, *args, **kwargs):
            """
            Runs this test case. Construct inputs that occludes placeholder nodes,
            and feed them into the network. Use the MPE result to compare to the
            groundtruth. Count correctness per type.

            (Only run for one topological map because resources for Ops need to
             be freed to avoid slow behavior.)

            sess (tf.Session): tensorflow session.

            *args:
               
            **kwargs:
               topo_map (TopologicalMap)
               seq_id (str)
               instance_spn (InstanceSpn)
               expand (bool) Whether likelihoods are included for non-placeholder nodes.
            """
            def func_mask_placeholders(topo_map, **mp_kwargs):
                topo_map.mask_by_policy(random_fixed_plus_placeholders, **mp_kwargs)

            kwargs['func_mask_graph'] = func_mask_placeholders
            kwargs['consider_placeholders'] = True
            super(InstanceSpnExperiment.TestCase_InferPlaceholders, self).run(sess, *args, **kwargs)
    
            
    ##################################################
    #                                                #
    #     TestCase_LocalClassificationCorrection     #
    #                                                #
    ##################################################
    class TestCase_LocalClassificationCorrection(TestCase_GraphInference):
        
        
        def __init__(self, experiment):
            """
            """
            super().__init__(experiment)

            
        def run(self, sess, *args, **kwargs):
            """
            Runs this test case. For a given topological map test instance,
            construct an InstanceSpn for it using the template models trained
            for this experiment. Expand it, then initialize Ops for it.
            Then, construct inputs that occludes random nodes, and create a likelihoods
            vector that mimic incorrect classification behavior for local variable.
            Use the MPE result to compare to the groundtruth. Count correctness per type.

            (Only run for one topological map because resources for Ops need to
             be freed to avoid slow behavior.)

            *args:

            **kwargs:
               topo_map (TopologicalMap)
               seq_id (str)
               db_name (str)
               instance_spn (InstanceSpn)
               func_mask_policy (function): A function that takes as input a topo_map and a node decides if this
                                  node should be masked. (e.g. is it a doorway? percentage of masking?)
                                  (Required)
               other arguments that will be passed to the policy.
            """
            if 'func_mask_policy' not in kwargs:
                raise ValueError("No policy provided for generating likelihoods!")

            def func_mask_graph(topo_map, **mp_kwargs):  # mp_kwargs: mask policy parameters
                topo_map.mask_by_policy(kwargs.get("func_mask_policy"), **mp_kwargs)

            kwargs['func_mask_graph'] = func_mask_graph
            super(InstanceSpnExperiment.TestCase_LocalClassificationCorrection, self).run(sess, *args, **kwargs, expand=True)
            
    ##################################################
    #                                                #
    #     TestCase_InferMissing                      #
    #                                                #
    ##################################################
    class TestCase_InferMissing(TestCase_GraphInference):
        
        
        def __init__(self, experiment):
            """
            """
            super().__init__(experiment)

            
        def run(self, sess, *args, **kwargs):
            """
            Runs this test case. For a given topological map test instance,
            construct an InstanceSpn for it using the template models trained
            for this experiment. Expand it, then initialize Ops for it. Then,
            occlude nodes based on given policy. There is no likelihoods involved.
            Use the MPE result to compare to the groundtruth. Count correctness per type.

            (Only run for one topological map because resources for Ops need to
             be freed to avoid slow behavior.)

            *args:
               topo_map (TopologicalMap)
               seq_id (str)
               db_name (str)
               instance_spn (InstanceSpn)

            **kwargs:
               func_mask_policy (function): A function that takes as input a topo_map and a node decides if this
                                  node should be masked. (e.g. is it a doorway? percentage of masking?)
                                  (Required)
               other arguments that will be passed to the policy.
            """
            if 'func_mask_policy' not in kwargs:
                raise ValueError("No policy provided for generating likelihoods!")

            def func_mask_graph(topo_map, **mp_kwargs):  # mp_kwargs: mask policy parameters
                topo_map.mask_by_policy(kwargs.get("func_mask_policy"), **mp_kwargs)

            kwargs['func_mask_graph'] = func_mask_graph
            super(InstanceSpnExperiment.TestCase_InferMissing, self).run(sess, *args, **kwargs, expand=False)


    ##################################################
    #                                                #
    #     TestCase_NoveltyDetection                  #
    #                                                #
    ##################################################
    class TestCase_NoveltyDetection(TestCase):

        """
        The test results are not saved in `report.log` as in other tasks. They are
        stored in `novelty.log`.
        """
        
        def __init__(self, experiment):
            super().__init__(experiment)
            self._topo_map = None
            self._seq_id = None


            
        def run(self, sess, *args, **kwargs):
            """
            Swap classes, and check likelihood values.
            Construct inputs based on a graph where a pair of
            classes swapped location. Compare the original likelihood vs swapped
            location likelihoods.

            **kwargs:
               num_partitions (int): 
               load_if_exists (bool): Loads the spn from a saved spn file. The
                                      saved path is conventional.
               save (bool): Saves the spn (if not already loaded from a file)
                            to conventional path.
               topo_map (TopologicalMap)
               seq_id (str)
               db_name (str)
               instance_spn (InstanceSpn)
               cases (list: a list of tuples of two category names, which will be swapped.
            """
            topo_map = kwargs.get('topo_map', None)
            seq_id = kwargs.get('seq_id', None)
            db_name = kwargs.get('db_name', None)
            instance_spn = kwargs.get('instance_spn', None)
            cases = kwargs.get('cases', None)
            self._topo_map = topo_map
            self._seq_id = seq_id
            self._db_name = db_name
            
            __record = {
                'results':{},
                'instance': {}
            }
            true_catg_map = topo_map.current_category_map()
            __record['instance']['groundtruth'] = true_catg_map

            for swapped_classes in cases:
                c1, c2 = swapped_classes
                print("Swap %s and %s" % (c1, c2))
                topo_map.swap_classes(swapped_classes)

                if self._experiment.template_mode == 0: ## NodeTemplate
                    query = topo_map.current_category_map()

                likelihood_val = float(instance_spn.evaluate(sess, query)[0])
                __record['results'][swapped_classes] = likelihood_val
                topo_map.reset_categories()

            # Record groundtruth
            likelihood_val = float(instance_spn.evaluate(sess, true_catg_map)[0])
            __record['results']['groundtruth'] = likelihood_val
            self._record = __record
            
    
        def _report(self):
            return self._record


        def save_results(self, save_path):
            # report
            report = self._report()

            with open(os.path.join(save_path, "novelty.log"), "w") as f:
                yaml.dump(report, f)

            return report
            
    

def run_node_template_experiment(seed, train_kwargs, test_kwargs, to_do, amount=1, num_rounds=3, name=None, seq_id=None):
    """
    Arguments:
    amount (int): the number of test instances to load.
    num_rounds (int): the number of inference tasks for each test instance. Only used for INF_RAND and INF_FIXED_RAND.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    train_kwargs['timestamp'] = timestamp
    test_kwargs['timestamp'] = timestamp
    
    print_in_box(["NodeTemplate experiments"], ho="+", vr="+")

    spn_params = TbmExperiment.strip_spn_params(train_kwargs, train_kwargs['learning_algorithm'])

    template_spns = []
    for template in train_kwargs['templates']:
        tspn = NodeTemplateSpn(template, seed=seed, **spn_params)
        template_spns.append(tspn)

    run_experiments(train_kwargs, test_kwargs, to_do, *template_spns,
                    template_mode=NodeTemplate.code(), amount=amount, num_rounds=num_rounds, name=name, seq_id=seq_id)


def run_edge_relation_template_experiment(seed, train_kwargs, test_kwargs, to_do, amount=1, num_rounds=3, name=None, seq_id=None):
    """
    Arguments:
    amount (int): the number of test instances to load.
    num_rounds (int): the number of inference tasks for each test instance. Only used for INF_RAND and INF_FIXED_RAND.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    train_kwargs['timestamp'] = timestamp
    test_kwargs['timestamp'] = timestamp
    
    print_in_box(["EdgeRelationTemplate experiments"], ho="+", vr="+")

    spn_params = TbmExperiment.strip_spn_params(train_kwargs, train_kwargs['learning_algorithm'])

    template_spns = []
    for template in train_kwargs['templates']:
        tspn = EdgeRelationTemplateSpn(template, seed=seed, **spn_params)
        template_spns.append(tspn)

    run_experiments(train_kwargs, test_kwargs, to_do, *template_spns,
                    template_mode=NodeTemplate.code(), amount=amount, num_rounds=num_rounds, name=name, seq_id=seq_id)


def run_experiments(train_kwargs, test_kwargs, to_do,
                    *spns, amount=1, num_rounds=3, template_mode=None, name=None, seq_id=None):
    """
    Runs an experiment for instance SPN

    seed (random.Random): random generator instance used to generate structure for template spn.
    train_kwargs (dict): dictionary for training parameters for template SPNs.
        'db_names': (list)
        'num_partitions': (int)
        'num_batches': (int)
        'save': (bool)5C
        'load_if_exists': (bool)
        'likelihood_thres': (float)
        'save_training_info': (bool)

        # spn_structure
        'num_decomps': (int)
        'num_subsets': (int)
        'num_mixtures': (int)
        'num_input_mixtures': (int

    test_kwargs (dict): dictionary for testing parameters
        'db_name': (str)
        'test_name': (str)
        'num_partitions': (int),
        'high_likelihood_correct': (2-tuple) e.g. (0.5, 0.80),
        'low_likelihood_correct': (2-tuple) e.g. (0.20, 0.50),
        'high_likelihood_incorrect': (2-tuple) e.g. (0.5, 0.80),
        'low_likelihood_incorrect': (2-tuple) e.g. (0.20, 0.50),
        'tiny_size': (int). Optional. Number of nodes on the minigraph. If provided,
                            will load minigraphs instead of full graphs.

    Arguments:

    *spns: list of untrained template spns
    novelty (bool): True if do novelty detection test case
    load_if_exists (bool): Set to true if want to load the instance spn from file if it exists
    save (bool): Set to true if want to save the instance spn to a file after initialization.
    amount (int): amount of testing topo maps to load.
    num_rounds (int): the number of inference tasks for each test instance. Only used for INF_RAND and INF_FIXED_RAND.
    train_dbs (list): List of db names for training
    test_db (str): a string name of the db for testing
    template_mode (int): set to 0 if this experiment is for node template. 1 for edge template.
    no_likelihoods (bool): Ignored if do_rand_rate_test is False. If not ignored and True, will
                           run the InferMissing test case.
    mix_ratio (float): See runner.py:load_training_data for definition.
    """
    if name is None:
        if template_mode == NodeTemplate.code():
            name = "C_InstanceSpnExperiment_NodeTemplate"

    exp = InstanceSpnExperiment(TOPO_MAP_DB_ROOT, *spns,
                                root_dir=GRAPHSPN_RESULTS_ROOT, name=name)
    print_in_box(["Experiment %s" % name])
    print_banner("Start", ch='v')

    exp.load_training_data(*train_kwargs['db_names'], skip_unknown=train_kwargs['skip_unknown'],
                           segment=train_kwargs['segment'], mix_ratio=train_kwargs['mix_ratio'])
    spn_paths = {model.template.__name__:exp.model_save_path(model) for model in spns}
    try:
        # Prepare trained spn for instance spn build-up
        with tf.Session() as sess:
            train_info = exp.train_models(sess, **train_kwargs)

            # Relax priors for simple template SPNs:
            if exp._template_mode == NodeTemplate.code():
                # Relax the single template spn's prior
                single_node_spn = exp.get_model(SingletonTemplate)
                print("Relaxing prior...")
                SpnModel.make_weights_same(sess, single_node_spn.root)
            elif exp._template_mode == EdgeRelationTemplate.code():
                tspn11 = exp.get_model(SingleRelTemplate)
                tspn10 = exp.get_model(SingleTemplate)
                tspn01 = exp.get_model(RelTemplate)

                print("Relaxing prior...")
                SpnModel.make_weights_same(sess, tspn11.root)
                SpnModel.make_weights_same(sess, tspn10.root)
                SpnModel.make_weights_same(sess, tspn01.root)
            
            for template_spn in spns:                
                # remove inputs
                template_spn._conc_inputs.set_inputs()
                
            tiny_size = test_kwargs['tiny_size'] if 'tiny_size' in test_kwargs else 0
            exp.load_testing_data(test_kwargs['db_name'], skip_unknown=train_kwargs['skip_unknown'], tiny_size=tiny_size, segment=train_kwargs['segment'])
            test_instances = exp.get_test_instances(db_name=test_kwargs['db_name'], amount=amount, seq_id=seq_id, auto_load_splitted=seq_id is None)
            for db_seq_id in test_instances:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                print_banner(seq_id, length=40)

                # Create instance SPN
                topo_map = test_instances[db_seq_id]
                true_catg_map = topo_map.current_category_map()
                spns_tmpls = [(spns[i], spns[i].template) for i in range(len(spns))]
                visp_dirpath = os.path.join(exp.root_path, "results", "partitions_%s_%s_%s" % (db_name, test_kwargs['timestamp'], seq_id))
                if not os.path.exists(visp_dirpath):
                    os.makedirs(visp_dirpath)
                if exp._template_mode == NodeTemplate.code():
                    instance_spn = NodeTemplateInstanceSpn(topo_map, sess, *spns_tmpls,
                                                           num_partitions=test_kwargs['num_partitions'],
                                                           seq_id=seq_id,
                                                           divisions=8,
                                                           visualize_partitions_dirpath=visp_dirpath,
                                                           db_name=db_name)
                    
                elif exp._template_mode == EdgeRelationTemplate.code():
                    instance_spn = EdgeRelationTemplateInstanceSpn(topo_map, sess, *spns_tmpls,
                                                                   num_partitions=test_kwargs['num_partitions'],
                                                                   seq_id=seq_id,
                                                                   divisions=8,
                                                                   visualize_partitions_dirpath=visp_dirpath,
                                                                   db_name=db_name)
                    
                test_kwargs['instance_spn'] = instance_spn
                test_kwargs['db_name'] = db_name
                test_kwargs['seq_id'] = seq_id
                test_kwargs['topo_map'] = topo_map

                instance_spn.print_params()

                if NOVELTY in to_do:
                    test_kwargs['subname'] = '%s_novelty--%s' % (test_kwargs['test_name'], seq_id)
                    instance_spn.init_ops(no_mpe=True)
                    exp.test(InstanceSpnExperiment.TestCase_NoveltyDetection, sess, **test_kwargs)
                else:
                    #--- Bunch of tests ---#
                    no_mpe = False
                    if test_kwargs['inference_type'] == MARGINAL:
                        print("No MPE states will be generated. Using Marginal inference.")
                        no_mpe = True

                    #--- Check if we want to do random inference task without likelihoods. If so, we won't do 'with likelihoods'. ---#
                    if LH not in to_do:                        
                        print("Initializing Ops. Will take a while...")
                        instance_spn.init_ops(no_mpe=no_mpe)
                        
                        #--- Check if want to do infer placeholder task ---#
                        if INF_PLACEHOLDER in to_do:
                            test_kwargs['expand'] = False
                            test_kwargs['subname'] = '%s_infer_placeholder_no_lh--%s' % (test_kwargs['test_name'], seq_id)
                            test_kwargs['func_mask_params']['rate_occluded'] = test_kwargs['occlude_rand_rate']
                            exp.test(InstanceSpnExperiment.TestCase_InferPlaceholders, sess, **test_kwargs)
                            # Need to revert whatever is done to the topo map's labels
                            topo_map.reset_categories()
                        
                        if INF_RAND in to_do:
                            for kk in range(num_rounds):
                                test_kwargs['func_mask_policy'] = random_policy
                                test_kwargs['subname'] = '%s_random_policy_no_lh--%s-%d' % (test_kwargs['test_name'], seq_id, kk)
                                test_kwargs['func_mask_params']['rand_rate'] = test_kwargs['occlude_rand_rate']
                                exp.test(InstanceSpnExperiment.TestCase_InferMissing, sess, **test_kwargs)
                                # Need to revert whatever is done to the topo map's labels
                                topo_map.reset_categories()

                        elif INF_FIXED_RAND in to_do:
                            for kk in range(num_rounds):  # Do multiple times
                                test_kwargs['func_mask_policy'] = random_fixed_policy
                                test_kwargs['subname'] = '%s_random_fixed_policy_no_lh--%s-%d' % (test_kwargs['test_name'], seq_id, kk)
                                test_kwargs['func_mask_params']['rate_occluded'] = test_kwargs['occlude_rand_rate']
                                exp.test(InstanceSpnExperiment.TestCase_InferMissing, sess, **test_kwargs)
                                # Need to revert whatever is done to the topo map's labels
                                topo_map.reset_categories()

                                
                    #--- Check if we want to do experiments that requires expanding the SPN. ---#
                    else:  
                        instance_spn.expand(use_cont_vars=True)
                        print("Initializing Ops. Will take a while...")
                        instance_spn.init_ops(no_mpe=no_mpe)
                        
                        #--- Check if want to do infer placeholder task ---#
                        if INF_PLACEHOLDER in to_do:
                            for kk in range(num_rounds):  # Do multiple times
                                test_kwargs['expand'] = True
                                test_kwargs['subname'] = '%s_infer_placeholder--%s-%d' % (test_kwargs['test_name'], seq_id, kk)
                                test_kwargs['func_mask_params']['rate_occluded'] = test_kwargs['occlude_rand_rate']
                                exp.test(InstanceSpnExperiment.TestCase_InferPlaceholders, sess, **test_kwargs)
                                # Need to revert whatever is done to the topo map's labels
                                topo_map.reset_categories()

                        #--- Check if we want to do random inference task with likelihoods. ---#
                        if INF_RAND in to_do:
                            for kk in range(num_rounds):  # Do multiple times
                                test_kwargs['func_mask_policy'] = random_policy
                                test_kwargs['subname'] = '%s_random_policy--%s-%d' % (test_kwargs['test_name'], seq_id, kk)
                                test_kwargs['func_mask_params']['rand_rate'] = test_kwargs['occlude_rand_rate']
                                exp.test(InstanceSpnExperiment.TestCase_LocalClassificationCorrection, sess, **test_kwargs)
                                # Need to revert whatever is done to the topo map's labels
                                topo_map.reset_categories()

                        #--- Check if we want to do random inference task with likelihoods for fixed number of nodes (tiny graph only). ---#
                        if INF_FIXED_RAND in to_do:
                            for kk in range(num_rounds):  # Do multiple times
                                test_kwargs['func_mask_policy'] = random_fixed_policy
                                test_kwargs['subname'] = '%s-%s_random_fixed_policy--%s-%d' % (test_kwargs['db_name'], test_kwargs['test_name'], seq_id, kk)
                                test_kwargs['func_mask_params']['rate_occluded'] = test_kwargs['occlude_rand_rate']
                                exp.test(InstanceSpnExperiment.TestCase_LocalClassificationCorrection, sess, **test_kwargs)
                                # Need to revert whatever is done to the topo map's labels
                                topo_map.reset_categories()

                        #--- Check if we want to do random inference task with likelihoods. Only runs if we are not running any test suite.---#
                        if INF_DW in to_do:
                            test_kwargs['func_mask_policy'] = doorway_policy
                            test_kwargs['subname'] = '%s_doorway_policy' % test_kwargs['test_name']
                            print("Doorway policy!")
                            exp.test(InstanceSpnExperiment.TestCase_LocalClassificationCorrection, sess, **test_kwargs)
                        #END
                    #END
                #END
            #END (with tf..)
        #END (for seq_id ...)
    except KeyboardInterrupt as ex:
        print("Terminating...\n")
    finally:
        stats = exp.get_stats()
        print_banner("Stats", ch="-", length=40)
        pprint(stats)
        with open(os.path.join(exp.root_path, 'results',
                               'overall_%s_%s.log' % (test_kwargs['test_name'], test_kwargs['timestamp'])), 'w') as f:
            yaml.dump(stats, f)


def load_kwargs_from_file(filepath, eval_keys={}):

    with open(filepath) as f:
        params = yaml.load(f)

    # Evaluate several keys
    for key in eval_keys:
        if key not in params:
            print("Warning: eval_key %s is not in the parameters." % key)
        else:
            params[key] = eval(params[key])
    return params

def main():
    """
    Keys in the train_kwargs file (Values are examples):

    train_kwargs = {
        "db_names": ["Stockholm", "Freiburg"], # default train db_names
        "num_partitions": 10,
        "num_batches": 10,
        "save": True,
        "load_if_exists": True,
        "likelihood_thres": 0.2,
        "will_upsample": TbmExperiment.will_upsample,
        "skip_unknown": CategoryManager.SKIP_UNKNOWN,

        # spn_structure
        "num_decomps": 1,
        "num_subsets": 3,
        "num_mixtures": 5,
        "num_input_mixtures": 5

       'segment': False (Optional)
    }

    Keys in the test_kwargs file (Values are examples):

    test_kwargs = {
        'db_name': 'Saarbrucken', # default test db
        'test_name': 'unnamed',
        'num_partitions': 5,
        'high_likelihood_correct': (2-tuple) e.g. (0.5, 0.80),
        'low_likelihood_correct': (2-tuple) e.g. (0.20, 0.50),
        'high_likelihood_incorrect': (2-tuple) e.g. (0.5, 0.80),
        'low_likelihood_incorrect': (2-tuple) e.g. (0.20, 0.50),
        'func_mask_params': {},
        'inference_type': MARGINAL,
        'extra_partition_multiplyer': 3,
        'occlude_rand_rate': 0.2,

        'tiny_size': 4 (Optional)
        'uniform_for_incorrect': True (Optional)
        'cases': (list) e.g. [CR-1PO, CR-DW, 1PO-2PO] (Optional, used for novelty detection)
    }

    """

    parser = argparse.ArgumentParser(description='Run instance-SPN test.')
    parser.add_argument('train_kwargs_file', type=str, help="Path to YAML file for training parameters. It must contain key \"templates\".")
    parser.add_argument('test_kwargs_file', type=str, help="Path to YAML file for testing parameters. It must contain key \"to_do\"."\
                        "which defines what tasks to run. Possible to_do specifications are: INF_PLACEHOLDER, INF_RAND, INF_FIXED_RAND,"\
                        "INF_DW, LH, RAND_RATE_TEST, NOISE_LEVEL_TEST, NOVELTY.")
    parser.add_argument('-N', '--num-runs', type=int, help="number of sequences to test on. Default: 1.", default=1)
    parser.add_argument('-n', '--num-rounds', type=int, help="number of inference tasks per sequence. Only used for INF_RAND and INF_FIXED_RAND. Default: 3.", default=3)
    parser.add_argument('-s', '--seed', type=int, help="Seed of randomly generating SPN structure. Default 100", default=100)
    parser.add_argument('-e', '--exp-name', type=str, help="Name to label this experiment. Default: InstanceSpnExperiment", default="InstanceSpnExperiment")
    parser.add_argument('-t', '--test-name', type=str, help="Name of this test. An experiment may have multiple tests. A test name is used to identify tests" \
                        "that should belong to the same parameter setting. Overrides `test_name` in config file.")
    parser.add_argument('-i', '--seq-id', type=str, help="Name of sequence id to run test on. Overrides -n option.")
    parser.add_argument('-d', '--db-abrv', type=str, help="Name of database abbreviation to load testing samples from. Convention is that the first abbreviation" \
                        "is for the testing building. For example: sa = test on Saarbrucken, train on Stockholm and Freiburg; sa_only = test on Saarbrucken, train on Saarbrucken;" \
                        "(special) sa-fr = test on Freiburg, but train on Frebiurg and Stockholm (complement of Saarbrucken). Using this option will override the database settings defined"  \
                        "in the kwargs files.")
    parser.add_argument('-r', '--rate-occluded', type=float, help="Percentage of randomly occluded nodes. Default is 0.2." \
                        "Overrides `occlude_rand_rate` in the test_kwargs file")
    parser.add_argument('-lo-c', '--low-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-c', '--high-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-lo-ic', '--low-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-ic', '--high-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-unif', '--uniform-for-incorrect', action="store_true", help="Set likelihoods for nodes labeled incorrect to be uniform. If supplied, override what is in the test_kwargs file")
    parser.add_argument('-seg', '--segment', action="store_true", help="Segment the graph such that each node is a room, instead of a place. If supplied, override what is in the test_kwargs file")
    parser.add_argument('-p', '--num-partitions', type=int, help="Number of partitions of the graph. If supplied, override what is in the test_kwargs file.")

    args = parser.parse_args()

    train_kwargs_file = args.train_kwargs_file
    test_kwargs_file = args.test_kwargs_file

    train_kwargs = load_kwargs_from_file(train_kwargs_file, eval_keys={'skip_unknown', 'learning_type', 'learning_algorithm'})
    test_kwargs = load_kwargs_from_file(test_kwargs_file, eval_keys={'inference_type',
                                                                     'high_likelihood_correct', 'low_likelihood_correct',
                                                                     'high_likelihood_incorrect', 'low_likelihood_incorrect'})  # cases is only used for Novelty detection.

    # Eval templates
    for i in range(len(train_kwargs['templates'])):
        train_kwargs['templates'][i] = eval(train_kwargs['templates'][i])
    
    # Change to_do to a set. Eval every to do item
    to_do = set({})
    for item in test_kwargs['to_do']:
        to_do.add(eval(item))
    test_kwargs['to_do'] = to_do

    # Create novelty detection cases, if specified
    if 'cases' in test_kwargs:
        for i, case in enumerate(test_kwargs['cases']):
            test_kwargs['cases'][i] = tuple(case.split("-"))  # eval the string version of a tuple.

    if "segment" not in train_kwargs:
        train_kwargs["segment"] = False

    if "mix_ratio" not in train_kwargs:
        train_kwargs["mix_ratio"] = None


    # ...Override parameters from kwargs file...
    # Override -n if seq_id is provided.
    if args.seq_id:
        args.num_runs = 1
    if args.db_abrv:
        db = args.db_abrv
        if db == "st":
            train_kwargs['db_names'] = ['Freiburg', 'Saarbrucken']
            test_kwargs['db_name'] = 'Stockholm'
        elif db == "fr":
            train_kwargs['db_names'] = ['Stockholm', 'Saarbrucken']
            test_kwargs['db_name'] = 'Freiburg'
        elif db == "sa":
            train_kwargs['db_names'] = ['Freiburg', 'Stockholm']
            test_kwargs['db_name'] = 'Saarbrucken'
        elif db == 'fr_only':
            train_kwargs['db_names'] = ['Freiburg']
            test_kwargs['db_name'] = 'Freiburg'
        elif db == 'st_only':
            train_kwargs['db_names'] = ['Stockholm']
            test_kwargs['db_name'] = 'Stockholm'
        elif db == 'sa_only':
            train_kwargs['db_names'] = ['Saarbrucken']
            test_kwargs['db_name'] = 'Saarbrucken'
        # Train on two. Test on one of trained buildings.
        elif db == 'sa-st':
            train_kwargs['db_names'] = ['Freiburg', 'Stockholm']
            test_kwargs['db_name'] = 'Stockholm'
        elif db == 'sa-fr':
            train_kwargs['db_names'] = ['Freiburg', 'Stockholm']
            test_kwargs['db_name'] = 'Freiburg'
        elif db == 'st-sa':
            train_kwargs['db_names'] = ['Freiburg', 'Saarbrucken']
            test_kwargs['db_name'] = 'Saarbrucken'
        elif db == 'st-fr':
            train_kwargs['db_names'] = ['Freiburg', 'Saarbrucken']
            test_kwargs['db_name'] = 'Freiburg'
        elif db == 'fr-st':
            train_kwargs['db_names'] = ['Stockholm', 'Saarbrucken']
            test_kwargs['db_name'] = 'Stockholm'
        elif db == 'fr-sa':
            train_kwargs['db_names'] = ['Stockholm', 'Saarbrucken']
            test_kwargs['db_name'] = 'Saarbrucken'
        else:
            print("HEY! '%s' not recognized as db shorthand." % db)
            exit()

    if args.rate_occluded:
        test_kwargs['occlude_rand_rate'] = args.rate_occluded
    if args.test_name:
        test_kwargs['test_name'] = args.test_name
    if args.low_likelihood_correct:
        test_kwargs['low_likelihood_correct'] = eval(args.low_likelihood_correct)
    if args.high_likelihood_correct:
        test_kwargs['high_likelihood_correct'] = eval(args.high_likelihood_correct)
    if args.low_likelihood_incorrect:
        test_kwargs['low_likelihood_incorrect'] = eval(args.low_likelihood_incorrect)
    if args.high_likelihood_incorrect:
        test_kwargs['high_likelihood_incorrect'] = eval(args.high_likelihood_incorrect)
    if args.uniform_for_incorrect:
        test_kwargs['uniform_for_incorrect'] = True
    if args.segment:
        train_kwargs['segment'] = True
    if args.num_partitions:
        test_kwargs['num_partitions'] = args.num_partitions

    if "high_likelihood_correct" in test_kwargs \
       and "low_likelihood_correct" in test_kwargs \
       and "high_likelihood_incorrect" in test_kwargs \
       and "low_likelihood_incorrect" in test_kwargs:
        test_kwargs['noisification_level'] = get_noisification_level(test_kwargs['high_likelihood_correct'],
                                                                     test_kwargs['low_likelihood_correct'],
                                                                     test_kwargs['high_likelihood_incorrect'],
                                                                     test_kwargs['low_likelihood_incorrect'],
                                                                     uniform_for_incorrect=args.uniform_for_incorrect)
        print("<Noisification Level>")
        pprint(test_kwargs['noisification_level'])


    if train_kwargs['templates'][0].code() == NodeTemplate.code():
        run_node_template_experiment(args.seed, train_kwargs, test_kwargs, to_do,
                                     amount=args.num_runs,
                                     num_rounds=args.num_rounds,
                                     name=args.exp_name, seq_id=args.seq_id)
    elif train_kwargs['templates'][0].code() == EdgeRelationTemplate.code():
        run_edge_relation_template_experiment(args.seed, train_kwargs, test_kwargs, to_do,
                                              amount=args.num_runs,
                                              num_rounds=args.num_rounds,
                                              name=args.exp_name, seq_id=args.seq_id)
    else:
        raise ValueError("Unrecognized template %s" % train_kwargs['templates'][0])

if __name__ == "__main__":
    main()
