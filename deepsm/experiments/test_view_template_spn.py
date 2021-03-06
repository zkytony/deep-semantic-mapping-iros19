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
import itertools

import tensorflow as tf
import libspn as spn

from deepsm.graphspn.spn_model import SpnModel
from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.graphspn.tbm.template import EdgeRelationTemplate, ThreeRelTemplate, SingleTemplate, SingleRelTemplate, RelTemplate
from deepsm.graphspn.tbm.spn_template import TemplateSpn, EdgeRelationTemplateSpn
from deepsm.graphspn.tests.tbm.runner import TbmExperiment
from deepsm.graphspn.tests.runner import TestCase
from deepsm.util import CategoryManager, print_banner, print_in_box

from deepsm.experiments.common import COLD_ROOT, GRAPHSPN_RESULTS_ROOT, TOPO_MAP_DB_ROOT

MPE = 'mpe'
MARGINAL = 'marginal'


RANDOM = 1
SEMANTIC = 2
ALL_PROB = 3


def category_abrv(catg_num):
    abrv = CategoryManager.category_map(catg_num, rev=True)
    if abrv == "OC":
        abrv = -1
    return abrv

class EdgeRelationTemplateExperiment(TbmExperiment):

    def __init__(self, db_root, *spns, **kwargs):
        """
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.
        """
        super().__init__(db_root, *spns, **kwargs)

        
    ##############################################
    #                                            #
    #          All Probilities                   #
    #                                            #
    ##############################################
    class TestCase_AllProbabilities(TestCase):
        def __init__(self, experiment):
            super().__init__(experiment)

            self._test_cases = []

        def run(self, sess, **kwargs):
            """
            Enumerate all value combinations, and produce likelihood for each
            """
            template = kwargs.get('template', None)
            
            model = self._experiment.get_model(template)
            assert model.expanded == False

            num_view_dists = model._divisions // 2

            all_cases = set()

            # First get all combinations of classes with repeats
            catg_combs = set(itertools.product(CategoryManager.known_categories(), repeat=template.num_nodes()))

            # Then, enumerate all possible view distances
            for comb in catg_combs:
                if template.num_edge_pair() > 0:
                    for i in range(num_view_dists):
                        all_cases.add(tuple(list(comb) + [i]))
                else:
                    all_cases.add(tuple(comb))

            _i = 0
            for case in all_cases:
                catg_nums = list(map(CategoryManager.category_map, case[:template.num_nodes()]))
                lh = float(model.evaluate(sess, catg_nums + list(case[template.num_nodes():]))[0])
                
                self._test_cases.append({
                    'case': case[:template.num_nodes()] + (case[-1]+1,),
                    'likelihood': lh
                })
                _i += 1
                sys.stdout.write("Testing [%d/%d]\r" % (_i, len(all_cases)))
                sys.stdout.flush()
            sys.stdout.write("\n")

            
        def _report(self):
            return self._test_cases

        
        def save_results(self, save_path):
            """
            save_path (str): path to saved results (Required).

            Also return the report.
            """
            # Save all test cases into a json file.
            with open(os.path.join(save_path, "test_cases_lh_order.csv"), "w") as f:
                writer = csv.writer(f, delimiter=',', quotechar='"')
                for tc in sorted(self._test_cases, key=lambda x: x['likelihood'], reverse=True):
                    writer.writerow(list(tc['case']) + [tc['likelihood']])
                    
            with open(os.path.join(save_path, "test_cases_cl_order.csv"), "w") as f:
                writer = csv.writer(f, delimiter=',', quotechar='"')
                for tc in sorted(self._test_cases, key=lambda x: x['case'], reverse=True):
                    writer.writerow(list(tc['case']) + [tc['likelihood']])

            print("Everything saved in %s" % save_path)

            return self._report()
    
        
    ##############################################
    #                                            #
    #          Semantic Understanding            #
    #                                            #
    ##############################################
    class TestCase_SemanticUnderstanding(TestCase):
        def __init__(self, experiment):
            super().__init__(experiment)
            self._results = {k:{'correct': 0, 'wrong': 0} for k in range(CategoryManager.NUM_CATEGORIES)}
            self._test_cases = []

        def run(self, sess, **kwargs):
            """
            Tests the model's understanding of semantic relations, by inferring
            semantic attributes.
            """

            template = kwargs.get('template', None)
            
            model = self._experiment.get_model(template)
            assert model.expanded == False
            
            dw = CategoryManager.category_map('DW')
            cr = CategoryManager.category_map('CR')
            po1 = CategoryManager.category_map('1PO')
            po2 = CategoryManager.category_map('2PO')
            if CategoryManager.NUM_CATEGORIES > 4:
                lo = CategoryManager.category_map('LO')
                lab = CategoryManager.category_map('LAB')
                ba = CategoryManager.category_map('BA')
                kt = CategoryManager.category_map('KT')
                mr = CategoryManager.category_map('MR')
                ut = CategoryManager.category_map('UT')

            if template == ThreeRelTemplate:
                masked_samples = [
                    # [cr, cr, cr, -1],
                    # [lab, lab, lab, -1],
                    # [-1, -1, -1, 3],
                    [po1, dw, cr, 1],
                    [po1, dw, cr, 2],
                    [po1, dw, cr, 3],
                    [po1, dw, cr, 4],
                    [po1, dw, po2, 2],
                    [po1, -1, cr, 2],
                    [-1, dw, -1, 1],
                    [-1, dw, -1, 2],
                    [-1, dw, -1, 3],
                    [-1, dw, -1, 4],
                    [-1, -1, -1, -1],
                    [cr, -1, cr, 1],
                    [cr, -1, cr, 2],
                    [cr, -1, cr, 3],
                    [po1, dw, cr, -1],
                ]
                if CategoryManager.NUM_CATEGORIES > 4:
                    masked_samples.extend([[lab, -1, lab, -1],
                                           [lab, -1, lab, 2],
                                           [lab, -1, lab, 3],
                                           [lab, -1, lab, 4]])


            num_test_samples = 0
            for masked_sample in masked_samples:
                num_test_samples += 1
                # If the view distance value != -1, subtract 1, because vdist ranges from 1-4 but we want 0-3 as input to the network
                if masked_sample[-1] != -1:
                    masked_sample[-1] -= 1
                
                if kwargs.get("inference_type") == MARGINAL:
                    marginals_nodes, marginals_view = model.marginal_inference(sess,
                                                                               np.array(masked_sample, dtype=int),
                                                                               masked_only=True)
                    filled_sample = list(masked_sample)
                    for i in range(len(masked_sample)):
                        if masked_sample[i] == -1:
                            if i < template.num_nodes():
                                mgnl = list(marginals_nodes[i])
                                filled_sample[i] = mgnl.index(max(mgnl))
                            else:
                                mgnl = list(marginals_view)
                                filled_sample[i] = mgnl.index(max(mgnl))
                    filled_sample = np.array(filled_sample, dtype=int)
                lh = float(model.evaluate(sess, filled_sample)[0])

                self._test_cases.append({
                    'query': [category_abrv(c) for c in masked_sample[:template.num_nodes()]] + [(masked_sample[-1]+1)],
                    'response': [category_abrv(c) for c in filled_sample.tolist()[:template.num_nodes()]] + [filled_sample.tolist()[-1]+1],
                    'likelihood': lh
                })
                sys.stdout.write("Testing [%d/%d]\r" % (num_test_samples, len(masked_samples)))
                sys.stdout.flush()
            sys.stdout.write("\n")

        def _report(self):
            return self._test_cases

        
        def save_results(self, save_path):
            """
            save_path (str): path to saved results (Required).

            Also return the report.
            """
            
            # report
            report = self._report()

            with open(os.path.join(save_path, "report.json"), "w") as f:
                json.dump(report, f, indent=4, sort_keys=True)

            # Save all test cases into a json file.
            with open(os.path.join(save_path, "test_cases.json"), "w") as f:
                json.dump(self._test_cases, f, indent=4, sort_keys=True)

            print("Everything saved in %s" % save_path)

            return report

            
    ##############################################
    #                                            #
    #          Random Completion                 #
    #                                            #
    ##############################################
    class TestCase_RandomCompletionTask(TestCase):

        def __init__(self, experiment):
            super().__init__(experiment)
            self._results = {k:{'correct': 0, 'wrong': 0} for k in range(CategoryManager.NUM_CATEGORIES)}
            self._test_cases = []

        def run(self, sess, func_mask_edge_relation_template=None, **kwargs):
            """
            kwargs:
                template
                num_partitions (int) optional
                limit (int) optional
            """
            num_partitions = kwargs.get("num_partitions", 10)
            limit = kwargs.get("limit", -1)
            template = kwargs.get("template", None)
            high_likelihood_correct = kwargs.get("high_likelihood_correct")
            low_likelihood_correct = kwargs.get("low_likelihood_correct")
            high_likelihood_incorrect = kwargs.get("high_likelihood_incorrect")
            low_likelihood_incorrect = kwargs.get("low_likelihood_incorrect")
            
            model = self._experiment.get_model(template)
            test_samples = self._experiment.dataset.create_edge_relation_template_dataset(model.template,
                                                                                          db_names=self._experiment.test_db,
                                                                                          seqs_limit=limit,
                                                                                          num_partitions=num_partitions)
            for db in test_samples:
                num_test_samples = 0

                for test_sample in test_samples[db]:  # test_sample is groundtruth.
                    masked_sample, metadata = TbmExperiment.mask_edge_relation_template_sample(test_sample, model, num_nodes_masked=1)
                    if model.expanded:
                        likelihoods = TbmExperiment.create_likelihoods_vector(test_sample[:model.template.num_nodes()], model,
                                                                              high_likelihood_correct, low_likelihood_correct,
                                                                              high_likelihood_incorrect, low_likelihood_incorrect,
                                                                              masked_sample=masked_sample)
                    num_test_samples += 1
                    if kwargs.get('inference_type') == MARGINAL:
                        if model.expanded:
                            marginals_nodes, _ = model.marginal_inference(sess,
                                                                          np.array(masked_sample, dtype=int),
                                                                          query_lh=np.array(likelihoods, dtype=float32), masked_only=True)
                        else:
                            marginals_nodes, _ = model.marginal_inference(sess,
                                                                          np.array(masked_sample, dtype=int),  masked_only=True)

                        filled_sample = list(masked_sample)
                        for i in range(len(masked_sample)):
                            if masked_sample[i] == -1:
                                mgnl = list(marginals_nodes[i])
                                filled_sample[i] = mgnl.index(max(mgnl))
                        filled_sample = np.array(filled_sample, dtype=int)
                            
                    # assert that everything besides the masked variables are the same
                    masked_sample_copy = np.copy(masked_sample)
                    masked_sample_copy[masked_sample_copy == filled_sample] = 0
                    assert np.unique(masked_sample_copy)[0] == -1

                    self._record_results(test_sample, masked_sample, filled_sample.tolist(), metadata=metadata)
                    # Print status
                    sys.stdout.write("Testing %s [%d]\r" % (db, num_test_samples))
                    sys.stdout.flush()
                self._experiment.data_count({db:num_test_samples}, update=True)
            sys.stdout.write("\n")


        def _record_results(self, test_sample, masked_sample, filled_sample, metadata={}):
            """
            test_sample (list): groundtruth test sample
            masked_sample (list): query test sample
            filled_sample (list): inference result
            """
            results = {k:{'correct':0,'wrong':0} for k in test_sample}
            for i in range(len(masked_sample)):
                if masked_sample[i] == -1:
                    if filled_sample[i] == test_sample[i]:
                        results[test_sample[i]]['correct'] += 1
                    else:
                        results[test_sample[i]]['wrong'] += 1
            # update global results
            for k in self._results:
                if k in results:
                    self._results[k]['correct'] += results[k]['correct']
                    self._results[k]['wrong'] += results[k]['wrong']
            # record test case
            self._test_cases.append({
                "groundtruth": test_sample,
                "query": masked_sample,
                "filled": filled_sample,
                "results": results
            })

        def _report(self):
            """
            Reports some custom results right after the test case finishes.
            """

            # Simply report:
            # - What's the training data? (database, number of samples)
            # - What's the testing data? (database, number of samples)
            # - Accuracy on different categories.
            
            report = {
                "data": {
                    'count':self._experiment.data_count(),
                    'train': self._experiment._train_db,
                    'test': self._experiment._test_db
                },
                'results': {}
            }
            for k in self._results:
                kabrv = CategoryManager.category_map(k, rev=True)
                report['results'][kabrv] = [
                    self._results[k]['correct'],
                    self._results[k]['correct']+self._results[k]['wrong'],
                ]
                if (self._results[k]['correct']+self._results[k]['wrong']) != 0:
                    report['results'][kabrv].append(report['results'][kabrv][0] / report['results'][kabrv][1] )
                else:
                    report['results'][kabrv].append("untested")
            
            count_correct, total = 0, 0
            for k in self._results:
                if type(report['results'][CategoryManager.category_map(k, rev=True)]) is not str:
                      count_correct += self._results[k]['correct']
                      total += self._results[k]['correct'] + self._results[k]['wrong']
            report["results"]["_overall_"] =  count_correct / total
            return report


        def save_results(self, save_path):
            """
            save_path (str): path to saved results (Required).

            Also return the report.
            """
            
            # report
            report = self._report()

            with open(os.path.join(save_path, "report.json"), "w") as f:
                json.dump(report, f, indent=4, sort_keys=True)

            # Save all test cases into a json file.
            with open(os.path.join(save_path, "test_cases.json"), "w") as f:
                json.dump(self._test_cases, f, indent=4, sort_keys=True)

            print("Everything saved in %s" % save_path)

            return report


                        

def run_edge_relation_template_experiment(train_kwargs, test_kwargs, seed=None):
    # EdgeRelation template experiments

    timestamp = time.strftime("%Y%m%d-%H%M")
    train_kwargs['timestamp'] = timestamp
    test_kwargs['timestamp'] = timestamp
    
    print_in_box(["NodeTemplate experiments"], ho="+", vr="+")

    spn_params = TbmExperiment.strip_spn_params(train_kwargs, train_kwargs['learning_algorithm'])

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
        
        model = exp.get_model(test_kwargs['template'])
        model.print_params()
        
        train_info = exp.train_models(sess, **train_kwargs)

        if test_kwargs['expand']:
            print("Expand the model.")
            model.expand()   # expand the model
        if test_kwargs['inference_type'] == MPE:
            model.init_mpe_states()  # init MPE ops. (after expanding)
        else:
            print("Not initializing MPE states. Using marginal inference")
            model.init_learning_ops()
        
        if test_kwargs['to_do'] == SEMANTIC:
            test_kwargs['subname'] = 'semantic_%s_%s' % ("-".join(train_kwargs['train_db']), model.template.__name__)
            exp.test(EdgeRelationTemplateExperiment.TestCase_SemanticUnderstanding, sess, **test_kwargs)
        elif test_kwargs['to_do'] == ALL_PROB:
            test_kwargs['subname'] = 'allprob_%s_%s' % ("-".join(train_kwargs['train_db']), model.template.__name__)
            exp.test(EdgeRelationTemplateExperiment.TestCase_AllProbabilities, sess, **test_kwargs)
        elif test_kwargs['to_do'] == RANDOM:
            exp.load_testing_data(test_kwargs['test_db'], skip_unknown=CategoryManager.SKIP_UNKNOWN)

            test_kwargs['subname'] = 'random_lh_%s' % model.template.__name__
            report_rnd = exp.test(EdgeRelationTemplateExperiment.TestCase_RandomCompletionTask, sess, **test_kwargs)
        else:
            raise ValueError("Unrecognized test case %s" % test_kwargs['to_do'])

        print_banner("Finish", ch='^')


if __name__ == "__main__":
    
    CategoryManager.TYPE = "SIMPLE"
    CategoryManager.init()
    
    seed = random.randint(200,1000)

    template = ThreeRelTemplate

    # Config
    train_kwargs = {
        'num_partitions': 10,#10,
        'num_batches': 10,
        'num_epochs': None,
        'save': True,
        'load_if_exists': False,
        'likelihood_thres': 0.1,
        'save_training_info': True,

        # spn_learning
        'learning_algorithm': spn.EMLearning,
        'additive_smoothing': 30,
        
        # 'learning_algorithm': spn.GDLearning,
        # 'learning_type': spn.LearningType.GENERATIVE,
        # 'learning_rate': 0.001,

        # spn_structure
        'num_decomps': 1,
        'num_subsets': 4,
        'num_mixtures': 2,
        'num_input_mixtures': 2,

        'template': template,

        'train_db': ['Freiburg']
    }

    test_kwargs = {
        'high_likelihood_correct': (0.995, 0.999),
        'low_likelihood_correct': (0.001, 0.005),
        'high_likelihood_incorrect': (0.995, 0.999),
        'low_likelihood_incorrect': (0.001, 0.005),

        'template': template,
        'limit': -1,
        'num_partitions': 1,
        'inference_type': MARGINAL,
        'test_db': 'Stockholm',
        'expand': False,
        'to_do': ALL_PROB
    }

    # if test_kwargs['to_do'] == RANDOM:
    #     test_kwargs['semantic'] = True
    
    run_edge_relation_template_experiment(train_kwargs, test_kwargs, seed=seed)
