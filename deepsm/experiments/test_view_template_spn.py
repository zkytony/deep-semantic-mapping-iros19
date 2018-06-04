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
            def category_abrv(catg_num):
                abrv = CategoryManager.category_map(catg_num, rev=True)
                if abrv == "OC":
                    abrv = -1
                return abrv

            template = kwargs.get('template', None)
            
            model = self._experiment.get_model(template)
            assert model.expanded == False
            
            dw = CategoryManager.category_map('DW')
            cr = CategoryManager.category_map('CR')
            po1 = CategoryManager.category_map('1PO')
            po2 = CategoryManager.category_map('2PO')

            if template == ThreeRelTemplate:
                masked_samples = [
                    [po1, dw, cr, 2],
                    [po1, dw, po2, 2],
                    [po1, -1, cr, 2],
                    [-1, dw, -1, 2],
                    [-1, dw, -1, 3],
                    [-1, dw, -1, 4],
                    [po2, -1, cr, 2],
                    [po2, dw, cr, -1],
                    [cr, dw, po2, -1],
                    [po1, dw, cr, -1],
                    [cr, dw, po1, -1]
                ]

            num_test_samples = 0
            for masked_sample in masked_samples:
                num_test_samples += 1
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
                    'query': [category_abrv(c) for c in masked_sample[:template.num_nodes()]] + masked_sample[template.num_nodes():],
                    'response': [category_abrv(c) for c in filled_sample.tolist()[:template.num_nodes()]] + filled_sample.tolist()[template.num_nodes():],
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
    
    spn_params = TbmExperiment.strip_spn_params(train_kwargs)

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
        model = exp.get_model(test_kwargs['template'])
        
        if test_kwargs['expand']:
            assert not test_kwargs['semantic']
            print("Expand the model.")
            model.expand()   # expand the model
        if test_kwargs['inference_type'] == MPE:
            model.init_mpe_states()  # init MPE ops. (after expanding)
        else:
            print("Not initializing MPE states. Using marginal inference")
            model.init_learning_ops()

        if test_kwargs['semantic']:
            test_kwargs['subname'] = 'semantic_%s_%s' % ("-".join(train_kwargs['train_db']), model.template.__name__)
            exp.test(EdgeRelationTemplateExperiment.TestCase_SemanticUnderstanding, sess, **test_kwargs)
            print_banner("Finish", ch='^')
        else:
            exp.load_testing_data(test_kwargs['test_db'], skip_unknown=CategoryManager.SKIP_UNKNOWN)

            test_kwargs['subname'] = 'random_lh_%s' % model.template.__name__
            report_rnd = exp.test(EdgeRelationTemplateExperiment.TestCase_RandomCompletionTask, sess, **test_kwargs)
            print_banner("Finish", ch='^')


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
        'num_decomps': 2,
        'num_subsets': 4,
        'num_mixtures': 2,
        'num_input_mixtures': 2,

        # spn_learning
        'additive_smoothing': 50,

        'template': ThreeRelTemplate,

        'train_db': ['Stockholm456']
    }

    test_kwargs = {
        'high_likelihood_correct': (0.995, 0.999),
        'low_likelihood_correct': (0.001, 0.005),
        'high_likelihood_incorrect': (0.995, 0.999),
        'low_likelihood_incorrect': (0.001, 0.005),

        'template': ThreeRelTemplate,
        'limit': -1,
        'num_partitions': 10,
        'inference_type': MARGINAL,
        'test_db': 'Stockholm7',
        'expand': False,
        'semantic': True
    }
    
    run_edge_relation_template_experiment(train_kwargs, test_kwargs, seed=seed)
