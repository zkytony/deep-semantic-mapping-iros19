#!/usr/bin/env python
#
# Big Question: “Does introducing view variables have any positive effect
#                on the result?”
#
# Test A: Test within the template SPN
#   Training set = topo maps from two databases
#   Testing set = topo maps from the other database
#
#   During testing:
#     1. For a template sample, mask a random one node’s label. Lower the
#        likelihood of that node’s groundtruth class. Test to see if the model
#        infers the groundtruth class for that node.
#     2. Count the accuracy for different classes
#        (Pay special attention to the case where doorway is masked and at the
#         center (should give reasonable result), whereas if the doorway is on the
#         side, there shouldn’t be good inference result.
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

from spn_topo.spn_model import SpnModel
from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, PairTemplate, SingletonTemplate, \
    ThreeNodeTemplate, PairEdgeTemplate, SingleEdgeTemplate, StarTemplate
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn
from spn_topo.tests.tbm.runner import TbmExperiment
from spn_topo.tests.runner import TestCase
from spn_topo.util import CategoryManager, print_banner, print_in_box

from spn_topo.tests.constants import COLD_ROOT, RESULTS_DIR, TOPO_MAP_DB_ROOT


MPE = 'mpe'
MARGINAL = 'marginal'


class TemplateSpnExperiment(TbmExperiment):

    """
    As described above.
    """

    def __init__(self, db_root, *spns, **kwargs):
        """
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.
        """
        super().__init__(db_root, *spns, **kwargs)

    ########## Below are methods used in test cases ##########
    @classmethod
    def mask_node_template_sample(cls, sample, num_nodes_masked=1):
        """
        Returns: a tuple of two elements:
           masked_sample (list):  a new array of same shape as sample. 
           count (dict): a dictionary mapping from category number to number of
                         times it is masked

        sample (list): contains semantic variable values, in order.
        num_nodes_masked (int): number of different nodes to be masked.
                                For any number bigger than len(sample),
                                will return an array with -1's.
        """
        num_nodes_masked = min(num_nodes_masked, len(sample))
        indices = np.random.choice(np.arange(len(sample)), num_nodes_masked).tolist()
        masked_sample = list(sample)
        count = {}
        for i in indices:
            masked_sample[i] = -1
            count[sample[i]] = count.get(i, 0) + 1
        return masked_sample, count
            

    @classmethod
    def mask_edge_template_sample(cls, sample, model, num_nodes_masked=1):
        """
        Returns: a tuple of two elements:
           masked_sample (list):  a new array of same shape as sample. 
           count (dict): a dictionary mapping from category number to number of
                         times it is masked

        sample (list): contains semantic variable values, in order.
        model (EdgeTemplateSpn): the spn model that `sample` suits for.
        num_nodes_masked (int): number of different nodes to be masked.
                                For any number bigger than model.template.num_nodes,
                                will return an array with -1's.
        """
        if not isinstance(model, EdgeTemplateSpn):
            raise ValueError("The given model is not EdgeTemplateSpn!")
        masked_sample = list(sample)
        count = {}
        
        num_nodes_masked = min(num_nodes_masked, model.template.num_nodes())
        mask_indices = np.random.choice(np.arange(model.template.num_nodes()), num_nodes_masked).tolist()
        node_idx = 0  # node index. Not sample index
        for indices in model.template.semantic_sample_node_iterator(sample):
            if node_idx in mask_indices:
                catg_num = None
                for j in indices:
                    masked_sample[j] = -1
                    catg_num = sample[j]
                    
                count[catg_num] = count.get(catg_num, 0) + 1
            node_idx += 1
        return masked_sample, count


    @classmethod
    def mask_node_template_doorway(cls, sample):
        """
        Masks ONE doorway node if there is any. We ASSUME that the given sample is for
        the ThreeNodeTemplate, thus it has 3 elements.

        Returns: a tuple of two elements:
           masked_sample (list):  a new array of same shape as sample.
                                  None if the given sample does not have a variable that is a doorway.
           center (bool): True if we masked the center. False otherwise


        sample (list): contains semantic variable values, in order.
        """
        assert len(sample) == 3
        
        masked_sample = list(sample)
        
        for i in range(3):
            if sample[i] == CategoryManager.category_map("DW"):
                masked_sample[i] = -1
                if i == 0 or i == 2:
                    return masked_sample, False
                else:
                    return masked_sample, True
        return None, False
                    

    @classmethod
    def mask_edge_template_doorway(cls, sample, model):
        """
        Masks ONE doorway node if there is any. We ASSUME that the given sample is for
        the PairEdgeTemplate, thus it has 4 elements. [a,b,b,c]

        Returns: a tuple of two elements:
           masked_sample (list):  a new array of same shape as sample.
                                  None if the given sample does not have a variable that is a doorway.
           center (bool): True if we masked the center. False otherwise


        sample (list): contains semantic variable values, in order.
        """
        if model.template != PairEdgeTemplate:
            raise ValueError("Requires the sample to be for PairEdgeTemplate.")

        masked_sample = list(sample)
        
        for i in range(4):
            if sample[i] == CategoryManager.category_map("DW"):
                masked_sample[i] = -1
                if i == 1:
                    assert sample[i] == sample[i+1]
                    masked_sample[i+1] = -1
                    return masked_sample, True
                else:
                    return masked_sample, False
        return None, False


    ####################TestCase#######################
    class MyTestCase(TestCase):
        
        def __init__(self, experiment):
            super().__init__(experiment)
            self._test_cases = []


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



    ##################################################
    #                                                #
    #     TestCase_CompletionTask                    #
    #                                                #
    ##################################################
    class TestCase_CompletionTask(MyTestCase):

        def __init__(self, experiment):
            super().__init__(experiment)

            
        def run(self, sess, func_mask_node_template=None, func_mask_edge_template=None, **kwargs):
            """
            Runs this test case. For each sample in the test dataset, produce a random
            likelihood array of shape (n,m)* for all semantics variables. Then, create a
            query with semantics values and randomly mask one of them to -1. Lower the
            likelihood of the corresponding class in the likelihood array, and raise others.
            Test to see if the model infers the groundtruth class for that node.

            Count the accuracy for different classes.

            *: n is the number of semantics variables. m is the number of semantic categories.
            
            sess (tf.Session): tensorflow session.
            template (Template): template used to get model

            **kwargs:
               num_partitions (int): number of partition attempts. See TopoMapDataset for details.
                                     Default 10.
               high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
               low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                                       and the other (untrue) classes.
               limit (int): Maximum number of test samples (sequences) to be used for testing. Default -1,
                            meaning all.
               template (Template): The template to do test on
            """
            print("Running test case %s" % self.__class__.__name__)
            
            num_partitions = kwargs.get("num_partitions", 10)  # deprecated. useless.
            high_likelihood_correct = kwargs.get("high_likelihood_correct")
            low_likelihood_correct = kwargs.get("low_likelihood_correct")
            high_likelihood_incorrect = kwargs.get("high_likelihood_incorrect")
            low_likelihood_incorrect = kwargs.get("low_likelihood_incorrect")
            limit = kwargs.get("limit", -1)
            template = kwargs.get("template", None)

            model = self._experiment.get_model(template)
            
            ## NodeTemplate
            if self._experiment.template_mode == 0:
                test_samples = self._experiment.dataset.load_template_dataset(model.template,
                                                                              db_names=self._experiment.test_db,
                                                                              seqs_limit=limit)
            ## EdgeTemplate
            # else:
            #     test_samples = self._experiment.dataset.create_edge_template_dataset(model.template,
            #                                                                          db_names=self._experiment.test_db,
            #                                                                          seqs_limit=limit,
            #                                                                          num_partitions=num_partitions)

            for db in test_samples:
                num_test_samples = 0

                for test_sample in test_samples[db]:  # test_sample is groundtruth.
                    if self._experiment.template_mode == NodeTemplate.code():  ## NodeTemplate
                        masked_sample, metadata = func_mask_node_template(test_sample)
                        # If masked_sample is None, we would skip this test_sample.
                        if masked_sample is None:
                            continue
                        if model.expanded:
                            likelihoods = TbmExperiment.create_likelihoods_vector(test_sample, model,
                                                                                  high_likelihood_correct, low_likelihood_correct,
                                                                                  high_likelihood_incorrect, low_likelihood_incorrect,
                                                                                  masked_sample=masked_sample)

                    # else:  ## EdgeTemplate
                    #     masked_sample, metadata = func_mask_edge_template(test_sample[:model.num_nodes], model)
                    #     # If masked_sample is None, we would skip this test_sample.
                    #     if masked_sample is None:
                    #         continue
                    #     likelihoods = TbmExperiment.create_likelihoods_vector(test_sample[:model.num_nodes], model, high_likelihood,
                    #                                                           low_likelihood, masked_sample=masked_sample)
                    #     masked_sample += test_sample[model.num_nodes:]

                    num_test_samples += 1
                        
                    #  MPE inference. We expect the result to have shape (n,)
                    if kwargs.get('inference_type') == MPE:
                        if model.expanded:
                            filled_sample = np.array(list(model.mpe_inference(sess,
                                                                              np.array(masked_sample, dtype=int),
                                                                              np.array(likelihoods, dtype=float32))),
                                                 dtype=int).flatten()
                        else:
                            filled_sample = np.array(list(model.mpe_inference(sess,
                                                                              np.array(masked_sample, dtype=int))),
                                                 dtype=int).flatten()
                    elif kwargs.get('inference_type') == MARGINAL:
                        if model.expanded:
                            marginals = model.marginal_inference(sess,
                                                                 np.array(masked_sample, dtype=int),
                                                                 query_lh=np.array(likelihoods, dtype=float32), masked_only=True)
                        else:
                            marginals = model.marginal_inference(sess,
                                                                 np.array(masked_sample, dtype=int), masked_only=True)
                        filled_sample = list(masked_sample)
                        for i in range(len(masked_sample)):
                            if masked_sample[i] == -1:
                                mgnl = list(marginals[i])
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


        @abstractmethod
        def _report(self):
            """
            Reports some custom results right after the test case finishes.
            """
            pass

        @abstractmethod
        def _record_results(self, test_sample, masked_sample, filled_sample, metadata={}):
            """
            test_sample (list): groundtruth test sample
            masked_sample (list): query test sample
            filled_sample (list): inference result
            
            metadata: metadata for the results (such as count for each category's correctness)
            """
            pass


    ##################################################
    #                                                #
    #     TestCase_RandomCompletionTask              #
    #                                                #
    ##################################################
    class TestCase_RandomCompletionTask(TestCase_CompletionTask):

        def __init__(self, experiment):
            super().__init__(experiment)

            self._results = {k:{'correct': 0, 'wrong': 0} for k in range(CategoryManager.NUM_CATEGORIES)}


        def run(self, sess, **kwargs):
            """
            Count the accuracy for different classes.

            sess (tf.Session): tensorflow session.

            **kwargs:
               num_partitions (int): number of partition attempts. See TopoMapDataset for details.
                                     Default 10.
               high_likelihood_(in)correct (float): the min likelihood of the semantics variable's true class.
               low_likelihood_(in)correct (float): the max likelihood of the masked semantics variable's true class.
                                       and the other (untrue) classes.
               limit (int): Maximum number of test samples (sequences) to be used for testing. Default -1,
                            meaning all.
               template (Template): template to do test on
            """
            super(TemplateSpnExperiment.TestCase_RandomCompletionTask, self).run(sess,
                                                                                 func_mask_node_template=TemplateSpnExperiment.mask_node_template_sample,
                                                                                 func_mask_edge_template=TemplateSpnExperiment.mask_edge_template_sample,
                                                                                 **kwargs)

        def _record_results(self, test_sample, masked_sample, filled_sample, metadata={}):
            """
            test_sample (list): groundtruth test sample
            masked_sample (list): query test sample
            filled_sample (list): inference result
            """
            results = {k:{'correct':0, 'wrong':0} for k in test_sample}
            for i in range(len(masked_sample)):
                if masked_sample[i] == -1:
                    # Check if we get it right. Record the result
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


                


    ##################################################
    #                                                #
    #    TestCase_DoorwayCompletionTask              #
    #                                                #
    ##################################################
    class TestCase_DoorwayCompletionTask(TestCase_CompletionTask):

        def __init__(self, experiment):
            super().__init__(experiment)
            self._results = {'center': {'correct': 0, 'wrong': 0},
                             'side': {'correct': 0, 'wrong': 0}}


        def run(self, sess, *args, **kwargs):
            """
            Same as above, except in this test case, we only mask doorway variables, and we
            keep track of the number of cases that the doorway is at the side versus. at the
            center.

            **kwargs:
               num_partitions (int): number of partition attempts. See TopoMapDataset for details.
                                     Default 10.
               high_likelihood_(in)correct (float): the min likelihood of the semantics variable's true class.
               low_likelihood_(in)correct (float): the max likelihood of the masked semantics variable's true class.
                                       and the other (untrue) classes.
               limit (int): Maximum number of test samples (sequences) to be used for testing.
            """
            kwargs['template'] = self._experiment.main_model.template
            super(TemplateSpnExperiment.TestCase_DoorwayCompletionTask, self).run(sess,
                                                                                             func_mask_node_template=TemplateSpnExperiment.mask_node_template_doorway,
                                                                                             func_mask_edge_template=TemplateSpnExperiment.mask_edge_template_doorway,
                                                                                             **kwargs)


        def _record_results(self, test_sample, masked_sample, filled_sample, metadata={}):
            """
            metadata is a bool that indicates if the doorway is masked at the center.
            """
            center = bool(metadata)
            results = {'center':{'correct':0, 'wrong':0},
                       'side': {'correct':0, 'wrong':0}}

            for i in range(len(masked_sample)):
                if masked_sample[i] == -1:
                    # Check if we get it right. Record the result
                    if filled_sample[i] == test_sample[i]:
                        if center:
                            results['center']['correct'] += 1
                        else:
                            results['side']['correct'] += 1
                    else:
                        if center:
                            results['center']['wrong'] += 1
                        else:
                            results['side']['wrong'] += 1

            # update global results
            for k in self._results:
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
            Custom report
            """
            report = {
                "data": {
                    'count':self._experiment.data_count(),
                    'train': self._experiment._train_db,
                    'test': self._experiment._test_db
                },
                'results': {}
            }

            for k in self._results:
                report['results'][k] = [                    
                    self._results[k]['correct'],
                    self._results[k]['correct']+self._results[k]['wrong'],
                ]
                if (self._results[k]['correct']+self._results[k]['wrong']) != 0:
                    report['results'][k].append(report['results'][k][0] / report['results'][k][1] )
                else:
                    report['results'][k].append("untested")
            return report


    ##################################################
    #                                                #
    #     TestCase_SemanticUnderstanding             #
    #                                                #
    ##################################################
    class TestCase_SemanticUnderstanding(MyTestCase):

        def __init__(self, experiment):
            super().__init__(experiment)


        def run(self, sess, **kwargs):
            """
            Tests the model's understanding of semantic relations, by inferring
            semantic attributes in the following test cases:
            
            #1: [-1, 'DW', -1]
            #2: ['CR', 'DW', -1]
            #3: [-1, 'DW', 'CR']
            #4: ['1PO', 'DW', -1]
            #5: [-1, 'DW', '1PO']
            #6: ['CR', -1, '1PO']
            #7: ['CR', -1, '2PO']
            #8: ['CR', -1, 'UN']
            """
            def category_abrv(catg_num):
                abrv = CategoryManager.category_map(catg_num, rev=True)
                if abrv == "OC":
                    abrv = -1
                return abrv

            template = kwargs.get('template', None)
            
            dw = CategoryManager.category_map('DW')
            cr = CategoryManager.category_map('CR')
            po1 = CategoryManager.category_map('1PO')
            po2 = CategoryManager.category_map('2PO')
            mr = CategoryManager.category_map('MR')
            lab = CategoryManager.category_map('LAB')
            ut = CategoryManager.category_map('UT')
            
            model = self._experiment.get_model(template)
            assert model.expanded == False

            if template == PairEdgeTemplate or template == ThreeNodeTemplate:
                masked_samples = [
                    [-1, dw, -1],
                    [cr, dw, -1],
                    [-1, dw, cr],
                    [po1, dw, -1],
                    [-1, dw, po1],
                    [cr, -1, po1],
                    [cr, -1, po2],
                    [cr, cr, cr],
                    [-1, po1, -1],
                    [po1, cr, -1],
                    [lab, lab, -1],
                    [po1, -1, lab],
                    [-1, -1, cr],
                    [-1, cr, cr],
                    [dw, -1, cr]
                 ]
            elif template == PairTemplate:
                masked_samples = [
                    [-1, dw],
                    [dw, -1],
                    [-1, -1],
                    [cr, -1],
                    [po1, -1],
                    [po1, cr],
                    [cr, cr],
                    [po1, po1],
                    [po2, po2]
                ]
            elif template == SingletonTemplate:
                masked_samples = [
                    [-1],
                    [cr],
                    [po1],
                    [po2],
                    [dw]
                ]
            elif template == StarTemplate:
                masked_samples = [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, dw, -1, -1],
                    [cr, -1, dw, cr, -1],
                    [po1, -1, dw, po1, -1],
                    [po2, cr, -1, po2, cr],
                    [cr, cr, -1, lab, lab],
                    [cr, cr, cr, lab, lab],
                    [po2, po2, -1, cr, cr],
                ]

            for masked_sample in masked_samples:
                if self._experiment.template_mode == EdgeTemplate.code():  ## EdgeTemplate
                    masked_sample = [masked_sample[i] for i in [0, 1, 1, 2]] + [4, 0, 1, 5]

                if kwargs.get("inference_type") == MPE:
                    filled_sample = np.array(list(model.mpe_inference(sess,
                                                                      np.array(masked_sample, dtype=int))),
                                             dtype=int).flatten()

                elif kwargs.get("inference_type") == MARGINAL:
                    marginals = model.marginal_inference(sess,
                                                         np.array(masked_sample, dtype=int),
                                                         masked_only=True)
                    filled_sample = list(masked_sample)
                    for i in range(len(masked_sample)):
                        if masked_sample[i] == -1:
                            mgnl = list(marginals[i])
                            filled_sample[i] = mgnl.index(max(mgnl))
                    filled_sample = np.array(filled_sample, dtype=int)
                lh = float(model.evaluate(sess, filled_sample)[0])

                # assert that everything besides the masked variables are the same
                masked_sample_copy = np.copy(masked_sample)
                masked_sample_copy[masked_sample_copy == filled_sample] = 0
                #assert np.unique(masked_sample_copy)[0] == -1

                self._test_cases.append({
                    'query': [category_abrv(c) for c in masked_sample[:model.num_nodes]] + masked_sample[model.num_nodes:],
                    'response': [category_abrv(c) for c in filled_sample.tolist()[:model.num_nodes]] + filled_sample.tolist()[model.num_nodes:],
                    'likelihood': lh
                })

        def _report(self):
            return self._test_cases
            

    ##################################################
    #                                                #
    #     TestCase_GeometricUnderstanding            #
    #                                                #
    ##################################################
    class TestCase_GeometricUnderstanding(MyTestCase):

        def __init__(self, experiment):
            super().__init__(experiment)

            assert self._experiment.template_mode == 1


        def run(self, sess, *args, **kwargs):
            """
            Tests the model's understanding of geometric relations, by inferring
            view numbers.
            
            #1: ['CR', 'DW', '1PO']
            #2: ['CR', 'DW', '2PO']
            #3: ['CR', 'DW', 'UN']
            #4: ['CR', 'DW', 'CR']
            #5: ['1PO', '1PO', '1PO']
            #6: ['CR', 'CR', 'CR']
            """
            def category_abrv(catg_num):
                abrv = CategoryManager.category_map(catg_num, rev=True)
                if abrv == "OC":
                    abrv = -1
                return abrv
            
            dw = CategoryManager.category_map('DW')
            cr = CategoryManager.category_map('CR')
            po1 = CategoryManager.category_map('1PO')
            po2 = CategoryManager.category_map('2PO')
            un = CategoryManager.category_map('UN')
            
            model = self._experiment.main_model
            assert model.expanded == False

            test_samples = [
                [cr, dw, dw, cr,  -1, -1, -1, -1],
                [cr, dw, dw, po2, -1, -1, -1, -1],
                [cr, dw, dw, un,  -1, -1, -1, -1],
                [cr, dw, dw, cr,  -1, -1, -1, -1],
                [po1, po1, po1, po1, -1, -1, -1, -1],
                [cr, cr, cr, cr, -1, 3, -1, -1],
                [-1, -1, -1, -1, 6, 3, -1, -1],
                [-1, -1, -1, -1, 1, 5, 1, 5],
                [-1, -1, -1, -1, -1, -1, 6, 3],
            ]
            for test_sample in test_samples:            
                masked_sample = test_sample #[test_sample[i] for i in [0, 1, 1, 2]] + [-1, -1, -1, -1]
                filled_sample = np.array(list(model.mpe_inference(sess,
                                                                  np.array(masked_sample, dtype=int))),
                                         dtype=int).flatten()

                # assert that everything besides the masked variables are the same
                masked_sample_copy = np.copy(masked_sample)
                masked_sample_copy[masked_sample_copy == filled_sample] = 0
                assert np.unique(masked_sample_copy)[0] == -1

                self._test_cases.append({
                    'query': [category_abrv(c) for c in masked_sample[:model.num_nodes]] + masked_sample[model.num_nodes:],
                    'response': [category_abrv(c) for c in filled_sample.tolist()[:model.num_nodes]] + filled_sample.tolist()[model.num_nodes:]
                })

        def _report(self):
            return self._test_cases




def run_nodetemplate_experiment(train_kwargs, test_kwargs, seed=None, semantic=False):
    # Node template experiments

    timestamp = time.strftime("%Y%m%d-%H%M")
    train_kwargs['timestamp'] = timestamp
    test_kwargs['timestamp'] = timestamp
    
    print_in_box(["NodeTemplate experiments"], ho="+", vr="+")

    spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}

    template_spn = NodeTemplateSpn(test_kwargs['template'], seed=seed, **spn_params)

    with tf.Session() as sess:
        name = "A_WithinTemplateExperiment_NodeTemplate"
        print_in_box(["Experiment %s" % name])
        print_banner("Start", ch='v')
        exp = TemplateSpnExperiment(TOPO_MAP_DB_ROOT,
                                    template_spn,
                                    root_dir=RESULTS_DIR, name=name)
        exp.load_training_data(*train_kwargs['train_db'], skip_unknown=CategoryManager.SKIP_UNKNOWN)
        train_info = exp.train_models(sess, **train_kwargs)

        model = exp.get_model(test_kwargs['template'])
        # Add constant.
        #SpnModel.add_constant_to_weights(sess, model.root, 200)

        if semantic:
            # Test semantic understanding
            model.init_mpe_states()  # init MPE ops. (before expanding)
            print(model.template)
            test_kwargs['subname'] = 'semantic_%s_%s' % ("-".join(train_kwargs['train_db']), model.template.__name__)
            exp.test(TemplateSpnExperiment.TestCase_SemanticUnderstanding, sess, **test_kwargs)
            print_banner("Finish", ch='^')
        else:
            if test_kwargs['expand']:
                print("Expand the model.")
                model.expand()   # expand the model
            if test_kwargs['inference_type'] == MPE:
                model.init_mpe_states()  # init MPE ops. (after expanding)
            else:
                print("Not initializing MPE states. Using marginal inference")
                model.init_learning_ops()

            # Test local classification correction
            exp.load_testing_data(test_kwargs['test_db'], skip_unknown=CategoryManager.SKIP_UNKNOWN)
            test_kwargs['subname'] = 'random_lh_%s' % model.template.__name__
            report_rnd = exp.test(TemplateSpnExperiment.TestCase_RandomCompletionTask, sess, **test_kwargs)
            print_banner("Finish", ch='^')

            report_dw = {}
            if model.template == ThreeNodeTemplate:
                test_kwargs['subname'] = 'doorway_lh'
                report_dw = exp.test(TemplateSpnExperiment.TestCase_DoorwayCompletionTask, sess, **test_kwargs)
            return train_info, report_dw, report_rnd



# def run_edgetemplate_experiment(train_kwargs, test_kwargs, semantic=False, seed=None):
#     # Edge template experiments
    
#     timestamp = time.strftime("%Y%m%d-%H%M")
#     train_kwargs['timestamp'] = timestamp
#     test_kwargs['timestamp'] = timestamp    
    
#     print_in_box(["EdgeTemplate experiments"], ho="+", vr="+")

#     spn_params = {k:train_kwargs[k] for k in ['num_decomps', 'num_subsets', 'num_mixtures', 'num_input_mixtures']}

#     pair_edge_spn = EdgeTemplateSpn(PairEdgeTemplate, seed=seed, **spn_params)
#     single_edge_spn = EdgeTemplateSpn(SingleEdgeTemplate, seed=seed, **spn_params)
    
#     with tf.Session() as sess:
#         name = "A_WithinTemplateExperiment_EdgeTemplate"
#         print_in_box(["Experiment %s" % name])
#         print_banner("Start", ch='v')
#         exp = TemplateSpnExperiment(TOPO_MAP_DB_ROOT, pair_edge_spn, single_edge_spn,
#                                           root_dir=RESULTS_DIR, name=name)
#         exp.load_training_data("Stockholm", "Freiburg", skip_unknown=CategoryManager.SKIP_UNKNOWN)
#         train_info = exp.train_models(sess, **train_kwargs)

#         model = exp.main_model
        
#         # Test semantic understanding
#         if semantic:
#             model.init_mpe_states()  # init MPE ops. (before expanding)
#             exp.test(TemplateSpnExperiment.TestCase_SemanticUnderstanding, sess, **test_kwargs)
#             exp.test(TemplateSpnExperiment.TestCase_GeometricUnderstanding, sess, **test_kwargs)
#             print_banner("Finish", ch='^')
#         else:
#             model.expand()   # expand the model
#             model.init_mpe_states()  # init MPE ops. (after expanding)

#             # Test local classification correction
#             exp.load_testing_data("Saarbrucken", skip_unknown=CategoryManager.SKIP_UNKNOWN)
#             report_dw = exp.test(TemplateSpnExperiment.TestCase_DoorwayCompletionWithLikelihoods, sess, **test_kwargs)
#             report_rnd = exp.test(TemplateSpnExperiment.TestCase_RandomCompletionWithLikelihoods, sess, **test_kwargs)
#             print_banner("Finish", ch='^')
#             return train_info, report_dw,report_rnd



def eval_results(train_info, report_dw,
                 report_rnd,
                 param_training=True, node=True):
    dw_results = report_dw['results']
    rnd_results = report_rnd['results']
    if node:
        likelihoods = train_info['likelihoods']['ThreeNodeTemplate']
    else:
        likelihoods = train_info['likelihoods']['PairEdgeTemplate']

    # We look for:
    #   doorway (center) accuracy  0.25
    #   overall accuracy           0.25
    #   likelihood drop            0.25
    #   zero behavior?            -0.25

    doorway_accuracy = dw_results['center'][-1]
    overall_accuracy = rnd_results['_overall_']
    if param_training:
        likelihood_drop = (likelihoods[-1] - likelihoods[0])
    else:
        likelihood_drop = 0   # no effect if parameter of interest is for testing.
        
    zero_behavior = 0
    for catg in rnd_results:
        if not catg.startswith("_"):
            if rnd_results[catg][-1] == 0.0:
                zero_behavior += 1
    score = doorway_accuracy * 0.25 \
            + overall_accuracy * 0.25 \
            + likelihood_drop * 0.25 \
            - zero_behavior * 0.25
    return score, [doorway_accuracy, overall_accuracy, zero_behavior, likelihood_drop]
        

def analyze(results_dir, parameter_ranges, default_train, default_test):
    """
    Control variable analysis
    """

    save_dir = os.path.join(results_dir, "analysis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fnode = open(os.path.join(save_dir, "node_analysis.csv"), 'w')
    fedge = open(os.path.join(save_dir, "edge_analysis.csv"), 'w')

    csv_header = ["changed"] + ['value'] + ["score"] \
                 + ['doorway_ctr_acc', 'overall_acc', 'zero_behavior', 'likelihood_drop'] \
                 + sorted(list(parameter_ranges['train'].keys())) \
                 + sorted(list(parameter_ranges['test'].keys()))
    node_writer = csv.writer(fnode, delimiter=',', quotechar='"')
    node_writer.writerow(csv_header)
    edge_writer = csv.writer(fedge, delimiter=',', quotechar='"')
    edge_writer.writerow(csv_header)

    rcmd_train = {
        'E':{
            k:(default_train[k], float('-inf')) for k in default_train
        },
        'N':{
            k:(default_train[k], float('-inf')) for k in default_train
        }
    }
    rcmd_test = {
        'E':{
            k:(default_test[k], float('-inf')) for k in default_test
        },
        'N':{
            k:(default_test[k], float('-inf')) for k in default_test
        }
    }

    # Training parameters
    for tp in sorted(parameter_ranges['train']):
        for val in sorted(parameter_ranges['train'][tp]):

            train_kwargs = dict(default_train)
            train_kwargs[tp] = val
            
            # Node
            train_info, report_dw, report_rnd = run_nodetemplate_experiment(train_kwargs, default_test)
            score, metrics = eval_results(train_info, report_dw, report_rnd, node=True, param_training=True)
            row = [tp, val, score,
                   *metrics,
                   *[train_kwargs[k] for k in sorted(parameter_ranges['train'])],
                   *[default_test[k] for k in sorted(parameter_ranges['test'])]
            ]
            node_writer.writerow(row)
            fnode.flush()
            if score > rcmd_train['N'][tp][1]:
                rcmd_train['N'][tp] = (val, score)
                        
            # Edge
            train_info, report_dw, report_rnd = run_edgetemplate_experiment(train_kwargs, default_test)
            score, metrics = eval_results(train_info, report_dw, report_rnd, node=False, param_training=True)
            row = [tp, val, score,
                   *metrics,
                   *[train_kwargs[k] for k in sorted(parameter_ranges['train'])],
                   *[default_test[k] for k in sorted(parameter_ranges['test'])]
            ]
            edge_writer.writerow(row)
            fedge.flush()
            if score > rcmd_train['E'][tp][1]:
                rcmd_train['E'][tp] = (val, score)
            

    # Testing parameters
    for tp in sorted(parameter_ranges['test']):
        for val in sorted(parameter_ranges['test'][tp]):
            test_kwargs = dict(default_test)
            test_kwargs[tp] = val
            
            # Node
            train_info, report_dw, report_rnd = run_nodetemplate_experiment(default_train, test_kwargs)
            score, metrics = eval_results(train_info, report_dw, report_rnd, node=True, param_training=False)
            row = [tp, val, score,
                   *metrics
                   *[default_train[k] for k in sorted(parameter_ranges['train'])],
                   *[test_args[k] for k in sorted(parameter_ranges['test'])],
            ]
            node_writer.writerow(row)
            fnode.flush()
            if score > rcmd_test['N'][tp][1]:
                rcmd_test['N'][tp] = (val, score)
                        
            # Edge
            train_info, report_dw, report_rnd = run_edgetemplate_experiment(default_train, test_kwargs)
            score, metrics = eval_results(train_info, report_dw, report_rnd, node=False, param_training=False)
            row = [tp, val, score,
                   *metrics
                   *[default_trains[k] for k in sorted(parameter_ranges['train'])],
                   *[test_args[k] for k in sorted(parameter_ranges['test'])],
            ]
            edge_writer.writerow(row)
            fedge.flush()
            if score > rcmd_test['E'][tp][1]:
                rcmd_test['E'][tp] = (val, score)

    # csv_header = ["changed"] + ['value'] + ["score"] \
    #              + ['doorway_ctr_acc', 'overall_acc', 'likelihood_drop', 'zero_behavior'] \
    #              + sorted(list(parameter_ranges['train'].keys())) \
    #              + sorted(list(parameter_ranges['test'].keys()))
    node_writer.writerow(['-']*7
                         + [rcmd_train['N'][k] for k in sorted(parameter_ranges['train'])]
                         + [rcmd_test['N'][k] for k in sorted(parameter_ranges['test'])])
    edge_writer.writerow(['-']*7
                         + [rcmd_train['E'][k] for k in sorted(parameter_ranges['train'])]
                         + [rcmd_test['E'][k] for k in sorted(parameter_ranges['test'])])

    fnode.close()
    fedge.close()


if __name__ == "__main__":

    seed = random.randint(200,1000)

    db = ['Stockholm', 'Saarbrucken', 'Freiburg']
    
    # Config
    train_kwargs = {
        'num_partitions': 7,#10,
        'num_batches': 10,
        'save': True,
        'load_if_exists': True,
        'likelihood_thres': 0.1,
        'save_training_info': True,

        # spn_structure
        'num_decomps': 3,#1,
        'num_subsets': 3,#2,
        'num_mixtures': 2,#2,
        'num_input_mixtures': 2,#2

        'train_db': ['Saarbrucken', 'Freiburg']
    }

    test_kwargs = {
        'high_likelihood_correct': (0.995, 0.999),
        'low_likelihood_correct': (0.001, 0.005),
        'high_likelihood_incorrect': (0.995, 0.999),
        'low_likelihood_incorrect': (0.001, 0.005),
        'template': StarTemplate,
        'limit': -1,
        'num_partitions': 2,
        'inference_type': MARGINAL,
        'test_db': list(set(db) - set(train_kwargs['train_db']))[0],
        'expand': False
    }
    

    # Analyis
    parameter_ranges = {
        'train': {
            # 'num_batches': [1, 5, 10, 20],
            # 'num_decomps': [1, 2, 3, 5, 10],              # spn_structure
            # 'num_partitions': [1, 5, 10, 20],            
            # 'num_input_mixtures': [2, 3, 4, 5]            # spn_structure
            # 'num_mixtures': [2, 3, 4, 5],                 # spn_structure
            # 'num_subsets': [2, 3, 4, 5],                  # spn_structure
            # 'upsample_rate': [0, 2, 5, 10, 20],
        },
        'test': {
            'high_likelihood': [(0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.6, 0.9), (0.7, 0.9)],
            'low_likelihood': [(0.2, 0.5), (0.3, 0.5), (0.4, 0.5), (0.1, 0.3), (0.2, 0.4)],
        }
    }

    #analyze(results_dir, parameter_ranges, train_kwargs, test_kwargs)
    run_nodetemplate_experiment(train_kwargs, test_kwargs, semantic=False, seed=seed)
    #run_edgetemplate_experiment(train_kwargs, test_kwargs, semantic=False, seed=seed)
