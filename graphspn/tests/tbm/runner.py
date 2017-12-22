# This file contains necessasry components to run the experiments.
# 
# author: Kaiyu Zheng

import os, sys
import random
import heapq
import time
import numpy as np
from pprint import pprint

from spn_topo.tests.runner import Experiment, TestCase
from spn_topo.tbm.template import EdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, NodeTemplate, ThreeRelTemplate, StarTemplate
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn
from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.util import CategoryManager, ColdDatabaseManager
from spn_topo.tests.constants import COLD_ROOT, TOPO_MAP_DB_ROOT


class TbmExperiment(Experiment):
    """
    (graph) template-based method experiments.

    Allows running the same test cases several times conveniently, without overwriting previous results.

    Special definiton:
       The following is a parameter for the test() function. Pay special attention of its format.
       subname (str): used to distinguish test cases of the same type. It is essential to ensure
                      that the subname follows the following format:
                      {General}--{Specific} where 'General' is a broader name such as 'random_policy',
                      and 'Specific' is a narrower name such as the sequence id of the tested instance.
                      'General' is required. And '--{Specific}' is optional.

    Example usage:

       three_node_spn = NodeTemplateSpn(ThreeNodeTemplate, ...)
       pair_node_spn = NodeTemplateSpn(PairTemplate, ...)
       single_node_spn = NodeTemplateSpn(SingletonTemplate, ...)

       with tf.Session as sess:
       
           exp = WithinTemplateSpnExperiment(db_root, three_node_spn, pair_node_spn, single_node_spn)
           exp.load_training_data("Stockholm", "Freiburg")
           exp.train_models(sess)

           exp.load_testing_data("Saarbrucken")
           exp.test(WithinTemplateSpnExperiment.TestCaseA)
           exp.test(WithinTemplateSpnExperiment.TestCaseB)
           ...
    """

    def __init__(self, db_root, *spns, **kwargs):
        """
        db_root (str): path to the root directory of databases. See TopoMapDataset for details.
        spns (list): A list of untrained SpnModel objects. Note that we can obtain
                     the class of its modeled template by [obj].template.

        **kwargs:
        root_dir (str): path to the root directory of the directory* that holds data/results generated
                        by this experiment.  * at {root_dir}/{name}. Default "~"
        name (str): name of this experiment.
        """
        root_dir = kwargs.get("root_dir", "~")
        name = kwargs.get("name", "TbmExperiment")
        super().__init__(name=name)

        # sort spns by template size.
        self._spns = sorted(spns, key=lambda x: x.template.size(), reverse=True)
        # template mode: 0 means node template, 1 means edge template.
        self._template_mode = self._spns[0].template.code()
        
        self._dataset = TopoMapDataset(db_root)
        self._train_db = []
        self._test_db = []
        self._train_seqs = None  # *_seqs are used only when we mix the sequences from different buildings.
        self._test_seqs = None
        self._root_path = os.path.join(root_dir, name)
        self._completed = {}  # dictionary from 'subname' to a set of test cases completed under that subname.

        # Map from db_name to number of training/testing samples generated from it.
        self._data_count = {}


    @property
    def train_db(self):
        return self._train_db


    @property
    def test_db(self):
        return self._test_db
    

    @property
    def main_model(self):
        """
        Returns the TemplateSpn with largest size.
        """
        return self._spns[0]

    def get_model(self, template):
        """
        Returns the model corresponding to the given template class
        """
        for spn in self._spns:
            if spn.template == template:
                return spn
        return None

    @property
    def root_path(self):
        return self._root_path
    

    @property
    def template_mode(self):
        """
        0 means node template, 1 means edge template.
        """
        return self._template_mode
        
    @property
    def dataset(self):
        return self._dataset


    def model_save_path(self, model):
        trained_dbs = "-".join(self._train_db)
        return os.path.join(self._root_path, 'models', "%s_%d_%s.spn" % (model.template.__name__, CategoryManager.NUM_CATEGORIES, trained_dbs))


    def data_count(self, db=None, update=False):
        if update:
            if db is None:
                raise ValueError("Do not know what db to update.")
            self._data_count.update(db)
        else:
            if db is None:
                return self._data_count
            else:
                if db in self._data_count:
                    return self._data_count[db]
                else:
                    return 0  # the model is already trained. Didn't load any training data.
        

    def load_training_data(self, *db_names, skip_unknown=False, segment=False, mix_ratio=None):
        """
        Load training data from given dbs
        
        segment (bool): If True, segment the topological maps while loading them,
                        such that each node is a room.
        mix_ratio (float): When not None, it is the percentage of sequences in the given dbs used for training (e.g. 0.8).
                           If this value is set, the testing data would be the remaining sequences that take 1-mixed_ratio
                           percentage
        """
        for db_name in db_names:
            self._dataset.load(db_name, skip_unknown=skip_unknown, segment=segment)
            self._train_db.append(db_name)
            if mix_ratio is None:
                print("Loaded %d training sequences from database %s" % (len(self._dataset.get_topo_maps(db_name=db_name, amount=-1)),
                                                                         db_name))
        if mix_ratio is not None:
            self._train_seqs, self._test_seqs = self._dataset.split(mix_ratio, *db_names)
            print("Loaded %d training sequences from databases %s" % (len(self._train_seqs), list(db_names)))
                                                                      


    def load_testing_data(self, *db_names, skip_unknown=False, tiny_size=0, segment=False):
        """
        Load training data from given dbs. If mixed_ratio is provided when calling load_training_data, this function will
        load from sequences unused for training in the training dbs, and the provided `db_names` will be ignored.

        tiny_size (int): If want to do tests on tiny graphs, set this to the number of
                         nodes you want on that graph. Default: 0 (means not using tiny graphs.)
        segment (bool): If True, segment the topological maps while loading them,
                        such that each node is a room. Ignored when tiny_size > 0
        """
        if self._test_seqs is not None:
            loaded = True
            for db_name in db_names:
                if db_name not in self._train_db:
                    loaded = False
                    break
            if loaded:
                print("Loaded %d testing sequences from databases same as training." % (len(self._test_seqs)))
                if tiny_size <= 0:
                    return # Do nothing, because sequences have been loaded already
                else:
                    self._dataset.load_tiny_graphs(None, seq_ids=self._test_seqs, skip_unknown=skip_unknown, num_nodes=tiny_size)
                    return
            
        for db_name in db_names:
            if tiny_size <= 0:
                # We want to test on full graphs.
                self._dataset.load(db_name, skip_unknown=skip_unknown, segment=segment)
            else:
                # We want to test on tiny graphs.
                self._dataset.load_tiny_graphs(db_name, skip_unknown=skip_unknown, num_nodes=tiny_size)
            self._test_db.append(db_name)
            print("Loaded %d testing sequences from database %s" % (len(self._dataset.get_topo_maps(db_name=db_name, amount=-1)),
                                                                    db_name))


    def get_test_instances(self, db_name=None, amount=None, seq_id=None, auto_load_splitted=True):
        """
        Returns a dictionary that maps from db_seq_id to topological map instance. If `auto_load_splitted` is True,
        will load the instances resulted from splitting certain sequences into two groups (see load_training_data
        for more details), and ignore all other parmeter settings. Otherwise, will work the same as TopoMapDataset.get_topo_maps().

        Note that a db_seq_id is a string of the format "{db}-{seq_id}"
        """
        topo_maps = {}
        if auto_load_splitted and self._test_seqs is not None:
            for db_seq_id in self._test_seqs:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = list(self.dataset.get_topo_maps(db_name=db_name, seq_id=seq_id).items())[0][1]
                if amount is not None and len(topo_maps) >= amount:
                    return topo_maps
        else:
            topo_maps_pp = self.dataset.get_topo_maps(db_name=db_name, amount=amount, seq_id=seq_id)
            for seq_id in topo_maps_pp:
                topo_maps[db_name+"-"+seq_id] = topo_maps_pp[seq_id]
        return topo_maps


    def train_models(self, sess, *args, **kwargs):
        """
        Train the untrained models provided in initialization.

        sess (tf.Session): a tensorflow session.

        **kwargs:
           save (bool): saves the trained model. Path is {root_dir}/{name}/models/{TemplateSpn}.spn.
                        Default: False.
           load_if_exists (bool): if True, checks if there's already a trained model saved
                                  in conventional path. If so, load it; don't train it. Default: False.
           num_partitions (int): number of partition attempts. See TopoMapDataset for details.
                                 Default 10.
           num_batches (int): number of batches to split the training data into.
                              Default: 1
           likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
           will_upsample func(model, sample): a function that determines whether the given
                                             sample should be upsampled. Returns True if upsample.
           upsample_rate (float): the percentage of increase samples. (e.g. 0.1 means the
                                  upsampled samples will have 10% more presence in the training set.)
           timestamp (str): starting timestamp of the experiment that requested this training.
           save_training_info (bool): If set true, save a formatted file of the configuration and training
                                      data count. The file is located at:
                                           {root_dir}/{name}/models/training_info_{timestamp}
        """
        save = kwargs.get("save", False)
        load_if_exists = kwargs.get("load_if_exists", False)
        num_partitions = kwargs.get("num_partitions", 10)
        num_batches = kwargs.get("num_batches", 1)
        likelihood_thres = kwargs.get("likelihood_thres", 0.05)
        will_upsample = kwargs.get("will_upsample", None)
        upsample_rate = kwargs.get("upsample_rate", None)
        timestamp = kwargs.get("timestamp", None)
        save_training_info = kwargs.get("save_training_info", False)

        likelihoods = {}

        train_info = {}

        for model in self._spns:

            model_save_path = self.model_save_path(model)
            if load_if_exists and os.path.exists(model_save_path):
                model.init_weights_ops()
                model.initialize_weights(sess)
                model.load(model_save_path, sess)
                model.init_learning_ops()
                continue

            # Load training samples
            if self._train_seqs:
                source = {'seq_ids': self._train_seqs}
            else:
                source = {'db_names': self._train_db}
            if self._template_mode == NodeTemplate.code():    ## NodeTemplate
                samples_dict = self._dataset.create_template_dataset(model.template,
                                                                     num_partitions=num_partitions,
                                                                     **source)
            else:
                 raise ValueError("Invalid template mode %d" % self._template_mode)
                
            # Upsampling.
            if upsample_rate is not None and upsample_rate > 0:
                total_upsampled = 0
                original_amount = 0
                for db in samples_dict:
                    samples = samples_dict[db]
                    uniq, counts = np.unique(samples, axis=0, return_counts=True)  # Requires Numpy >= 1.13
                    sample_counts = dict(zip(map(tuple, uniq), counts))
                    for sample, count in sample_counts.items():
                        if will_upsample(model, sample):
                            amount_add = int(round(count * upsample_rate))
                            total_upsampled = total_upsampled + amount_add
                            original_amount = original_amount + int(count)
                            samples.extend([sample] * amount_add)
                        
                self._data_count['upsampled_%s' % model.template.__name__] = {'added': total_upsampled,
                                                                              'original': original_amount}

            # Convert dictionary into list of samples. Shape is (D,n) if NodeTemplate,
            #                                                (D,2*n) if Edgetemplate
            samples = np.array([ p for db in samples_dict for p in samples_dict[db]], dtype=int)

            self._data_count['train_%s' % model.template.__name__] = {db:len(samples_dict[db]) for db in samples_dict}
            
            if self._template_mode == 1:  ## EdgeTemplate
                samples = samples.reshape(-1, 2, model.num_nodes)

            # initialize model with random weights
            model.generate_random_weights()
            model.init_weights_ops()
            model.init_learning_ops()
            model.initialize_weights(sess)
            lh = model.train(sess, samples, shuffle=True, num_batches=num_batches, likelihood_thres=likelihood_thres)
            likelihoods[model.template.__name__] = lh
            
            if save:
                sys.stdout.write("Saving trained %s-SPN saved to path: %s ..." % (model.template.__name__, model_save_path))
                if not os.path.exists(os.path.join(self._root_path, 'models')):
                    os.makedirs(os.path.join(self._root_path, 'models'), exist_ok=True)
                model.save(model_save_path, sess)
                model._save_path = model_save_path
                
            train_info[model.template.__name__] = {'config': { k: dict(kwargs)[k]
                                                         for k in kwargs
                                                         if k not in ['will_upsample', 'save_training_info'] },
                                             'data_stats': self._data_count,
                                             'likelihoods': likelihoods}
        # loop ends

        if save_training_info:
            with open(os.path.join(os.path.dirname(model_save_path), "training_info_%s.log" % timestamp), 'w') as f:
                pprint(dict(kwargs), stream=f)
            sys.stdout.write("Saved training info. \n")
        return train_info


    def case_path(self, test_case_name, timestamp, subname=None):
        if subname is not None:  # Here, subname refers to the full {General}--{Specific}
            case_name = "%s_%s_%s" % (test_case_name, timestamp, subname)
        else:
            case_name = "%s_%s" % (test_case_name, timestamp)
        save_path = os.path.join(self._root_path, 'results', case_name)
        return save_path, case_name
    
                
    def test(self, test_case, sess, **kwargs):
        """
        High-level test call. Results are saved in {root_dir}/{name}/results/{test_case}_{timestamp}.

        test_case (runner.TestCase): A class that defines behaviors of a test case.
        sess (tf.Session): tensorflow session

        **kwargs: arguments for the test_case's run() method.
           limit (int): Maximum number of test samples (sequences) to be used for testing.
           timestamp (str) identifier for the results directory. Default is current time.
           subname (str): used to distinguish test cases of the same type. It is essential to ensure
                          that the subname follows the following format:
                          {General}--{Specific} where 'General' is a broader name such as 'random_policy',
                          and 'Specific' is a narrower name such as the sequence id of the tested instance.
                          'General' is required. And '--{Specific}' is optional.
        """
        timestamp = kwargs.get("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        
        tc = test_case(self)
        try:
            tc.run(sess, **kwargs)
        except KeyboardInterrupt:
            print("Terminating...")
        subname = kwargs.get("subname", None)
        save_path, case_name = self.case_path(tc.name(), timestamp, subname)
        os.makedirs(save_path)
        report = tc.save_results(save_path=save_path)
        # save testing args
        with open(os.path.join(save_path, "testing_info_%s.log" % timestamp), 'w') as f:
            pprint(dict(kwargs), stream=f)
        # Save case name (used to compute statistics)
        general_subname = subname.split("--")[0]
        if general_subname not in self._completed:
            self._completed[general_subname] = set({})
        self._completed[general_subname].add(case_name)
        return report


    def get_stats(self, *args, **kwargs):
        raise NotImplementedError("`get_stats` is not implemented in TbmExperiment")


    @classmethod
    def create_likelihoods_for_single_node(cls, catg,
                                           high_likelihood_correct, low_likelihood_correct,
                                           high_likelihood_incorrect, low_likelihood_incorrect, 
                                           masked=False, uniform_for_incorrect=False,
                                           consider_placeholders=False, is_placeholder=False):
        """
        Creates a numpy.array of shape (m,) where m is the number of semantic categories. Each element
        is a float number indicating the likelihood of that category.

        catg (int): true category number (semantic value)
        high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
        low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                    and the other (untrue) classes.
        masked (bool): True if this semantic value will be masked to -1.
        uniform_for_incorrect (bool): True if use uniform distribution for likelihoods in incorrectly classified nodes.
        consider_placeholders (bool): If True, will set the likelihoods for a placeholder node to be uniform, regardless
                                      of the high/low likelihoods setting.
        is_placeholder (bool): If True, the current node is a placeholder.
        """
        likelihoods = np.zeros((CategoryManager.NUM_CATEGORIES,))
        if not masked:
            for k in range(CategoryManager.NUM_CATEGORIES):
                if catg == k:
                    likelihoods[k] = random.uniform(high_likelihood_correct[0], high_likelihood_correct[1])
                else:
                    likelihoods[k] = random.uniform(low_likelihood_correct[0], low_likelihood_correct[1])
        else:
            highest = float('-inf')
            # Randomly select a non-groundtruth class to be the correct class
            cc = random.randint(0, CategoryManager.NUM_CATEGORIES)
            while cc == catg:
                cc = random.randint(0, CategoryManager.NUM_CATEGORIES)
            for k in range(CategoryManager.NUM_CATEGORIES):
                if uniform_for_incorrect or (consider_placeholders and is_placeholder):
                    likelihoods[k] = 1.0
                else:
                    if cc == k:
                        likelihoods[k] = random.uniform(high_likelihood_incorrect[0], high_likelihood_incorrect[1])
                    else:
                        likelihoods[k] = random.uniform(low_likelihood_incorrect[0], low_likelihood_incorrect[1])
        # normalize so that likelihoods sum up to 1.
        likelihoods = likelihoods / np.sum(likelihoods)
        return likelihoods
        
    
    @classmethod
    def create_likelihoods_vector(cls, sample, model,
                                  high_likelihood_correct, low_likelihood_correct,
                                  high_likelihood_incorrect, low_likelihood_incorrect, 
                                  masked_sample=None):
        """
        Creates a numpy.array of shape (n, m) where n is the number of semantics variables (per template).
        m is the number of semantic categories. 

        sample (list): contains semantic variable values, in order.
        model (TemplateModel): the spn model that `sample` suits for.
        high_likelihood (tuple): the min & max likelihood of the semantics variable's true class.
        low_likelihood (tuple): the min & max likelihood of the masked semantics variable's true class.
                    and the other (untrue) classes.

        masked_sample (list): Masked sample that contains -1. For masked values,
                      we reverse the pattern of assigning likelihoods to classes.
        """
        likelihoods = np.zeros((model.num_nodes, CategoryManager.NUM_CATEGORIES))
        
        for i in range(len(sample)):
            likelihoods[i] = cls.create_likelihoods_for_single_node(sample[i],
                                                                    high_likelihood_correct, low_likelihood_correct,
                                                                    high_likelihood_incorrect, low_likelihood_incorrect, 
                                                                    masked=masked_sample is not None and masked_sample[i] == -1)
            
        return likelihoods

    
    @classmethod
    def create_instance_spn_likelihoods(cls, template_mode, topo_map, true_catg_map,
                                        high_likelihood_correct, low_likelihood_correct,
                                        high_likelihood_incorrect, low_likelihood_incorrect, 
                                        uniform_for_incorrect=False, consider_placeholders=False):
        """
        Create a dictionary with format as described in mpe_inference():sample_lh.

        template_mode (0): 0 if the experiment is for NodeTemplate. 1 for EdgeTemplate.
        topo_map (TopologicalMap) The topo map
        true_catg_map (dict): a dictionary mapping from node id to groundtruth category value.
        high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
        low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                    and the other (untrue) classes.
        consider_placeholders (bool): If True, will set the likelihoods for a placeholder node to be uniform, regardless
                                      of the high/low likelihoods setting.
        """
        lh = {}

        if template_mode == NodeTemplate.code(): ## NodeTemplate
            for nid in topo_map.nodes:
                lh[nid] = tuple(cls.create_likelihoods_for_single_node(true_catg_map[nid],
                                                                       high_likelihood_correct, low_likelihood_correct,
                                                                       high_likelihood_incorrect, low_likelihood_incorrect, 
                                                                       masked=CategoryManager.category_map(topo_map.nodes[nid].label) == -1,
                                                                       uniform_for_incorrect=uniform_for_incorrect,
                                                                       consider_placeholders=consider_placeholders, is_placeholder=topo_map.nodes[nid].placeholder))
        else:
            raise ValueError("Invalid template mode %d" % self._template_mode)
        return lh


    @classmethod
    def create_category_map_from_likelihoods(cls, template_mode, likelihoods):
        """
        likelihoods (dict) a dictionary with format as described in mpe_inference():sample_lh.
        """
        catg_map = {}
        if template_mode == NodeTemplate.code():
            for nid in likelihoods:
                catg_num = likelihoods[nid].index(max(likelihoods[nid]))
                catg_map[nid] = catg_num
        return catg_map

    
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


    ########## Upsampling ##########
    @classmethod
    def will_upsample(cls, model, sample):
        """
        Returns True if the given sample satisfies any of the upsample rules.
        """
        return cls.check_doorway_connecting(model, sample)# \
            #or cls.check_same_room(model, sample)
    
    @classmethod
    def check_same_room(cls, model, sample):
        """
        Returns True if the given sample has the same class
        """
        if issubclass(model.template, NodeTemplate):
            for i in range(1, len(sample)):
                if sample[i] != sample[i-1]:
                    return False
            return True
                

    @classmethod
    def check_doorway_connecting(cls, model, sample):
        """
        Returns True if the given sample is representing the form

            A - doorway - B
        """
        dw = CategoryManager.category_map('DW')
        # Check the model's template class
        if model.template == ThreeNodeTemplate:
            return sample[1] == dw and sample[0] != dw \
                    and sample[2] != dw and sample[0] != sample[2]
        if model.template == StarTemplate:
            # For star template, the structure is a little complex. Just upsample
            # all cases that contains doorway.
            return dw in sample
        elif model.template == PairEdgeTemplate:
            matched = sample[1] == dw and sample[2] == dw \
                      and sample[0] != dw and sample[3] != dw and sample[0] != sample[3]
            assert len(sample) == model.num_nodes * 2
            return matched
        elif model.template == ThreeRelTemplate:
            # First three elements in sample are the three nodes.
            assert len(sample) == 4
            return sample[1] == dw and sample[0] != dw \
                    and sample[2] != dw and sample[0] != sample[2]
        else:
            return False




# Policy functions
def doorway_policy(topo_map, node, **kwargs):
    return node.label == "DW"

def random_policy(topo_map, node, rand_rate=0.2):
    return random.uniform(0, 1.0) <= rand_rate

def random_fixed_policy(topo_map, node, rate_occluded=0.2):
    """
    Randomly occlude a fixed percentage of nodes from topo_map. Default
    percentage is 0.2.
    """
    cur_num = sum(1 for nid in topo_map.nodes if CategoryManager.category_map(topo_map.nodes[nid].label) == -1)
    if cur_num > len(topo_map.nodes) * rate_occluded:
        return False
    return True

def random_fixed_plus_placeholders(topo_map, node, rate_occluded=0.2):
    """
    Randomly occlude a fixed percentage of nodes from topo_map. Default
    percentage is 0.2. ALSO occlude placeholder nodes, but does not consider placeholders when
    counting the number of masked nodes; The number of masked nodes does not depend on placeholder nodes.
    """
    if node.placeholder:
        return True
    cur_num = sum(1 for nid in topo_map.nodes if CategoryManager.category_map(topo_map.nodes[nid].label) == -1)
    num_ph = topo_map.num_placeholders()
    if cur_num - num_ph > (len(topo_map.nodes) - num_ph) * rate_occluded:
        return False
    return True


# More utility functions for tbm tests

def get_noisification_level(high_likelihood_correct, low_likelihood_correct,
                            high_likelihood_incorrect, low_likelihood_incorrect,
                            uniform_for_incorrect=False):
    """
    This function computes the average highest likelihood - next likelihood and std.
    These numbers reflect the extremity of noisification.
    - For nodes made correct (D_80):
    A closer value indicate more noisification
    and less confidence in local classification results.
    - For nodes made incorrect (D_20):
    we focus on the difference between the true class's likelihood and the highest likelihood.
    If the difference is large, then nosification is large.
    """
    coldmgr = ColdDatabaseManager("Stockholm", COLD_ROOT)
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    print("Loading data...")
    dataset.load("Stockholm", skip_unknown=True)
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=-1)
    stat_incrct = []
    stat_crct = []
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        groundtruth = topo_map.current_category_map()
        topo_map.mask_by_policy(random_policy)
        masked = topo_map.current_category_map()
        likelihoods = TbmExperiment.create_instance_spn_likelihoods(0, topo_map,
                                                                    groundtruth,
                                                                    high_likelihood_correct, low_likelihood_correct,
                                                                    high_likelihood_incorrect, low_likelihood_incorrect,
                                                                    uniform_for_incorrect=uniform_for_incorrect)
        for nid in likelihoods:
            if masked[nid] == -1:
                # Made incorrect
                truecl_lh = likelihoods[nid][groundtruth[nid]]
                largest_lh = max(likelihoods[nid])
                stat_incrct.append(largest_lh - truecl_lh)
            else:
                # Not made incorrect
                largest2 = heapq.nlargest(2, likelihoods[nid])
                stat_crct.append(largest2[0] - largest2[1])

    result = {
        '_D_20_': {
            'avg': np.mean(stat_incrct),
            'std': np.std(stat_incrct),
        },
        '_D_80_': {
            'avg': np.mean(stat_crct),
            'std': np.std(stat_crct),
        }
    }
    return result
