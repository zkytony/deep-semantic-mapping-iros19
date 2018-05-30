# Contains implementation of NodeTemplateSpn (spn for node templates)
# and EdgeTemplateSpn (spn for edge templates).
#
# This file contains:
#
# TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn, EdgeRelationTemplateSpn, InstanceSpn
#
# Note: All likelihoods fed into the network should be in the log space. We use
# the spn.RawInput which directly feeds the log value into the SPN which preserves
# precision.
#
# author: Kaiyu Zheng

import sys
from abc import abstractmethod
import tensorflow as tf
import libspn as spn
import numpy as np
from numpy import float32
import random
import copy
import os

from deepsm.graphspn.spn_model import SpnModel, mod_compute_graph_up
from deepsm.graphspn.tbm.template import EdgeTemplate, NodeTemplate, SingleEdgeTemplate, EdgeRelationTemplate
from deepsm.util import CategoryManager, ColdDatabaseManager
from deepsm.experiments.common import GROUNDTRUTH_ROOT, COLD_ROOT


class TemplateSpn(SpnModel):

    def __init__(self, template, *args, **kwargs):
        """
        Initialize an TemplateSpn.

        template (NodeTemplate or EdgeTemplate): a class of the modeled template.

        **kwargs
           seed (int): seed for the random generator. Set before generating
                       the structure.

        """
        super().__init__(*args, **kwargs)

        self._template = template
        self._dense_gen = spn.DenseSPNGenerator(num_decomps=self._num_decomps, num_subsets=self._num_subsets,
                                                num_mixtures=self._num_mixtures, input_dist=self._input_dist,
                                                num_input_mixtures=self._num_input_mixtures)
        self._expanded = False
        self._rnd = random.Random()
        self._seed = kwargs.get('seed', None)
        self._saved_path = None
        

    @property
    def template(self):
        return self._template


    @property
    def expanded(self):
        return self._expanded


    @property
    @abstractmethod
    def root(self):
        pass

    @property
    def dense_gen(self):
        return self._dense_gen

    
    @property
    def rnd(self):
        return self._rnd

    
    @property
    def seed(self):
        return self._seed

    
    @property
    def num_nodes(self):
        """
        Number of semantic variables (i.e. number of nodes represented by the modeled template).
        
        Note: for edge templates, num_nodes equals to the number of covered edges times 2, because
              each edge has two nodes.
        """
        return self._num_nodes

    def generate_random_weights(self, trainable=True):
        """
        Generates random weights for this spn.
        """
        weights_gen = spn.WeightsGenerator(init_value=spn.ValueType.RANDOM_UNIFORM(self.weight_init_min,
                                                                                   self.weight_init_max),
                                           trainable=trainable)
        weights_gen.generate(self._root)

    def init_weights_ops(self):
        print("Generating weight initialization Ops...")
        self._initialize_weights = spn.initialize_weights(self._root)


    def init_learning_ops(self):
        print("Initializing learning Ops...")
        learning = spn.EMLearning(self._root, log=True, value_inference_type = self._value_inference_type,
                                  additive_smoothing = self._additive_smoothing_var)
        self._reset_accumulators = learning.reset_accumulators()
        self._accumulate_updates = learning.accumulate_updates()
        self._update_spn = learning.update_spn()
        self._train_likelihood = learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)


    def initialize_weights(self, sess):
        print("Initializing weights...")
        sess.run(self._initialize_weights)


    def expand(self):
        """
        Expand input to include likelihoods for semantics.

        Do nothing if already expanded.
        """
        if not self._expanded:
            self._likelihood_inputs = spn.RawInput(num_vars=self._num_nodes * CategoryManager.NUM_CATEGORIES,
                                                   name=self.vn['LH_CONT'])
            self._semantic_inputs = spn.IVs(num_vars=self._num_nodes, num_vals=CategoryManager.NUM_CATEGORIES,
                                            name=self.vn['SEMAN_IVS'])
            prods = []
            for i in range(self._num_nodes):
                for j in range(CategoryManager.NUM_CATEGORIES):
                    prod = spn.Product(
                        (self._likelihood_inputs, [i*CategoryManager.NUM_CATEGORIES + j]),
                        (self._semantic_inputs, [i*CategoryManager.NUM_CATEGORIES + j])
                    )
                    prods.append(prod)
            self._conc_inputs.set_inputs(*map(spn.Input.as_input, prods))
            self._expanded = True


    def _start_training(self, samples, num_batches, likelihood_thres, sess, func_feed_samples):
        """
        Helper for train() in subclasses. Weights should have been initialized.

        `samples` (numpy.ndarray) numpy array of shape (D, ?).
        `func_feed_samples` (function; start, stop) function that feeds samples into the network.
                            It runs the train_likelihood, avg_train_likelihood, and accumulate_updates
                            Ops.
        """
        print("Resetting accumulators...")
        sess.run(self._reset_accumulators)

        batch_size = samples.shape[0] // num_batches

        likelihoods_log = []

        prev_likelihood = 100
        likelihood = 0
        epoch = 0
        while (abs(prev_likelihood - likelihood) > likelihood_thres):
            prev_likelihood = likelihood
            likelihoods = []
            for batch in range(num_batches):
                start = (batch)*batch_size
                stop = (batch+1)*batch_size
                print("EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)

                ads = max(np.exp(-epoch*self._smoothing_decay)*self._additive_smoothing,
                          self._min_additive_smoothing)
                sess.run(self._additive_smoothing_var.assign(ads))
                print("Smoothing: ", sess.run(self._additive_smoothing_var))

                train_likelihoods_arr, avg_train_likelihood_val, _, = func_feed_samples(self, samples, start, stop)

                # Print avg likelihood of this batch data on previous batch weights
                print("Avg likelihood (this batch data on previous weights): %s" % (avg_train_likelihood_val))
                likelihoods.append(avg_train_likelihood_val)
                # Update weights
                sess.run(self._update_spn)

            likelihood = sum(likelihoods) / len(likelihoods)
            print("Avg likelihood: %s" % (likelihood))
            likelihoods_log.append(likelihood)
            epoch += 1
            sess.run(self._reset_accumulators)
        return likelihoods_log

    @staticmethod
    def dup_fun_up(inpt, *args,
                    conc=None, tmpl_num_vars=[0], tmpl_num_vals=[0], graph_num_vars=[0], labels=[[]]):
        """
        Purely for template spn copying only. Supports template with multiple types of IVs.
        Requires that the template SPN contains only one concat node where all inputs go through.

        labels: (2D list) variable's numerical label, used to locate the variable's position in the big IVs.
                If there are multiple types of IVs, then this should be a 2D list, where each inner
                list is the label (starting from 0) for one type of IVs, and each outer list represents
                one type of IVs.
        """
        # Know what range of indices each variable takes
        node, indices = inpt
        if node.is_op:
            if isinstance(node, spn.Sum):
                # [2:] is to skip the weights node and the explicit IVs node for this sum.
                copied_node = spn.Sum(*args[2:], weights=args[0])
                assert copied_node.is_valid()
                return copied_node
            elif isinstance(node, spn.Product):
                copied_node = spn.Product(*args)
                assert copied_node.is_valid()
                return copied_node
            elif isinstance(node, spn.Concat):
                # The goal is to map from index on the template SPN's concat node to the index on
                # the instance SPN's concat node.
                
                # First, be able to tell which type of iv the index has
                ranges_tmpl = [0]  # stores the start (inclusive) index of the range of indices taken by a type of iv on template SPN
                ranges_instance = [0]  # stores the start (inclusive) index of the range of indices taken by a type of iv on instance SPN
                for i in range(len(tmpl_num_vars)):
                    ranges_tmpl.append(ranges_tmpl[-1] + tmpl_num_vars[i]*tmpl_num_vals[i])
                    ranges_instance.append(ranges_instance[-1] + graph_num_vars[i]*tmpl_num_vals[i])

                big_indices = []
                for indx in indices:
                    iv_type = -1
                    for i, start in enumerate(ranges_tmpl):
                        if indx < start + tmpl_num_vars[i]*tmpl_num_vals[i]:
                            iv_type = i
                            break
                    if iv_type == -1:
                        raise ValueError("Oops. Something wrong. Index out of range.")

                    # Then, figure out variable index and offset (w.r.t. template Concat node)
                    varidx = (indx - ranges_tmpl[iv_type]) // tmpl_num_vals[iv_type]
                    offset = (indx - ranges_tmpl[iv_type]) - varidx * tmpl_num_vals[iv_type]
                    # THIS IS the actual position of the variable's inputs in the big Concat.
                    varlabel = labels[iv_type][varidx]
                    big_indices.append(ranges_instance[iv_type] + varlabel * tmpl_num_vals[iv_type] + offset)
                return spn.Input(conc, big_indices)
        elif isinstance(node, spn.Weights):
            return node
        else:
            raise ValueError("Unexpected node %s. We don't intend to deal with IVs here. Please remove them from the concat." % node)
    # END fun_up
# -- END TemplateSpn -- #



class NodeTemplateSpn(TemplateSpn):

    """
    Spn on top of an n-node template.
    (The connectivity is not considered; it is
    implicit in the training data)
    """

    def __init__(self, template, *args, **kwargs):
        """
        Initialize an NodeTemplateSpn.

        template (NodeTemplate): a subclass of NodeTemplate.

        **kwargs:
           seed (int): seed for the random generator. Set before generating
                       the structure.
        """
        super().__init__(template, *args, **kwargs)

        self._num_nodes = self.template.num_nodes()

        # Initialize structure and learning ops
        self._init_struct(rnd=self._rnd, seed=self._seed)


    @property
    def vn(self):
        return {
            'CATG_IVS':"Catg_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'CONC': "Conc_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'SEMAN_IVS': "Exp_Catg_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'LH_CONT': "Exp_Lh_%s_%d" % (self.__class__.__name__, self._num_nodes)
        }
        
    @property
    def root(self):
        return self._root


    def _init_struct(self, *args, rnd=None, seed=None):
        """
        Initialize the structure for training.  (private method)

        rnd (Random): instance of a random number generator used for
                      dense generator. 
        """
        # An input to spn goes through a concat node before reaching the network. This makes
        # it convenient to expand the network downwards, by swapping the inputs to the concat
        # nodes. 

        # Inputs to the spn.
        self._catg_inputs = spn.IVs(num_vars = self._num_nodes,
                               num_vals = CategoryManager.NUM_CATEGORIES,
                               name = self.vn['CATG_IVS'])

        # Concat nodes, used to connect the inputs to the spn.
        self._conc_inputs = spn.Concat(spn.Input.as_input((self._catg_inputs, list(range(self._num_nodes*CategoryManager.NUM_CATEGORIES)))),
                                       name=self.vn['CONC'])

        # Generate structure, weights, and generate learning operations.
        print("Generating SPN structure...")
        if seed is not None:
            print("[Using seed %d]" % seed)
            rnd = random.Random(seed)
        self._root = self._dense_gen.generate(self._conc_inputs, rnd=rnd)
        

    def train(self, sess, *args, **kwargs):
        """
        Train the SPN. Weights should have been initialized.

        sess (tf.Session): Tensorflow session object
        *args:
          samples (numpy.ndarray): A numpy array of shape (D,n) where D is the
                                   number of data samples, and n is the number
                                   of nodes in template modeled by this spn.

        **kwargs:
          shuffle (bool): shuffles `samples` before training. Default: False.
          num_batches (int): number of batches to split the training data into.
                             Default: 1
          likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
        """
        def feed_samples(self, samples, start, stop):
            return sess.run([self._train_likelihood, self._avg_train_likelihood,
                             self._accumulate_updates],
                            feed_dict={self._catg_inputs: samples[start:stop]})

        
        samples = args[0]
        D, n = samples.shape
        if n != self._num_nodes:
            raise ValueError("Invalid shape for `samples`." \
                             "Expected (?,%d), got %s)" % (self._num_nodes, samples.shape))
        
        shuffle = kwargs.get('shuffle', False)
        num_batches = kwargs.get('num_batches', 1)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)

        if shuffle:
            np.random.shuffle(samples)

        # Starts training
        return self._start_training(samples, num_batches, likelihood_thres, sess, feed_samples)
        
    
    def evaluate(self, sess, *args, **kwargs):
        """
        Feeds inputs into the network and return the output of the network. Returns the
        output of the network

        sess (tf.Session): a session.

        *args:
          sample (numpy.ndarray): an (n,) numpy array as a sample.
          
          <if expanded>
          sample_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                      so [?,c] is a likelihood for class c (float).
        """
        # To feed to the network, we need to reshape the samples.
        sample = np.array([args[0]], dtype=int)
        
        if not self._expanded:
            likelihood_val = sess.run(self._train_likelihood, feed_dict={self._catg_inputs: sample})
        else:
            sample_lh = np.array([args[1].flatten()], dtype=float32)
            likelihood_val = sess.run(self._train_likelihood, feed_dict={self._semantic_inputs: sample,
                                                                         self._likelihood_inputs: sample_lh})
            
        return likelihood_val  # TODO: likelihood_val is now a numpy array. Should be a float.

            
    def marginal_inference(self, sess, query, query_lh=None, masked_only=False):
        """
        Performs marginal inference on node template SPN.

        `query` is an numpy array of shape (n,).
        `query_lh` is an numpy array of shape (n,m), used to supply likelihood values per node per class.
        `masked_only` is True if only perform marginal inference on nodes that are
                      masked, i.e. their value in the `query` is -1. Useful when there's no likelihoods and
                      we are just inferring missing semantics.

        Returns: a numpy array of shape (n,m) where n is the number of nodes in the node
                 template, and m is the number of classes. Each value represents the likelihood
                 that the node is assigned to a class.
        """
        # The only way to do marginal inference now is to iterate through all possible
        # classes per node.
        n, = query.shape
        marginals = np.zeros((n, CategoryManager.NUM_CATEGORIES))
        for i in range(query.shape[0]):
            if masked_only and query[i] != -1:
                continue
            orig = query[i]
            for val in range(CategoryManager.NUM_CATEGORIES):
                query[i] = val
                marginals[i, val] = float(self.evaluate(sess, query, query_lh)[0])
        return marginals
    

    def init_mpe_states(self):
        """
        Initialize Ops for MPE inference.
        """
        mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
        if not self._expanded:
            self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs)
        else:
            self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs)


    def mpe_inference(self, sess, *args, **kwargs):
        """
        Feeds inputs with some '-1' and infer their values. Returns the inferred mpe value
        for the query state.

        sess (tf.Session): a session.

        *args:
          query (numpy.ndarray): an (n,) numpy array as a query.

          <if expanded>
          query_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                     so [?,c] is a likelihood for class c (float).
        """
        query = np.array([args[0]], dtype=int)
        if not self._expanded:
            mpe_state_val = sess.run(self._mpe_state, feed_dict={self._catg_inputs: query})
        else:
            query_lh = np.array([args[1].flatten()], dtype=float32)
            mpe_state_val = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: query,
                                                                 self._likelihood_inputs: query_lh})
        return mpe_state_val


    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        spn.JSONSaver(path, pretty=pretty).save(self._root, save_param_vals=True, sess = sess)

    
    def load(self, path, sess):
        """
        Loads the SPN structure and parameters. Replaces the existing structure
        of this SPN.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        loader = spn.JSONLoader(path)
        self._root = loader.load(load_param_vals=True, sess=sess)

        self._catg_inputs = loader.find_node(self.vn['CATG_IVS'])
        self._conc_inputs = loader.find_node(self.vn['CONC'])
        
        if self._expanded:
            self._likelihood_inputs = loader.find_node(self.vn['LH_CONT'])
            self._semantic_inputs = loader.find_node(self.vn['SEMAN_IVS'])


        
# -- END NodeTemplateSpn -- #



class EdgeTemplateSpn(TemplateSpn):

    """
    Spn on top of an edge template. (The connectivity is not considered; it is
    implicit in the training data)
    """

    def __init__(self, template, divisions=8, *args, **kwargs):
        """
        Initialize an EdgeTemplateSPN.

        template (EdgeTemplate): a class of the modeled edge template.
        divisions (int): number of views per place.

        **kwargs:
           seed (int): seed for the random generator. Set before generating
                       the structure.
        """
        super().__init__(template, *args, **kwargs)

        self._divisions = divisions
        self._template = template
        self._num_nodes = self._template.num_edges() * 2  # number of nodes where variables are derived.

        self._init_struct(rnd=self._rnd, seed=self._seed)


    @property
    def template(self):
        return self._template

    @property
    def divisions(self):
        return self._divisions

    @property
    def root(self):
        return self._root

    @property
    def vn(self):
        return {
            'CATG_IVS':"Catg_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
            'VIEW_IVS':"View_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
            'CONC': "Conc_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
            'SEMAN_IVS': "Exp_Catg_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
            'LH_CONT': "Exp_Lh_%s_%d" % (self.__class__.__name__, self._template.num_edges())
        }

    def _init_struct(self, rnd=None, seed=None):
        """
        Initialize the structure for training.  (private method)

        rnd (Random): instance of a random number generator used for
                      dense generator. 
        """
        # An input to spn goes through a concat node before reaching the network. This makes
        # it convenient to expand the network downwards, by swapping the inputs to the concat
        # nodes. 

        # Inputs to the spn. `catg_inputs` are inputs for semantic categories; `view_inputs`
        # are inputs for view numbers.
        self._catg_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = CategoryManager.NUM_CATEGORIES,
                                    name = self.vn['CATG_IVS'])
        self._view_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = self._divisions,
                                    name = self.vn['VIEW_IVS'])

        # Concat nodes, used to connect the category inputs to the spn. 
        self._conc_inputs = spn.Concat(spn.Input.as_input(self._catg_inputs),
                                       name = self.vn['CONC'])

        # Generate structure, weights, and generate learning operations.
        print("Generating SPN structure...")
        if seed is not None:
            print("[Using seed %d]" % seed)
            rnd = random.Random(seed)
        self._root = self._dense_gen.generate(self._conc_inputs, self._view_inputs, rnd=rnd)
    

    def train(self, sess, *args, **kwargs):
        """
        Train the SPN. Weights should have been initialized.

        sess (tf.Session): Tensorflow session object
        *args:
          samples (numpy.ndarray): A numpy array of size (D,2,n) where D is the
                                   number of data samples, and n is the number
                                   of semantic variables in template modeled by
                                   this spn. (e.g. in EdgePairTemplate, n = 4)
                                   (note: n = self.num_nodes)

        **kwargs:
          shuffle (bool): shuffles `samples` before training. Default: False.
          num_batches (int): number of batches to split the training data into.
                             Default: 1
          likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
        """

        def feed_samples(self, samples, start, stop):
            return sess.run([self._train_likelihood, self._avg_train_likelihood,
                             self._accumulate_updates],
                            feed_dict={self._catg_inputs: samples[start:stop, 0],
                                       self._view_inputs: samples[start:stop, 1]})
        
        samples = args[0]
        D, m, n = samples.shape
        if n != self._num_nodes or m != 2:
            raise ValueError("Invalid shape for `samples`." \
                             "Expected (?, %d, %d), got %s)" % (2, self._num_nodes, samples.shape))
            
        shuffle = kwargs.get('shuffle', False)
        num_batches = kwargs.get('num_batches', 1)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)


        if shuffle:
            np.random.shuffle(samples)

        # Starts training
        return self._start_training(samples, num_batches, likelihood_thres, sess, feed_samples)


    def evaluate(self, sess, *args, **kwargs):
        """
        Feeds inputs into the network and return the output of the network. Returns the
        output of the network

        sess (tf.Session): a session.

        *args:
          sample (numpy.ndarray): an (2*n,) numpy array as a sample. n = self.num_nodes

          <if expanded>
          sample_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                      so [?,c] is a likelihood for class c (float).
        """

        if not self._expanded:
            sample = args[0].reshape(2,-1)
            likelihood_val = sess.run(self._train_likelihood, feed_dict={self._catg_inputs: np.array([sample[0,]], dtype=int),
                                                                         self._view_inputs: np.array([sample[1,]], dtype=int)})
        else:
            sample = args[0].reshape(2,-1)
            sample_lh = np.array([args[1].flatten()], dtype=float32)
            likelihood_val = sess.run(self._train_likelihood, feed_dict={self._semantic_inputs: np.array([sample[0,]], dtype=int),
                                                                         self._likelihood_inputs: sample_lh,
                                                                         self._view_inputs: np.array([sample[1,]], dtype=int)})
            
        return likelihood_val  # TODO: likelihood_val is now a numpy array. Should be a float.


    def init_mpe_states(self):
        """
        Initialize Ops for MPE inference.
        """
        mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
        if not self._expanded:
            self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs, self._view_inputs)
        else:
            self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs, self._view_inputs)

            
    def marginal_inference(self, sess, *args, **kwargs):
        """
        Performs marginal inference.
        """
        raise NotImplementedError


    def mpe_inference(self, sess, *args, **kwargs):
        """
        Feeds inputs with some '-1' and infer their values. Returns the inferred mpe value
        for the query state.

        sess (tf.Session): a session.

        *args:
          query (numpy.ndarray): an (2*n,) numpy array as a query. n = self.num_nodes

          <if expanded>
          query_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                     so [?,c] is a likelihood for class c (float).
        """
        if not self._expanded:
            query = args[0].reshape(2,-1)
            mpe_state_val = sess.run(self._mpe_state, feed_dict={self._catg_inputs: np.array([query[0,]], dtype=int),
                                                                 self._view_inputs: np.array([query[1,]], dtype=int)})
        else:
            query = args[0].reshape(2,-1)
            query_lh = np.array([args[1].flatten()], dtype=float32)
            mpe_state_val = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: np.array([query[0,]], dtype=int),
                                                                 self._likelihood_inputs: query_lh,
                                                                 self._view_inputs: np.array([query[1,]], dtype=int)})
        return mpe_state_val


    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        spn.JSONSaver(path, pretty=pretty).save(self._root, save_param_vals=True, sess = sess)

    
    def load(self, path, sess):
        """
        Loads the SPN structure and parameters. Replaces the existing structure
        of this SPN.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        loader = spn.JSONLoader(path)
        self._root = loader.load(load_param_vals=True, sess=sess)

        self._catg_inputs = loader.find_node(self.vn['CATG_IVS'])
        self._view_inputs = loader.find_node(self.vn['VIEW_IVS'])
        self._conc_inputs = loader.find_node(self.vn['CONC'])
        
        if self._expanded:
            self._likelihood_inputs = loader.find_node(self.vn['LH_CONT'])
            self._semantic_inputs = loader.find_node(self.vn['SEMAN_IVS'])
# -- END EdgeTemplateSpn -- #




class EdgeRelationTemplateSpn(TemplateSpn):
    
    def __init__(self, template, divisions=8, *args, **kwargs):
        """
        Initialize an EdgeRelationTemplateSpn.

        template (AbsEdgeRelationTemplate):
                   a class name which is a subclass of AbsEdgeRelationTemplate

        **kwargs:
           seed (int): seed for the random generator. Set before generating
                       the structure.
        """
        super().__init__(template, *args, **kwargs)

        template_tuple = template.to_tuple()
        self._num_nodes = template_tuple[0]
        self._num_edge_pair = template_tuple[1]  # 0 or 1
        self._divisions = divisions

        self._catg_inputs = None
        self._conc_inputs = None
        self._view_dist_input = None

        # Initialize structure and learning ops
        self._init_struct(rnd=self._rnd, seed=self._seed)

    @property
    def vn(self):
        return {
            'CATG_IVS':"Catg_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            'VIEW_IVS':"View_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            'CONC': "Conc_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            'SEMAN_IVS': "Exp_Catg_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            'LH_CONT': "Exp_Lh_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair)
        }


    def _init_struct(self, rnd=None, seed=None):

        # The variables modeled by an EdgeRelationTemplate are nodes (if any) and view distance (if any)
        if self._num_nodes != 0:
            self._catg_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = CategoryManager.NUM_CATEGORIES,
                                        name = self.vn['CATG_IVS'])
            self._conc_inputs = spn.Concat(spn.Input.as_input(self._catg_inputs), name=self.vn['CONC'])
        if self._num_edge_pair != 0:
            # If there are 8 divisions, there could (usually) be 4 different values (1, 2, 3, 4) for the
            # absolute distance between two views. The value 0 could only occur if two views are exactly
            # the same, which is a case we cannot exlucde (the model should be independent from how the
            # graph is constructed and whether there could be two colinear edges). Thus, we do + 1 and the
            # acceptable values are 0, 1, 2, 3, 4. This is also more convenient than having to subtract 1
            # from all view distances.
            self._view_dist_input = spn.IVs(num_vars = 1, num_vals = self._divisions // 2 + 1, name=self.vn['VIEW_IVS'])  ## WARNING Assuming absolute value distance
            if self._conc_inputs is not None:
                self._conc_inputs.add_inputs(self._view_dist_input)
            else:
                self._conc_inputs = spn.Concat(spn.Input.as_input(self._view_dist_input), name=self.vn['CONC'])
            
        # Generate structure, weights, and generate learning operations.
        print("Generating SPN structure...")
        if seed is not None:
            print("[Using seed %d]" % seed)
            rnd = random.Random(seed)
            
        self._root = self._dense_gen.generate(self._conc_inputs, rnd=rnd)


    @property
    def root(self):
        return self._root


    def _get_feed_dict(self, samples, start=0, stop=None):
        """
        samples (numpy.ndarray): If this template has both nodes and edges, then samples should
                                   be of shape (D, n+1) where D is the number of data samples, n
                                   is the number of nodes, and +1 is the edge view distance variable's value
                                   at the last place. If only nodes, then shape is (D, n). If only edges,
                                   then shape is (D, 1)
        """
        if stop is None:
            stop = samples.shape[0]

        feed_dict = {}
        catg_inputs = self._catg_inputs
        if self._expanded:
            catg_inputs = self._semantic_inputs
        
        if self._num_nodes != 0 and self._num_edge_pair != 0:
            feed_dict={catg_inputs: samples[start:stop, :self._num_nodes],
                       self._view_dist_input: samples[start:stop, self.num_nodes:]}
        elif self._num_nodes != 0:
            feed_dict={catg_inputs: samples[start:stop]}
        else:
            feed_dict={self._view_dist_input: samples[start:stop]}
        return feed_dict
    

    def train(self, sess, *args, **kwargs):
        """
        Train this template SPN. Weights should have been initialized.

        *args:
          samples (numpy.ndarray): If this template has both nodes and edges, then samples should
                                   be of shape (D, n+1) where D is the number of data samples, n
                                   is the number of nodes, and +1 is the edge view distance variable's value
                                   at the last place. If only nodes, then shape is (D, n). If only edges,
                                   then shape is (D, 1)
        **kwargs:
          shuffle (bool): shuffles `samples` before training. Default: False.
          num_batches (int): number of batches to split the training data into.
                             Default: 1
          likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
        """
        def feed_samples(self, samples, start, stop):
            feed_dict = self._get_feed_dict(samples, start=start, stop=stop)
            return sess.run([self._train_likelihood, self._avg_train_likelihood,
                             self._accumulate_updates],
                            feed_dict = feed_dict)

        samples = args[0]
        D, m = samples.shape
        if self._num_nodes != 0 and self._num_edge_pair != 0:
            if m != self._num_nodes + 1:
                raise ValueError("Invalid shape for `samples`." \
                                 "Expected (?, %d), got %s)" % (self._num_nodes+1, samples.shape))
        elif self._num_nodes != 0:
            if m != self._num_nodes:
                raise ValueError("Invalid shape for `samples`." \
                                 "Expected (?, %d), got %s)" % ( self._num_nodes, samples.shape))
        else:
            if m != 1:
                raise ValueError("Invalid shape for `samples`." \
                                 "Expected (?, %d), got %s)" % (1, samples.shape))
            
        shuffle = kwargs.get('shuffle', False)
        num_batches = kwargs.get('num_batches', 1)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)

        if shuffle:
            np.random.shuffle(samples)

        # Starts training
        return self._start_training(samples, num_batches, likelihood_thres, sess, feed_samples)


    def init_mpe_states(self):
        """
        Initialize Ops for MPE inference.
        """
        mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
        if self._num_nodes != 0:
            if not self._expanded:
                self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs, self._view_inputs)
            else:
                self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs, self._view_inputs)
        else:
            raise ValueError("Nothing to infer!")

        
    def evaluate(self, sess, *args, **kwargs):
        """
        Feeds inputs into the network and return the output of the network.

        sess (tf.Session): a session.

        *args:
            sample (numpy.ndarray): an (n+1,) numpy array as a sample, where n is the number of nodes,
                                    and +1 is the edge view distance variable's value at the last place.
                                    If only nodes, then shape is (n,). If only edges, then shape is (1,)
           <if expanded>
           sample_lh (numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                       so [?,c] is a likelihood for class c (float).
        """
        sample = np.array([args[0]], dtype=int)  # shape becomes (1, n+1)
        feed_dict = self._get_feed_dict(sample)
        if not self._expanded:
            likelihood_val = sess.run(self._train_likelihood, feed_dict=feed_dict)
        else:
            sample_lh = np.array([args[1].flatten()], dtype=float32)
            feed_dict[self._likelihood_inputs] = sample_lh
            likelihood_val = sess.run(self._train_likelihood, feed_dict=feed_dict)
        return likelihood_val        
        
    def marginal_inference(self, sess, query, query_lh=None, masked_only=False):
        """
        Performs marginal inference on node template SPN.

        `query` is an numpy array of shape (n+1,), where n is the number of nodes,
                and +1 is the edge view distance variable's value at the last place.
                If only nodes, then shape is (n,). If only edges, then shape is (1,)
        `query_lh` is an numpy array of shape (n,m), used to supply likelihood values per node per class.
        `masked_only` is True if only perform marginal inference on nodes that are
                      masked, i.e. their value in the `query` is -1. Useful when there's no likelihoods and
                      we are just inferring missing semantics.

        Returns: 
            A tuple of two elements:
              - First element: a numpy array of shape (n,m) where n is the number of nodes in the node
                template, and m is the number of classes. Each value represents the likelihood that the
                node is assigned to a class.
              - Second element: a numpy array of shape (d,) where d is the number of possible values for
                absolute view distance. If edge_pair for this model's template is 0, then this element
                will be an array of zeros.
        """
        # The only way to do marginal inference now is to iterate through all possible
        # classes per node.
        if query.shape[0] != self._num_nodes + self._num_edge_pair:
            raise ValueError("Query has wrong shape. Expect %d, Got %d"
                             % (self._num_nodes + self._num_edge_pair, query.shape[0]))
        marginals_nodes = np.zeros((self._num_nodes, CategoryManager.NUM_CATEGORIES))
        marginals_view = np.zeros((self._view_dist_input.num_vals,))
        if self._num_edge_pair != 0 and (query[-1] == -1 or not masked_only):
            # We want to infer view distance.
            for view_dist in range(self._view_dist_input.num_vals):
                query[-1] = view_dist
                marginals_view[view_dist] = float(self.evaluate(sess, query, query_lh)[0])
        for i in range(query[:self._num_nodes].shape[0]):
            if masked_only and query[i] != -1:
                continue
            orig = query[i]
            for val in range(CategoryManager.NUM_CATEGORIES):
                query[i] = val
                marginals_nodes[i, val] = float(self.evaluate(sess, query, query_lh)[0])
        return marginals_nodes, marginals_view
        

    def mpe_inference(self, sess, *args, **kwargs):
        """
        ASSUME self._num_nodes != 0. (Because otherwise, there's no point of doing MPE at the scale of a template)

        query (numpy.ndarray): an (n,) or (n+1,) numpy array where n is self._num_nodes.
                               if first dimension is of size n+1, then must follow self._num_edge_pair != 0.

        <if expanded>
        query_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                  so [?,c] is a likelihood for class c (float).

        """
        query = np.array([args[0]], dtype=int)
        if not self._expanded:
            if self._num_edge_pair == 0:
                mpe_state_val = sess.run(self._mpe_state, feed_dict={self._catg_inputs: query})
            else:
                mpe_state_val = sess.run(self._mpe_state, feed_dict={self._catg_inputs: query[:self._num_nodes-1],
                                                                     self._view_dist_input: query[self._num_nodes-1:]})
        else:
            query_lh = np.array([args[1].flatten()], dtype=float32)
            if self._num_edge_pair == 0:
                mpe_state_val = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: query,
                                                                     self._likelihood_inputs: query_lh})
            else:
                mpe_state_val = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: query[:self._num_nodes-1],
                                                                     self._likelihood_inputs: query_lh,
                                                                     self._view_dist_input: query[self._num_nodes-1:]})
        return mpe_state_val


    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        spn.JSONSaver(path, pretty=pretty).save(self._root, save_param_vals=True, sess = sess)

    
    def load(self, path, sess):
        """
        Loads the SPN structure and parameters. Replaces the existing structure
        of this SPN.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        loader = spn.JSONLoader(path)
        self._root = loader.load(load_param_vals=True, sess=sess)

        if self._num_nodes != 0:
            self._catg_inputs = loader.find_node(self.vn['CATG_IVS'])
            self._conc_inputs = loader.find_node(self.vn['CONC'])
        
            if self._expanded:
                self._likelihood_inputs = loader.find_node(self.vn['LH_CONT'])
                self._semantic_inputs = loader.find_node(self.vn['SEMAN_IVS'])
                
        if self._num_edge_pair != 0:
            self._view_dist_input = loader.find_node(self.vn['VIEW_IVS'])
            self._conc_inputs = loader.find_node(self.vn['CONC'])
# ---------- END EdgeRelationTemplateSpn ---------- #
