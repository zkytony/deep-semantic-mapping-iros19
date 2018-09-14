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
        self._dense_gen = spn.DenseSPNGeneratorLayerNodes(num_decomps=self._num_decomps, num_subsets=self._num_subsets,
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
        weight_init_value = spn.ValueType.RANDOM_UNIFORM(self._weight_init_min, self._weight_init_max)
        spn.generate_weights(self._root, init_value=weight_init_value, trainable=trainable)
        

    def init_weights_ops(self):
        print("Generating weight initialization Ops...")
        init_weights = spn.initialize_weights(self._root)
        self._initialize_weights = init_weights


    def init_learning_ops(self):
        print("Initializing learning Ops...")
        if self._learning_algorithm == spn.GDLearning:
            learning = spn.GDLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      learning_rate=self._learning_rate,
                                      learning_type=self._learning_type,
                                      learning_inference_type=self._learning_inference_type,
                                      use_unweighted=True)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.learn(optimizer=self._optimizer)
            
        elif self._learning_algorithm == spn.EMLearning:
            learning = spn.EMLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      additive_smoothing=self._additive_smoothing_var,
                                      use_unweighted=True,
                                      initial_accum_value=self._init_accum)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.accumulate_updates()
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
            print("Expanding...")
            
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

    def _compute_mle_loss(self, samples, func_feed_samples, dgsm_lh=None, batch_size=100):
        batch = 0
        likelihood_values = []
        stop = min(batch_size, len(samples))
        while stop < len(samples):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, len(samples))
            print("    BATCH", batch, "SAMPLES", start, stop)

            likelihood_val = func_feed_samples(self, samples, start, stop, dgsm_lh=dgsm_lh,
                                               ops=[self._avg_train_likelihood])
            likelihood_values.append(likelihood_val)
            batch += 1
        return -np.mean(likelihood_values)
    

    def _start_training(self, samples, batch_size, likelihood_thres,
                        sess, func_feed_samples, num_epochs=None, dgsm_lh=None,
                        shuffle=True, dgsm_lh_test=None, samples_test=None):
        """
        Helper for train() in subclasses. Weights should have been initialized.

        `samples` (numpy.ndarray) numpy array of shape (D, ?).
        `func_feed_samples` (function; start, stop) function that feeds samples into the network.
                            It runs the train_likelihood, avg_train_likelihood, and accumulate_updates
                            Ops.
        `lh` (numpy.ndarray) dgsm likelihoods
        """
        print("Resetting accumulators...")
        sess.run(self._reset_accumulators)

        batch_likelihoods = []  # record likelihoods within an epoch
        train_likelihoods = []  # record likelihoods for training samples
        test_likelihoods = []   # record likelihoods for testing samples

        prev_likelihood = 100   # previous likelihood
        likelihood = 0          # current likelihood
        epoch = 0
        batch = 0

        # Shuffle
        if shuffle:
            print("Shuffling...")
            p = np.random.permutation(len(samples))
            smaples = samples[p]
            if dgsm_lh is not None:
                dgsm_lh = dgsm_lh[p]

        print("Starts training. Maximum epochs: %s   Likelihood threshold: %.3f" % (num_epochs, likelihood_thres))
        while (num_epochs and epoch < num_epochs) \
              or (not num_epochs and (abs(prev_likelihood - likelihood) > likelihood_thres)):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, samples.shape[0])
            print("EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop, "  prev likelihood", prev_likelihood, "likelihood", likelihood)

            if self._learning_algorithm == spn.EMLearning:
                ads = max(np.exp(-epoch*self._smoothing_decay)*self._additive_smoothing,
                          self._min_additive_smoothing)
                sess.run(self._additive_smoothing_var.assign(ads))
                print("Smoothing: ", sess.run(self._additive_smoothing_var))

                train_likelihoods_arr, likelihood_train, _, = \
                    func_feed_samples(self, samples, start, stop, dgsm_lh=dgsm_lh,
                                      ops=[self._train_likelihood, self._avg_train_likelihood,
                                           self._learn_spn])
                sess.run(self._update_spn)

            batch_likelihoods.append(likelihood_train)
            batch += 1
            if stop >= samples.shape[0]:  # epoch finishes
                epoch += 1
                batch = 0

                # Shuffle
                if shuffle:
                    print("Shuffling...")
                    p = np.random.permutation(len(samples))
                    smaples = samples[p]
                    if dgsm_lh is not None:
                        dgsm_lh = dgsm_lh[p]

                if samples_test is not None:
                    print("Computing train, (test) likelihoods...")
                    likelihood_train = -np.mean(batch_likelihoods)
                    train_likelihoods.append(likelihood_train)
                    print("Train likelihood: %.3f  " % likelihood_train)

                    likelihood_test = self._compute_mle_loss(samples_test, func_feed_samples, dgsm_lh=dgsm_lh_test)
                    test_likelihoods.append(likelihood_test)
                    print("Test likelihood: %.3f  " % likelihood_test)

                prev_likelihood = likelihood
                likelihood = likelihood_train
                batch_likelihoods = []

        return train_likelihoods, test_likelihoods


    @staticmethod
    def dup_fun_up(inpt, *args,
                    conc=None, tmpl_num_vars=[0], tmpl_num_vals=[0], graph_num_vars=[0], labels=[[]], tspn=None):
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
                return spn.Sum(*args[2:], weights=args[0])
            elif isinstance(node, spn.ParSums):
                return spn.ParSums(*args[2:], weights=args[0], num_sums=tspn._num_mixtures)
            elif isinstance(node, spn.Product):
                return spn.Product(*args)
            elif isinstance(node, spn.PermProducts):
                return spn.PermProducts(*args)
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

        # Don't use the layered generator for now
        if self._num_nodes == 1:
            self._input_dist = spn.DenseSPNGenerator.InputDist.RAW
            self._dense_gen = spn.DenseSPNGenerator(num_decomps=self._num_decomps, num_subsets=self._num_subsets,
                                                    num_mixtures=self._num_mixtures, input_dist=self._input_dist,
                                                    num_input_mixtures=self._num_input_mixtures)

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
        def feed_samples(self, samples, start, stop, dgsm_lh=None, ops=[]):
            if dgsm_lh is None:
                return sess.run(ops, feed_dict={self._catg_inputs: samples[start:stop]})
                                
            else:
                return sess.run(ops,
                                feed_dict={self._semantic_inputs: np.full((stop-start, self._num_nodes),-1),
                                           self._likelihood_inputs: dgsm_lh[start:stop]})

        
        samples = args[0]
        D, n = samples.shape
        if n != self._num_nodes:
            raise ValueError("Invalid shape for `samples`." \
                             "Expected (?,%d), got %s)" % (self._num_nodes, samples.shape))
        
        shuffle = kwargs.get('shuffle', False)
        batch_size = kwargs.get('batch_size', 200)
        num_epochs = kwargs.get('num_epochs', None)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)
        dgsm_lh = kwargs.get('dgsm_lh', None)
        dgsm_lh_test = kwargs.get('dgsm_lh_test', None)
        samples_test = kwargs.get('samples_test', None)

        # Starts training
        return self._start_training(samples, batch_size, likelihood_thres,
                                    sess, feed_samples, num_epochs=num_epochs, dgsm_lh=dgsm_lh,
                                    shuffle=shuffle, samples_test=samples_test, dgsm_lh_test=dgsm_lh_test)
        
    
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



# class EdgeTemplateSpn(TemplateSpn):

#     """
#     Spn on top of an edge template. (The connectivity is not considered; it is
#     implicit in the training data)
#     """

#     def __init__(self, template, divisions=8, *args, **kwargs):
#         """
#         Initialize an EdgeTemplateSPN.

#         template (EdgeTemplate): a class of the modeled edge template.
#         divisions (int): number of views per place.

#         **kwargs:
#            seed (int): seed for the random generator. Set before generating
#                        the structure.
#         """
#         super().__init__(template, *args, **kwargs)

#         self._divisions = divisions
#         self._template = template
#         self._num_nodes = self._template.num_edges() * 2  # number of nodes where variables are derived.

#         self._init_struct(rnd=self._rnd, seed=self._seed)


#     @property
#     def template(self):
#         return self._template

#     @property
#     def divisions(self):
#         return self._divisions

#     @property
#     def root(self):
#         return self._root

#     @property
#     def vn(self):
#         return {
#             'CATG_IVS':"Catg_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
#             'VIEW_IVS':"View_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
#             'CONC': "Conc_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
#             'SEMAN_IVS': "Exp_Catg_%s_%d" % (self.__class__.__name__, self._template.num_edges()),
#             'LH_CONT': "Exp_Lh_%s_%d" % (self.__class__.__name__, self._template.num_edges())
#         }

#     def _init_struct(self, rnd=None, seed=None):
#         """
#         Initialize the structure for training.  (private method)

#         rnd (Random): instance of a random number generator used for
#                       dense generator. 
#         """
#         # An input to spn goes through a concat node before reaching the network. This makes
#         # it convenient to expand the network downwards, by swapping the inputs to the concat
#         # nodes. 

#         # Inputs to the spn. `catg_inputs` are inputs for semantic categories; `view_inputs`
#         # are inputs for view numbers.
#         self._catg_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = CategoryManager.NUM_CATEGORIES,
#                                     name = self.vn['CATG_IVS'])
#         self._view_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = self._divisions,
#                                     name = self.vn['VIEW_IVS'])

#         # Concat nodes, used to connect the category inputs to the spn. 
#         self._conc_inputs = spn.Concat(spn.Input.as_input(self._catg_inputs),
#                                        name = self.vn['CONC'])

#         # Generate structure, weights, and generate learning operations.
#         print("Generating SPN structure...")
#         if seed is not None:
#             print("[Using seed %d]" % seed)
#             rnd = random.Random(seed)
#         self._root = self._dense_gen.generate(self._conc_inputs, self._view_inputs, rnd=rnd)
    

#     def train(self, sess, *args, **kwargs):
#         """
#         Train the SPN. Weights should have been initialized.

#         sess (tf.Session): Tensorflow session object
#         *args:
#           samples (numpy.ndarray): A numpy array of size (D,2,n) where D is the
#                                    number of data samples, and n is the number
#                                    of semantic variables in template modeled by
#                                    this spn. (e.g. in EdgePairTemplate, n = 4)
#                                    (note: n = self.num_nodes)

#         **kwargs:
#           shuffle (bool): shuffles `samples` before training. Default: False.
#           num_batches (int): number of batches to split the training data into.
#                              Default: 1
#           likelihood_thres (float): threshold of likelihood difference between
#                                     interations to stop training. Default: 0.05
#         """

#         def feed_samples(self, samples, start, stop):
#             return sess.run([self._train_likelihood, self._avg_train_likelihood,
#                              self._learn_spn],
#                             feed_dict={self._catg_inputs: samples[start:stop, 0],
#                                        self._view_inputs: samples[start:stop, 1]})
        
#         samples = args[0]
#         D, m, n = samples.shape
#         if n != self._num_nodes or m != 2:
#             raise ValueError("Invalid shape for `samples`." \
#                              "Expected (?, %d, %d), got %s)" % (2, self._num_nodes, samples.shape))
            
#         shuffle = kwargs.get('shuffle', False)
#         num_batches = kwargs.get('num_batches', 1)
#         likelihood_thres = kwargs.get('likelihood_thres', 0.05)

#         if shuffle:
#             p = np.random.permutation(len(samples))
#             smaples = samples[p]
#             if lh is not None:
#                 lh = lh[p]

#         # Starts training
#         return self._start_training(samples, num_batches, likelihood_thres, sess, feed_samples, lh=lh)


#     def evaluate(self, sess, *args, **kwargs):
#         """
#         Feeds inputs into the network and return the output of the network. Returns the
#         output of the network

#         sess (tf.Session): a session.

#         *args:
#           sample (numpy.ndarray): an (2*n,) numpy array as a sample. n = self.num_nodes

#           <if expanded>
#           sample_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
#                                       so [?,c] is a likelihood for class c (float).
#         """

#         if not self._expanded:
#             sample = args[0].reshape(2,-1)
#             likelihood_val = sess.run(self._train_likelihood, feed_dict={self._catg_inputs: np.array([sample[0,]], dtype=int),
#                                                                          self._view_inputs: np.array([sample[1,]], dtype=int)})
#         else:
#             sample = args[0].reshape(2,-1)
#             sample_lh = np.array([args[1].flatten()], dtype=float32)
#             likelihood_val = sess.run(self._train_likelihood, feed_dict={self._semantic_inputs: np.array([sample[0,]], dtype=int),
#                                                                          self._likelihood_inputs: sample_lh,
#                                                                          self._view_inputs: np.array([sample[1,]], dtype=int)})
            
#         return likelihood_val  # TODO: likelihood_val is now a numpy array. Should be a float.


#     def init_mpe_states(self):
#         """
#         Initialize Ops for MPE inference.
#         """
#         mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
#         if not self._expanded:
#             self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs, self._view_inputs)
#         else:
#             self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs, self._view_inputs)

            
#     def marginal_inference(self, sess, *args, **kwargs):
#         """
#         Performs marginal inference.
#         """
#         raise NotImplementedError


#     def mpe_inference(self, sess, *args, **kwargs):
#         """
#         Feeds inputs with some '-1' and infer their values. Returns the inferred mpe value
#         for the query state.

#         sess (tf.Session): a session.

#         *args:
#           query (numpy.ndarray): an (2*n,) numpy array as a query. n = self.num_nodes

#           <if expanded>
#           query_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
#                                      so [?,c] is a likelihood for class c (float).
#         """
#         if not self._expanded:
#             query = args[0].reshape(2,-1)
#             mpe_state_val = sess.run(self._mpe_state, feed_dict={self._catg_inputs: np.array([query[0,]], dtype=int),
#                                                                  self._view_inputs: np.array([query[1,]], dtype=int)})
#         else:
#             query = args[0].reshape(2,-1)
#             query_lh = np.array([args[1].flatten()], dtype=float32)
#             mpe_state_val = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: np.array([query[0,]], dtype=int),
#                                                                  self._likelihood_inputs: query_lh,
#                                                                  self._view_inputs: np.array([query[1,]], dtype=int)})
#         return mpe_state_val


#     def save(self, path, sess, pretty=True):
#         """
#         Saves the SPN structure and parameters.

#         path (str): save path.
#         sess (tf.Session): session object, required to save parameters.
#         """
#         spn.JSONSaver(path, pretty=pretty).save(self._root, save_param_vals=True, sess = sess)

    
#     def load(self, path, sess):
#         """
#         Loads the SPN structure and parameters. Replaces the existing structure
#         of this SPN.

#         path(str): path to load spn.
#         sess (tf.Session): session object, required to load parameters into current session.
#         """
#         loader = spn.JSONLoader(path)
#         self._root = loader.load(load_param_vals=True, sess=sess)

#         self._catg_inputs = loader.find_node(self.vn['CATG_IVS'])
#         self._view_inputs = loader.find_node(self.vn['VIEW_IVS'])
#         self._conc_inputs = loader.find_node(self.vn['CONC'])
        
#         if self._expanded:
#             self._likelihood_inputs = loader.find_node(self.vn['LH_CONT'])
#             self._semantic_inputs = loader.find_node(self.vn['SEMAN_IVS'])
# # -- END EdgeTemplateSpn -- #




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
        # If there are 8 divisions, there could be 4 different values (1, 2, 3, 4) for the
        # absolute distance between two views. These values correspond to 0, 1, 2, 3 of the
        # actual value of the view distance IVs node.
        self._num_view_dists = self._divisions // 2

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
            'LH_CONT': "Exp_Lh_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            'EXP_VIEW_IVS':"Exp_View_%s_%d_%d" % (self.__class__.__name__, self._num_nodes, self._num_edge_pair),
            
        }

    def _get_all_catg_inputs(self):
        return spn.Input(self._conc_inputs, list(map(int, np.arange(self._num_nodes * CategoryManager.NUM_CATEGORIES, dtype=int))))

    def _get_all_view_inputs(self):
        return spn.Input(self._conc_inputs, list(map(int, np.arange(self._num_nodes * CategoryManager.NUM_CATEGORIES,
                                                                    self._num_nodes * CategoryManager.NUM_CATEGORIES + self._num_view_di, dtype=int))))
    
    def _get_view_input(self, view_dist):
        return spn.Input(self._conc_inputs, self._num_nodes * CategoryManager.NUM_CATEGORIES + view_dist)

    
    def _create_variables(self):
        """
        Create IVs and connect them into a Concat node.
        """
        if self._num_nodes != 0:
            self._catg_inputs = spn.IVs(num_vars = self._num_nodes, num_vals = CategoryManager.NUM_CATEGORIES,
                                        name = self.vn['CATG_IVS'])
            self._conc_inputs = spn.Concat(spn.Input.as_input(self._catg_inputs), name=self.vn['CONC'])
        if self._num_edge_pair != 0:
            self._view_dist_input = spn.IVs(num_vars = 1, num_vals = self._num_view_dists, name=self.vn['VIEW_IVS'])  ## WARNING Assuming absolute value distance
            if self._conc_inputs is not None:
                self._conc_inputs.add_inputs(self._view_dist_input)
            else:
                self._conc_inputs = spn.Concat(spn.Input.as_input(self._view_dist_input), name=self.vn['CONC'])
    
    def _init_struct(self, rnd=None, seed=None):
        """
        The structure of view template SPN:
        A root sum node has K children where K equals to the number of possible view difference values.
        Each child is a sub-SPN rooted by a product node that has two children, one being a template-SPN
        for N-node template constructed upon the IVs for categories, and another is an indicator variable
        that corresponds to a particular setting of the view distance.
        Still, the category variables and view distances are IVs that go through a single Conc node, in order
        for the duplication to work.
        Note that, for simple view templates, i.e. (1,0), (0,1) or (1,1), there is no need for anything complex
        so the resulting structure is dense generated with configurations for a simple structure.
        """
        # Create variables
        self._create_variables()
        
        if self._template.to_tuple() == (0, 1) \
           or self._template.to_tuple() == (1, 0) \
           or self._template.to_tuple() == (1, 1):
            # Simple structure by dense generation; Recreate the dense generator with simple parameters
            self._input_dist = spn.DenseSPNGenerator.InputDist.RAW
            self._dense_gen = spn.DenseSPNGenerator(num_decomps=1, num_subsets=2,
                                                    num_mixtures=2, input_dist=self._input_dist,
                                                    num_input_mixtures=self._num_input_mixtures)
            self._root = self._dense_gen.generate(self._conc_inputs, rnd=rnd)

        else:
            if self._template.to_tuple() != (3, 1):
                raise ValueError("Oops. Currently the only supported view templates are (1,0), (0,1), (1,1), and (3,1)." \
                                 "%s is not supported" % self._template)
            # For each view distance value, create a structure rooted by a product node with one child being
            # the indicator for that view distance value, and another child being the root of a 3-node template SPN.
            sub_roots = []
            for i in range(self._num_view_dists):
                tmpl3_root = self._dense_gen.generate(self._get_all_catg_inputs(), rnd=rnd)
                view_input = self._get_view_input(i)
                prod = spn.Product(tmpl3_root, view_input, name="View_%d_SubSPN_Product" % i)
                sub_roots.append(prod)
            self._root = spn.Sum(*sub_roots)
            

    @property
    def root(self):
        return self._root


    def _get_feed_dict(self, samples, start=0, stop=None, dgsm_lh=None):
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
            if dgsm_lh is None:
                feed_dict={catg_inputs: samples[start:stop, :self._num_nodes],
                           self._view_dist_input: samples[start:stop, self.num_nodes:]}
            else:
                feed_dict={catg_inputs: np.full((stop-start, self._num_nodes),-1),#samples[start:stop, :self._num_nodes],
                           self._view_dist_input: samples[start:stop, self.num_nodes:],
                           self._likelihood_inputs: dgsm_lh[start:stop]}
        elif self._num_nodes != 0:
            if dgsm_lh is None:
                feed_dict={catg_inputs: samples[start:stop]}
            else:
                feed_dict={catg_inputs: np.full((stop-start, self._num_nodes),-1),#samples[start:stop, :self._num_nodes],
                           self._likelihood_inputs: dgsm_lh[start:stop]}
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
          batch_size (int): batch size during training
          likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
        """
        def feed_samples(self, samples, start, stop, dgsm_lh=None, ops=[]):
            feed_dict = self._get_feed_dict(samples, start=start, stop=stop, dgsm_lh=dgsm_lh)
            return sess.run(ops,
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
        batch_size = kwargs.get('batch_size', 200)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)
        num_epochs = kwargs.get('num_epochs', None)
        dgsm_lh = kwargs.get('dgsm_lh', None)
        dgsm_lh_test = kwargs.get('dgsm_lh_test', None)
        samples_test = kwargs.get('samples_test', None)

        # Starts training
        return self._start_training(samples, batch_size, likelihood_thres, sess, feed_samples,
                                    num_epochs=num_epochs, dgsm_lh=dgsm_lh,
                                    samples_test=samples_test, dgsm_lh_test=dgsm_lh_test)

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

        
    def expand(self):

        if not self._expanded:
            print("Expanding...")

            if self._num_nodes != 0:
                super().expand()

            if self._num_edge_pair != 0:
                # Note: Because current libspn's compute_graph_up function does not handle Concat nodes
                # correctly, directly adding self._view_dist_input to _conc_input will lead to error in
                # weight generation. Instead, we do a trick similar to that is done for the node
                # expansion - replace each IV with a product node, but with only one child.
                self._view_dist_input = spn.IVs(num_vars=1, num_vals=self._num_view_dists, name=self.vn['EXP_VIEW_IVS'])
                prods = []
                for i in range(self._num_view_dists):
                    prod = spn.Product(
                        (self._view_dist_input, [i])
                    )
                    prods.append(prod)
                if self._num_nodes != 0:
                    self._conc_inputs.add_inputs(*map(spn.Input.as_input, prods))
                else:
                    self._conc_inputs.set_inputs(*map(spn.Input.as_input, prods))
                self._expanded = True

        
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
