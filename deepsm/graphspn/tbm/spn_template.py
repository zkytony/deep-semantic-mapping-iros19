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
    def _dup_fun_up(inpt, *args, conc=None, tmpl_num_vars=[0], tmpl_num_vals=[0], labels=[[]]):
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
            elif isinstance(node, spn.Product):
                return spn.Product(*args)
            elif isinstance(node, spn.Concat):
                # For each index in indices, find its corresponding variable index by dividing
                # it by tmpl_num_vals.
                ranges = []  # stores the start (inclusive) index of the range of indices taken by a type of iv
                start = 0
                for i in range(len(tmpl_num_vars)):
                    ranges.append(start)
                    start += tmpl_num_vars[i]*tmpl_num_vals[i]
                big_indices = []
                for indx in indices:
                    iv_type = -1
                    for i, start in enumerate(ranges):
                        if indx < start + tmpl_num_vars[i]*tmpl_num_vals[i]:
                            iv_type = i
                            break
                    if iv_type == -1:
                        raise ValueError("Oops. Something wrong. Index out of range.")
                    # Find corresponding variable index, and offset (i.e. which indicator of that variable)
                    varidx = indx // tmpl_num_vals[iv_type]
                    offset = indx - varidx * tmpl_num_vals[iv_type]
                    varlabel = labels[iv_type][varidx] # THIS IS the actual position of the variable's inputs in the big Concat.
                    # ASSUMING continous block of the (big) concat node, find the index in it.
                    big_indices.append(ranges[iv_type] + varlabel * tmpl_num_vals[iv_type] + offset)
                return spn.Input(conc, big_indices)
        elif isinstance(node, spn.Weights):
            return node
        else:
            raise ValueError("We don't intend to deal with IVs here. Please remove them from the concat.")
    # END fun_up
    

    @classmethod    
    def duplicate_template_spns(cls, ispn, tspns, template, supergraph, nodes_covered):
        """
        Convenient method for copying template spns. Modified `nodes_covered`.
        """    
        roots = []
        if ispn._template_mode == NodeTemplate.code():
            __i = 0
            for compound_nid in supergraph.nodes:
                nids = supergraph.nodes[compound_nid].to_place_id_list()
                ispn._template_nodes_map[ispn._id_incr] = nids

                # Make the right indices (with respect to the full conc node)
                labels = []
                for nid in nids:
                    # The ivs is arranged like: [...(num_catgs)] * num_nodes
                    label = ispn._node_label_map[nid]
                    num_catg = CategoryManager.NUM_CATEGORIES
                    nodes_covered.add(nid)
                    labels.append(label)

                print("Duplicating... %d" % (__i+1))
                copied_tspn_root = mod_compute_graph_up(tspns[template.__name__][0],
                                                        TemplateSpn._dup_fun_up, tmpl_num_vars=[len(nids)], tmpl_num_vals=[CategoryManager.NUM_CATEGORIES],
                                                        conc=ispn._conc_inputs,
                                                        labels=[labels])
                assert copied_tspn_root.is_valid()
                roots.append(copied_tspn_root)
                __i+=1
                ispn._id_incr += 1
        return roots

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
            self._view_dist_input = spn.IVs(num_vars = 1, num_vals = self._divisions // 2)  ## WARNING Assuming absolute value distance
            
        # Generate structure, weights, and generate learning operations.
        print("Generating SPN structure...")
        if seed is not None:
            print("[Using seed %d]" % seed)
            rnd = random.Random(seed)
        if self._num_nodes != 0 and self._num_edge_pair != 0:
            self._root = self._dense_gen.generate(self._conc_inputs, self._view_dist_input, rnd=rnd)
        elif self._num_nodes != 0:
            self._root = self._dense_gen.generate(self._conc_inputs, rnd=rnd)
        else:
            self._root = self._dense_gen.generate(self._view_dist_input, rnd=rnd)

    @property
    def root(self):
        return self._root

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
            if self._num_nodes != 0 and self._num_edge_pair != 0:
                return sess.run([self._train_likelihood, self._avg_train_likelihood,
                                 self._accumulate_updates],
                                feed_dict={self._catg_inputs: samples[start:stop, :self._num_nodes],
                                           self._view_dist_input: samples[start:stop, self.num_nodes:]})
            elif self._num_nodes != 0:
                return sess.run([self._train_likelihood, self._avg_train_likelihood,
                                 self._accumulate_updates],
                                feed_dict={self._catg_inputs: samples[start:stop]})
            else:
                return sess.run([self._train_likelihood, self._avg_train_likelihood,
                                 self._accumulate_updates],
                                feed_dict={self._view_dist_inputs: samples[start:stop]})

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
        raise NotImplementedError
        # """
        # Feeds inputs into the network and return the output of the network. Returns the
        # output of the network

        # sess (tf.Session): a session.

        # *args:
        #   sample (numpy.ndarray): an (n,) numpy array as a sample.
          
        #   <if expanded>
        #   sample_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
        #                               so [?,c] is a likelihood for class c (float).
        # """
        # # To feed to the network, we need to reshape the samples.
        # sample = np.array([args[0]], dtype=int)
        
        # if not self._expanded:
        #     likelihood_val = sess.run(self._train_likelihood, feed_dict={self._catg_inputs: sample})
        # else:
        #     sample_lh = np.array([args[1].flatten()], dtype=float32)
        #     likelihood_val = sess.run(self._train_likelihood, feed_dict={self._semantic_inputs: sample,
        #                                                                  self._likelihood_inputs: sample_lh})
            
        # return likelihood_val  # TODO: likelihood_val is now a numpy array. Should be a float.

        
        
    def marginal_inference(self, sess, *args, **kwargs):
        """
        Performs marginal inference.
        """
        raise NotImplementedError
        

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
            self._view_dist_input = loader.find_node(self.vn('VIEW_IVS'))
# ---------- END EdgeRelationTemplateSpn ---------- #



class InstanceSpn(SpnModel):
    """
    Given a topological map instance, create an spn from given template spns for this instance.
    """
    def __init__(self, topo_map, sess, *spns, **kwargs):
        """
        Initializes the structure and learning operations of an instance spn.

        Note that all ivs variables are ordered by sorted(topo_map.nodes). You can retrieve the
        value of the MPE result of a node by following this order.

        topo_map (TopologicalMap): a topo map instance.
        num_partitions (int): number of partitions (children for root node)
        spns (list): a list of tuples (TemplateSpn, Template). Note that assume
                     all templates are either node templates or edge templates.
        sess (tf.Session): a session that contains all weights.

        **kwargs:
           spn_paths (dict): a dictionary from Template to path. For loading the spn for Template at
                             path.
           num_partitions (int) number of child for the root sum node.
           seq_id (str): sequence id for the given topo map instance. Used as identified when
                         saving the instance spn. Default: "default_1"
           no_init (bool): True if not initializing structure; user might want to load structure
                           from a file.
           visualize_partitions_dirpath (str): Path to save the visualization of partitions on each child.
                                            Default is None, meaning no visualization is saved.
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
           db_name (str): Database name. Required if visualization_partitions_dirpath is not None.

           If template is EdgeTemplate, then:
             divisions (int) number of views per place
        """
        super().__init__(**kwargs)        
        self._seq_id = kwargs.get("seq_id", "default_1")
        # sort spns by template size.
        self._spns = sorted(spns, key=lambda x: x[1].size(), reverse=True)
        self._template_mode = int(issubclass(self._spns[0][1], EdgeTemplate))
        self._topo_map = topo_map
        self._id_incr = 0
        num_partitions = kwargs.get('num_partitions', 1)
        self._init_struct(sess, spn_paths=kwargs.get('spn_paths'), divisions=kwargs.get('divisions', -1),
                          num_partitions=kwargs.get('num_partitions', 1),
                          visualize_partitions_dirpath=kwargs.get('visualize_partitions_dirpath', None),
                          db_name=kwargs.get('db_name', None), extra_partition_multiplyer=kwargs.get('extra_partition_multiplyer', 1))
        self._expanded = False

        if self._template_mode == 0:
            self._inputs_size = len(self._topo_map.nodes)
        else:
            self._inputs_size = len(self._topo_map.nodes)*2


    @property
    def vn(self):
        return {
            'CATG_IVS': "Catg_%d_%s" % (self._template_mode, self._seq_id),
            'VIEW_IVS':"View_%d_%s" % (self._template_mode, self._seq_id),
            'CONC': "Conc_%d_%s" % (self._template_mode, self._seq_id),
            'SEMAN_IVS': "Exp_Catg_%d_%s" % (self._template_mode, self._seq_id),
            'LH_CONT': "Exp_Lh_%d_%s" % (self._template_mode, self._seq_id)
        }

    @property
    def root(self):
        return self._root


    def _init_struct(self, sess, spn_paths=None, divisions=-1, num_partitions=1,
                     visualize_partitions_dirpath=None, db_name=None, extra_partition_multiplyer=1):
        """
        Initialize the structure for training. (private method)

        sess: (tf.Session): a session that contains all weights.

        **kwargs:
           num_partitions (int): number of partitions (children for root node)
           If template is EdgeTemplate, then:
             divisions (int) number of views per place
           spn_paths (dict): a dictionary from Template to path. For loading the spn for Template at
                             path.
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
        """
            
        # Create vars and maps
        if self._template_mode == 0:  ## NodeTemplate
            self._catg_inputs = spn.IVs(num_vars=len(self._topo_map.nodes), num_vals=CategoryManager.NUM_CATEGORIES)
            self._conc_inputs = spn.Concat(self._catg_inputs)
            self._template_nodes_map = {}  # map from template id to list of node lds
            self._node_label_map = {}  # key: node id. Value: a number (0~num_nodes-1)
            self._label_node_map = {}  # key: a number (0~num_nodes-1). Value: node id
            _i = 0
            for nid in self._topo_map.nodes:
                self._node_label_map[nid] = _i
                self._label_node_map[_i] = nid
                _i += 1

        """Try partition the graph `extra_partition_multiplyer` times more than what is asked for. Then pick the top `num_partitions` with the highest
        coverage of the main template."""
        print("Partitioning the graph... (Selecting %d from %d attempts)" % (num_partitions, extra_partition_multiplyer*num_partitions))
        partitioned_results = {}
        main_template = self._spns[0][1]
        for i in range(extra_partition_multiplyer*num_partitions):
            """Note: here, we only partition with the main template. The results (i.e. supergraph, unused graph) are stored
            and will be used later. """
            supergraph, unused_graph = self._topo_map.partition(main_template, get_unused=True)
            if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                coverage = len(supergraph.nodes)*main_template.size() / len(self._topo_map.nodes)
                partitioned_results[(i, coverage)] = (supergraph, unused_graph)
        used_partitions = []
        used_coverages = set({})
        _i = 0
        for i, coverage in sorted(partitioned_results, reverse=True, key=lambda x:x[1]):
            used_partitions.append(partitioned_results[i, coverage])
            sys.stdout.write("%.3f  " % coverage)
            _i += 1
            if len(used_partitions) >= num_partitions:
                break
        sys.stdout.write("\n")
        """Delete unused partitions"""
        for coverage in list(partitioned_results.keys()):
            if coverage not in used_coverages:
                del partitioned_results[coverage]

        """Building instance spn"""
        print("Building instance spn...")
        pspns = []
        tspns = {}
        for template_spn, template in self._spns:
            tspns[template.__name__] = (template_spn.root, template_spn._catg_inputs, template_spn._conc_inputs)

        """Import if visualize partition is requested"""
        if visualize_partitions_dirpath:
            print("Importing matplotlib...")
            import matplotlib
            matplotlib.use('Agg')
            from pylab import rcParams
            import matplotlib.pyplot as plt
            rcParams['figure.figsize'] = 22, 14
            # Now, we create a cold database manager used to draw the map
            coldmgr = ColdDatabaseManager(db_name, COLD_ROOT, GROUNDTRUTH_ROOT)
            

        """Making an SPN"""
        if self._template_mode == NodeTemplate.code():  ## NodeTemplate
            """Now, partition the graph, copy structure, and connect self._catg_inputs appropriately to the network."""
            # Main template partition
            _k = 0
            for supergraph, unused_graph in used_partitions:
                print("Partition %d" % (_k+1))
                nodes_covered = set({})
                template_spn_roots = []
                main_template_spn = self._spns[0][0]
                print("Will duplicate %s %d times." % (main_template.__name__, len(supergraph.nodes)))
                template_spn_roots.extend(TemplateSpn.duplicate_template_spns(self, tspns, main_template, supergraph, nodes_covered))
                
                ## TEST CODE: COMMENT OUT WHEN ACTUALLY RUNNING
                # original_tspn_root = tspns[main_template.__name__][0]
                # duplicated_tspn_root = template_spn_roots[-1]
                # original_tspn_weights = sess.run(original_tspn_root.weights.node.get_value())
                # duplicated_tspn_weights = sess.run(duplicated_tspn_root.weights.node.get_value())
                # print(original_tspn_weights)
                # print(duplicated_tspn_weights)
                # print(original_tspn_weights == duplicated_tspn_weights)
                # import pdb; pdb.set_trace()
                
                tmp_graph = unused_graph
                """If visualize"""
                if visualize_partitions_dirpath:
                    ctype = 2
                    node_ids = []
                    for snid in supergraph.nodes:
                        node_ids.append(supergraph.nodes[snid].to_place_id_list())
                    img = self._topo_map.visualize_partition(plt.gca(), node_ids,
                                                             coldmgr.groundtruth_file(self._seq_id.split("_")[0], 'map.yaml'), ctype=ctype)
                    ctype += 1
                """Further partition the graph with remaining templates."""
                for template_spn, template in self._spns[1:]:  # skip main template
                    supergraph_2nd, unused_graph_2nd = tmp_graph.partition(template, get_unused=True)
                    print("Will duplicate %s %d times." % (template.__name__, len(supergraph_2nd.nodes)))
                    template_spn_roots.extend(TemplateSpn.duplicate_template_spns(self, tspns, template, supergraph_2nd, nodes_covered))
                    
                    ## TEST CODE: COMMENT OUT WHEN ACTUALLY RUNNING
                    # original_tspn_root = tspns[template.__name__][0]
                    # duplicated_tspn_root = template_spn_roots[-1]
                    # original_tspn_weights = sess.run(original_tspn_root.weights.node.get_value())
                    # duplicated_tspn_weights = sess.run(duplicated_tspn_root.weights.node.get_value())
                    # print(original_tspn_weights)
                    # print(duplicated_tspn_weights)
                    # print(original_tspn_weights == duplicated_tspn_weights)
                    # import pdb; pdb.set_trace()

                    tmp_graph = unused_graph_2nd
                    """If visualize"""
                    if visualize_partitions_dirpath:
                        # img should have been created from above.
                        node_ids = []
                        for snid in supergraph_2nd.nodes:
                            node_ids.append(supergraph_2nd.nodes[snid].to_place_id_list())
                        img = self._topo_map.visualize_partition(plt.gca(), node_ids,
                                                                 coldmgr.groundtruth_file(self._seq_id.split("_")[0], 'map.yaml'), ctype=ctype, img=img)
                        ctype += 1
                """If visualize. Save."""
                if visualize_partitions_dirpath:
                    plt.savefig(os.path.join(visualize_partitions_dirpath, "partition-%d.png" % (_k+1)))
                    plt.clf()
                    print("Visualized partition %d" % (_k+1))
                        
                assert nodes_covered == self._topo_map.nodes.keys()

                p = spn.Product(*template_spn_roots)
                assert p.is_valid()
                pspns.append(p) # add spn for one partition
                _k += 1  # increment partition counter
        # Sum up all
        self._root = spn.Sum(*pspns)
        assert self._root.is_valid()
        self._root.generate_weights(trainable=True)
        # initialize ONLY the weights node for the root
        sess.run(self._root.weights.node.initialize())


    def _inputs_list(self, input_type):
        """
        input_type can be 'catg' or 'conc', or 'view' (edge only)
        """
        inputs = [self._template_iv_map[tid][input_type] for tid in sorted(self._template_iv_map)]
        return inputs
        

    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        print("Initializing learning Ops...")
        learning = spn.EMLearning(self._root, log=True, value_inference_type = self._value_inference_type,
                                  additive_smoothing = self._additive_smoothing_var)
        self._reset_accumulators = learning.reset_accumulators()
        self._accumulate_updates = learning.accumulate_updates()
        self._update_spn = learning.update_spn()
        self._train_likelihood = learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)

        if not no_mpe:
            print("Initializing MPE Ops...")
            mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
            if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                if not self._expanded:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs)
                else:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs)

        # else:
        #     if not self._expanded:
        #         self._mpe_state = mpe_state_gen.get_state(self._root, *(self._inputs_list('catg')))# + self._inputs_list('view')))
        #     else:
        #         self._mpe_state = mpe_state_gen.get_state(self._root, *(self._inputs_list('semantic')))# + self._inputs_list('view')))



    def marginal_inference(self, sess, query_nids, query, query_lh=None):
        """
        Computes marginal distribution of queried variables.

        Now, only uses the DUMMY method. Iterate over all possible assignments of
        all inquired variables, evaluate the network and use the result as the likelihood
        for that assignment. Note that the marginals are in log space.

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.

        *args
          If NodeTemplate:
            query (dict): A dictionary mapping from node id to its category number
                           ASSUME that the inquired nodes have already been assigned to '-1'.
          else (EdgeTemplate):
            query (dict): A dictionary mapping from edge id to a tuple of 4 values:
                           (
                             node1 category number,
                             node2 category number,
                             node1 view number,
                             node2 view number,
                           )
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
            else (EdgeTemplate):
              query_lh (dict): A dictionary mapping from edge id to a tuple:
                          (
                            (m likelihoods for node1),
                            (m likelihoods for node2)
                          ), where m is the NUM_CATEGORIES

        Returns:
           a dictionary. Key: node id (corresponds to a node id in query_nids).
                         Value: log-space marginal probability distribution of this node.
        """
        marginals = {}
        for nid in query_nids:
            orig = query[nid]
            marginals[nid] = []
            for val in range(CategoryManager.NUM_CATEGORIES):
                query[nid] = val
                marginals[nid].append(self.evaluate(sess, query, sample_lh=query_lh))
                query[nid] = orig
        return marginals
        

                
    def evaluate(self, sess, sample, sample_lh=None):
        """
        sess (tf.Session): a session.

        *args
          If NodeTemplate:
            sample (dict): A dictionary mapping from node id to its category number
          else (EdgeTemplate):
            sample (dict): A dictionary mapping from edge id to a tuple of 4 values:
                           (
                             node1 category number,
                             node2 category number,
                             node1 view number,
                             node2 view number,
                           )
          If expanded (additional parameters)
            if NodeTemplate
              sample_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
            else (EdgeTemplate):
              sample_lh (dict): A dictionary mapping from edge id to a tuple:
                          (
                            (m likelihoods for node1),
                            (m likelihoods for node2)
                          ), where m is the NUM_CATEGORIES
        """
        if self._template_mode == NodeTemplate.code(): ## NodeTemplate
            ivs_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                ivs_assignment.append(sample[nid])
                    
            if not self._expanded:
                lh_val = sess.run(self._train_likelihood, feed_dict={self._catg_inputs: np.array([ivs_assignment], dtype=int)})[0]
            else:
                # expanded. So get likelihoods
                lh_assignment = []
                for l in range(len(self._label_node_map)):
                    nid = self._label_node_map[l]
                    lh_assignment.extend(list(sample_lh[nid]))
                lh_val = sess.run(self._train_likelihood, feed_dict={self._semantic_inputs: np.array([ivs_assignment], dtype=int),
                                                                     self._likelihood_inputs: np.array([lh_assignment])})[0]
            return lh_val
        
        # else:
        #     feed_dict = self._create_feed_dict_for_edge_templates(sample, sample_lh=sample_lh)
        #     lh_val = sess.run(self._train_likelihood, feed_dict=feed_dict)[0]
        #     return lh_val


    
    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.

        *args
          If NodeTemplate:
            query (dict): A dictionary mapping from node id to its category number
          else (EdgeTemplate):
            query (dict): A dictionary mapping from edge id to a tuple of 4 values:
                           (
                             node1 category number,
                             node2 category number,
                             node1 view number,
                             node2 view number,
                           )
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
            else (EdgeTemplate):
              query_lh (dict): A dictionary mapping from edge id to a tuple:
                          (
                            (m likelihoods for node1),
                            (m likelihoods for node2)
                          ), where m is the NUM_CATEGORIES


        Returns:
         
           If NodeTemplate:
              returns a category map (from id to category number)
           If EdgeTemplate:
              returns a tuple.
                  1st element: category map (id to category number)
                  2nd element: a dictionary, from node id to {'label': ..., 'counts': {'DW': ..., ...}}

        """

        if self._template_mode == NodeTemplate.code(): ## NodeTemplate
            ivs_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                ivs_assignment.append(query[nid])
                    
            if not self._expanded:
                result = sess.run(self._mpe_state, feed_dict={self._catg_inputs: np.array([ivs_assignment], dtype=int)})[0]
            else:
                # expanded. So get likelihoods
                lh_assignment = []
                for l in range(len(self._label_node_map)):
                    nid = self._label_node_map[l]
                    lh_assignment.extend(list(query_lh[nid]))
                result = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: np.array([ivs_assignment], dtype=int),
                                                              self._likelihood_inputs: np.array([lh_assignment])})[0]

            catg_map = {}
            for i in range(result.shape[1]):
                nid = self._label_node_map[i]
                catg_map[nid] = result[0][i]
            
            return catg_map


    def expand(self):
        """
        Custom method.

        Replaces the IVs inputs with a product node that has two children: a continuous
        input for likelihood, and a discrete input for semantics category.

        Do nothing if already expanded.
        """
        
        if not self._expanded:

            print("Expanding...")

            num_vars = len(self._topo_map.nodes)
            self._semantic_inputs = spn.IVs(num_vars=num_vars, num_vals=CategoryManager.NUM_CATEGORIES)
            self._likelihood_inputs = spn.RawInput(num_vars=num_vars*CategoryManager.NUM_CATEGORIES)

            prods = []
            for i in range(num_vars):
                for j in range(CategoryManager.NUM_CATEGORIES):
                    prod = spn.Product(
                        (self._likelihood_inputs, [i*CategoryManager.NUM_CATEGORIES + j]),
                        (self._semantic_inputs, [i*CategoryManager.NUM_CATEGORIES + j])
                    )
                    prods.append(prod)
            self._conc_inputs.set_inputs(*map(spn.Input.as_input, prods))
            self._expanded = True


    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        raise NotImplemented


    def load(self, path, sess):
        """
        Loads the SPN structure and parameters.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        raise NotImplemented


    def train(self, sess, *args, **kwargs):
        """
        Train the SPN.

        sess (tf.Session): Tensorflow session object
        """
        raise NotImplemented
