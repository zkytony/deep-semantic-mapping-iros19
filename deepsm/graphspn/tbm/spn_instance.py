import sys
from abc import abstractmethod
import tensorflow as tf
import libspn as spn
import numpy as np
from numpy import float32
import random
import copy
import os

from deepsm.graphspn.tbm.spn_template import TemplateSpn
from deepsm.graphspn.spn_model import SpnModel, mod_compute_graph_up
from deepsm.graphspn.tbm.template import EdgeTemplate, NodeTemplate, SingleEdgeTemplate, EdgeRelationTemplate
from deepsm.util import CategoryManager, ColdDatabaseManager, compute_edge_pair_view_distance
from deepsm.experiments.common import GROUNDTRUTH_ROOT, COLD_ROOT


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

           If template is EdgeTemplate or EdgeRelationTemplate, then:
             divisions (int) number of views per place
        """
        super().__init__(**kwargs)        
        self._seq_id = kwargs.get("seq_id", "default_1")
        # sort spns by template size.
        self._spns = sorted(spns, key=lambda x: x[1].size(), reverse=True)
        self._template_mode = self._spns[0][1].code()
        self._topo_map = topo_map
        self._expanded = False


    @property
    @abstractmethod
    def vn(self):
        pass

    @property
    def root(self):
        return self._root


    @abstractmethod
    def _init_struct(self, sess, divisions=-1, num_partitions=1,
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
        pass

    def _init_ops_basics(self):
        print("Initializing learning Ops...")
        if self._learning_algorithm == spn.GDLearning:
            learning = spn.GDLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      learning_rate=self._learning_rate,
                                      learning_type=self._learning_type,
                                      learning_inference_type=self._learning_inference_type)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.learn(optimizer=self._optimizer)
            
        elif self._learning_algorithm == spn.EMLearning:
            learning = spn.EMLearning(self._root, log=True, value_inference_type = self._value_inference_type,
                                  additive_smoothing = self._additive_smoothing_var, use_unweighted=True)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.accumulate_updates()
            self._update_spn = learning.update_spn()

        self._train_likelihood = learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)
        

    @abstractmethod
    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        pass

    @abstractmethod
    def marginal_inference(self, sess, query_nids, query, query_lh=None):
        """
        Computes marginal distribution of queried variables.

        Now, only uses the DUMMY method. Iterate over all possible assignments of
        all inquired variables, evaluate the network and use the result as the likelihood
        for that assignment. Note that the marginals are in log space.

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.
        """
        pass

    @abstractmethod
    def evaluate(self, sess, sample, sample_lh=None):
        """
        sess (tf.Session): a session.
        """
        pass


    @abstractmethod
    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.
        """
        pass


    def expand(self, use_cont_vars=False):
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
            # Note: use RawInput when doing cold database experiments with dgsm input. Use ContVars for synthetic experiments
            if use_cont_vars:
                self._likelihood_inputs = spn.ContVars(num_vars=num_vars*CategoryManager.NUM_CATEGORIES) #spn.RawInput(num_vars=num_vars*CategoryManager.NUM_CATEGORIES)
            else:
                self._likelihood_inputs = spn.RawInput(num_vars=num_vars*CategoryManager.NUM_CATEGORIES) #spn.RawInput(num_vars=num_vars*CategoryManager.NUM_CATEGORIES)

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


class NodeTemplateInstanceSpn(InstanceSpn):

    
    def __init__(self, topo_map, sess, *spns, **kwargs):
        super().__init__(topo_map, sess, *spns, **kwargs)

        assert self._template_mode == NodeTemplate.code()
        
        self._id_incr = 0
        num_partitions = kwargs.get('num_partitions', 1)
        self._init_struct(sess, divisions=kwargs.get('divisions', -1),
                          num_partitions=kwargs.get('num_partitions', 1),
                          visualize_partitions_dirpath=kwargs.get('visualize_partitions_dirpath', None),
                           db_name=kwargs.get('db_name', None), extra_partition_multiplyer=kwargs.get('extra_partition_multiplyer', 1))
        self._inputs_size = len(self._topo_map.nodes)


    @property
    def vn(self):
        return {
            'CATG_IVS': "Catg_%d_%s" % (self._template_mode, self._seq_id),
            'VIEW_IVS':"View_%d_%s" % (self._template_mode, self._seq_id),
            'CONC': "Conc_%d_%s" % (self._template_mode, self._seq_id),
            'SEMAN_IVS': "Exp_Catg_%d_%s" % (self._template_mode, self._seq_id),
            'LH_CONT': "Exp_Lh_%d_%s" % (self._template_mode, self._seq_id)
        }
        
    def _inputs_list(self, input_type):
        """
        input_type can be 'catg' or 'conc', or 'view' (edge only)
        """
        inputs = [self._template_iv_map[tid][input_type] for tid in sorted(self._template_iv_map)]
        return inputs

        
    def _init_struct(self, sess, divisions=-1, num_partitions=1,
                     visualize_partitions_dirpath=None, db_name=None, extra_partition_multiplyer=1):
        """
        Initialize the structure for training. (private method)

        sess: (tf.Session): a session that contains all weights.

        **kwargs:
           num_partitions (int): number of partitions (children for root node)
           If template is EdgeTemplate, then:
             divisions (int) number of views per place
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
        """
        
        # Import if visualize partition is requested
        if visualize_partitions_dirpath:
            print("Importing matplotlib...")
            import matplotlib
            matplotlib.use('Agg')
            from pylab import rcParams
            import matplotlib.pyplot as plt
            rcParams['figure.figsize'] = 22, 14
            # Now, we create a cold database manager used to draw the map
            coldmgr = ColdDatabaseManager(db_name, COLD_ROOT, GROUNDTRUTH_ROOT)
            
        # Create vars and maps
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

        """Making an SPN"""
        """Now, partition the graph, copy structure, and connect self._catg_inputs appropriately to the network."""
        # Main template partition
        _k = 0
        for supergraph, unused_graph in used_partitions:
            print("Partition %d" % (_k+1))
            nodes_covered = set({})
            template_spn_roots = []
            main_template_spn = self._spns[0][0]
            print("Will duplicate %s %d times." % (main_template.__name__, len(supergraph.nodes)))
            template_spn_roots.extend(NodeTemplateInstanceSpn._duplicate_template_spns(self, tspns, main_template, supergraph, nodes_covered))

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
                template_spn_roots.extend(NodeTemplateInstanceSpn._duplicate_template_spns(self, tspns, template, supergraph_2nd, nodes_covered))

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
        ## End for loop ##
        
        # Sum up all
        self._root = spn.Sum(*pspns)
        assert self._root.is_valid()
        self._root.generate_weights(trainable=True)
        # initialize ONLY the weights node for the root
        sess.run(self._root.weights.node.initialize())
    #---- end init_struct ----#

    @classmethod    
    def _duplicate_template_spns(cls, ispn, tspns, template, supergraph, nodes_covered):
        """
        Convenient method for copying template spns. Modified `nodes_covered`.
        """
        sys.stdout.write("Duplicating %s... " % template.__name__)
        roots = []
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

            sys.stdout.write("%d " % (__i+1))
            sys.stdout.flush()
            copied_tspn_root = mod_compute_graph_up(tspns[template.__name__][0],
                                                    TemplateSpn.dup_fun_up,
                                                    tmpl_num_vars=[len(nids)],
                                                    tmpl_num_vals=[CategoryManager.NUM_CATEGORIES],
                                                    graph_num_vars=[len(ispn._topo_map.nodes)],
                                                    conc=ispn._conc_inputs,
                                                    labels=[labels])
            assert copied_tspn_root.is_valid()
            roots.append(copied_tspn_root)
            __i+=1
            ispn._id_incr += 1
        sys.stdout.write("\n")
        return roots
    #---- end duplicate_template_spns ----#
        

    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        self._init_ops_basics()
        if not no_mpe:
            print("Initializing MPE Ops...")
            mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
            if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                if not self._expanded:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs)
                else:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs)

                    
    def marginal_inference(self, sess, query_nids, query, query_lh=None):
        """
        Computes marginal distribution of queried variables.

        Now, only uses the DUMMY method. Iterate over all possible assignments of
        all inquired variables, evaluate the network and use the result as the likelihood
        for that assignment. Note that the marginals are in log space.

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.

        *args
            query (dict): A dictionary mapping from node id to its category number
                           ASSUME that the inquired nodes have already been assigned to '-1'.
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
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
            sample (dict): A dictionary mapping from node id to its category number
          If expanded (additional parameters)
              sample_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
        """
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


    
    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.

        *args
          If NodeTemplate:
            query (dict): A dictionary mapping from node id to its category number
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
        Returns:
           If NodeTemplate:
              returns a category map (from id to category number)
        """
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
#------- END NodeTemplateinstancespn --------#


class EdgeRelationTemplateInstanceSpn(InstanceSpn):

    def __init__(self, topo_map, sess, *spns, **kwargs):
        super().__init__(topo_map, sess, *spns, **kwargs)

        assert self._template_mode == EdgeRelationTemplate.code()

        self._init_struct(sess, divisions=kwargs.get('divisions', 8),
                          num_partitions=kwargs.get('num_partitions', 1),
                          visualize_partitions_dirpath=kwargs.get('visualize_partitions_dirpath', None),
                          db_name=kwargs.get('db_name', None))
    @property
    def vn(self):
        return {
            'CATG_IVS': "Catg_%d_%s" % (self._template_mode, self._seq_id),
            'VIEW_IVS':"View_%d_%s" % (self._template_mode, self._seq_id),
            'CONC': "Conc_%d_%s" % (self._template_mode, self._seq_id),
            'SEMAN_IVS': "Exp_Catg_%d_%s" % (self._template_mode, self._seq_id),
            'LH_CONT': "Exp_Lh_%d_%s" % (self._template_mode, self._seq_id)
        }
    
    def _init_struct(self, sess, divisions=-1, num_partitions=1,
                     visualize_partitions_dirpath=None, db_name=None, **kwargs):
        """
        Initialize structure for instance spn using edge relation templates
        """
        
        # Import if visualize partition is requested
        if visualize_partitions_dirpath:
            print("Importing matplotlib...")
            import matplotlib
            matplotlib.use('Agg')
            from pylab import rcParams
            import matplotlib.pyplot as plt
            rcParams['figure.figsize'] = 22, 14
            # Now, we create a cold database manager used to draw the map
            coldmgr = ColdDatabaseManager(db_name, COLD_ROOT, GROUNDTRUTH_ROOT)
        
        # Specify labels for both category variables and edge pair variables (i.e. view distances)
        self._node_label_map = {}  # map from node id to label
        self._label_node_map = {}  # map from label to node id
        _i = 0
        for nid in self._topo_map.nodes:
            self._node_label_map[nid] = _i
            self._label_node_map[_i] = nid
            _i += 1

        # Each view variable has an associated edge pair. Label the edge pairs.
        _i = 0
        node_edge_pairs = self._topo_map.connected_edge_pairs()
        self._edpair_label_map = {}  # map from (edge1_id, edge2_id) to their label (integer)
        self._label_edpair_map = {}  # map from an label to (edge1_id, edge2_id)
        self._edpairs = {}  # map from (edge1_id, edge2_id) to a tuple of  edge objects
        for nid in node_edge_pairs:
            for edpair in node_edge_pairs[nid]:
                edpair_id = (edpair[0].id, edpair[1].id)
                self._edpairs[edpair_id] = edpair
                self._edpair_label_map[edpair_id] = _i
                self._label_edpair_map[_i] = edpair_id
                _i += 1
        
        # Single concat, first portion for category variables, and second
        # portion for view distance variables.
        num_view_dists = divisions // 2
        self._catg_inputs = spn.IVs(num_vars=len(self._topo_map.nodes), num_vals=CategoryManager.NUM_CATEGORIES, name=self.vn['CATG_IVS'])
        self._view_dist_inputs = spn.IVs(num_vars=len(self._edpairs), num_vals=num_view_dists, name=self.vn['VIEW_IVS'])
        self._conc_inputs = spn.Concat(self._catg_inputs, self._view_dist_inputs, name=self.vn['CONC'])

        spns_map = { tspn.template.to_tuple() : tspn for tspn, _ in self._spns }

        graph_num_vars = [len(self._topo_map.nodes), len(self._edpairs)]

        """Making an SPN"""
        pspns = []
        for _k in range(num_partitions):
            print("Partition %d" % (_k+1))
            template_spn_roots = []
            
            ert_map = self._topo_map.partition_by_edge_relations()
            
            for t in ert_map:
                num_nodes, num_edge_pair = t
                tspn = spns_map[t]

                tmpl_num_vars = [num_nodes, num_edge_pair]
                tmpl_num_vals = [CategoryManager.NUM_CATEGORIES, num_view_dists]
                
                # for each edge relation template, duplicate the appropriate template SPN to cover it.
                sys.stdout.write("Duplicating %s... " % str(t))
                __i = 0
                for ert in ert_map[t]:
                    labels = [[],[]]
                    if ert.nodes is not None:
                        for node in ert.nodes:
                            labels[0].append(self._node_label_map[node.id])
                    if ert.edge_pair is not None:
                        # Either edpair_id or its reverse should have been encountered. 
                        edpair_id = (ert.edge_pair[0].id, ert.edge_pair[1].id)
                        if edpair_id in self._edpairs:
                            labels[1].append(self._edpair_label_map[edpair_id])
                        elif tuple(reversed(edpair_id)) in self._edpairs:
                            labels[1].append(self._edpair_label_map[tuple(reversed(edpair_id))])
                        else:
                            raise ValueError("Unexpected edge pair %s" % edpair_id)

                    sys.stdout.write("%d " % (__i+1))
                    sys.stdout.flush()
                    copied_tspn_root = mod_compute_graph_up(tspn.root,
                                                            TemplateSpn.dup_fun_up,
                                                            tmpl_num_vars=tmpl_num_vars,
                                                            tmpl_num_vals=tmpl_num_vals,
                                                            graph_num_vars=graph_num_vars,
                                                            labels=labels,
                                                            conc=self._conc_inputs)
                    assert copied_tspn_root.is_valid()
                    template_spn_roots.append(copied_tspn_root)
                    __i += 1
                sys.stdout.write("\n")
                    
            # We can now create an SPN for this partition (i.e. pspn)
            p = spn.Product(*template_spn_roots)
            assert p.is_valid()
            pspns.append(p)

            """If visualize. Save."""
            if visualize_partitions_dirpath:
                self._topo_map.visualize_edge_relation_partition(plt.gca(),
                                                                 ert_map,
                                                                 coldmgr.groundtruth_file(self._seq_id.split("_")[0], 'map.yaml'))
                plt.savefig(os.path.join(visualize_partitions_dirpath, "partition-%d.png" % (_k+1)))
                plt.clf()
                print("Visualized partition %d" % (_k+1))
        ## End for loop ##
        
        # Sum up all
        self._root = spn.Sum(*pspns)
        assert self._root.is_valid()
        self._root.generate_weights(trainable=True)
        # initialize ONLY the weights node for the root
        sess.run(self._root.weights.node.initialize())
    ## END init_struct ##

    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        self._init_ops_basics()
        if not no_mpe:
            print("Initializing MPE Ops...")
            mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
            if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                if not self._expanded:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs, self._view_dist_inputs)
                else:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs, self._view_dist_inputs)


    def expand(self, use_cont_vars=False):
        if not self._expanded:
            super().expand(use_cont_vars=use_cont_vars)
            self._conc_inputs.add_inputs(self._view_dist_inputs)

            
    def evaluate(self, sess, sample, sample_lh=None):
        """
        Feed sample into the network and evaluate the likelihood.

        sess (tf.Session): a session.
        sample: either a tuple of two elements, or a single dictionary.

            If sample is a tuple of two elements, the first element is a dictionary,
            mapping from node id to its category number. The second element is a
            dictionary, mapping from pair of edge ids (edge1_id, edge2_id) to an integer
            indicating the absolute view distance between those two edges. It is assumed
            that the two edges share a meeting node. The second dictionary may not contain
            the view distance information for every pair of edges; For missing ones, the
            view distance based on the original topological map structure will be used.

            If sample is a single dictionary, then it is the same as the first element
            of the tuple defined above.

        sample_lh: a map from node id to a tuple of NUM_CATEGORIES number of float values.
        """
        view_dist_values = {}
        for edpair_id in self._edpairs:
            edge1, edge2 = self._edpairs[edpair_id]
            view_dist_values[edpair_id] = compute_edge_pair_view_distance(edge1, edge2) - 1  # -1 is because vdist ranges from 1-4 but we want 0-3 as input to the network
        
        if type(sample) == dict:
            catg_values = sample
        else:
            catg_values = sample[0]
            for edpair_id in sample[1]:
                if edpair_id in view_dist_values:
                    view_dist_values[edpair_id] = sample[1][edpair_id]
                elif tuple(reversed(edpair_id)) in view_dist_values:
                    view_dist_values[tuple(reversed(edpair_id))] = sample[1][edpair_id]
                else:
                    raise ValueError("Edge pair ids %s is not known to the graph" % edpair_id)

        catg_ivs_assignment = []
        for l in range(len(self._label_node_map)):
            nid = self._label_node_map[l]
            catg_ivs_assignment.append(catg_values[nid])
        view_ivs_assignment = []
        for l in range(len(self._label_edpair_map)):
            edpair = self._label_edpair_map[l]
            view_ivs_assignment.append(view_dist_values[edpair])
            
        if not self._expanded:
            lh_val = sess.run(self._train_likelihood,
                              feed_dict={self._catg_inputs: np.array([catg_ivs_assignment], dtype=int),
                                         self._view_dist_inputs: np.array([view_ivs_assignment], dtype=int)})[0]
        else:
            # expanded. So get likelihoods
            lh_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                lh_assignment.extend(list(sample_lh[nid]))
            lh_val = sess.run(self._train_likelihood,
                              feed_dict={self._semantic_inputs: np.array([catg_ivs_assignment], dtype=int),
                                         self._view_dist_inputs: np.array([view_ivs_assignment], dtype=int),
                                         self._likelihood_inputs: np.array([lh_assignment])})[0]
        return lh_val
                
    def marginal_inference(self, sess, query_nids, query, query_lh=None):
        """
        Computes marginal distribution of queried variables.

        Now, only uses the DUMMY method. Iterate over all possible assignments of
        all inquired variables, evaluate the network and use the result as the likelihood
        for that assignment. Note that the marginals are in log space.

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.
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

    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.
        """
        pass
