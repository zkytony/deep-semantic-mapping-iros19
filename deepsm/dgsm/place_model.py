# Single DGSM for all classes
from enum import Enum
import libspn as spn
import numpy as np
import tensorflow as tf
from deepsm.util import CategoryManager
from deepsm.dgsm.data import Data

class PlaceModel:
    """SPN place model for multiple classes.

    Discriminative Soft Gradient-Descent Learning."""
    
    class WeightInitValue(Enum):
        ONE = 1
        RANDOM = 2

    def __init__(self, data,
                 view_input_dist, view_num_decomps, view_num_subsets,
                 view_num_mixtures, view_num_input_mixtures, view_num_top_mixtures,
                 place_num_decomps, place_num_subsets, place_num_mixtures,
                 place_num_input_mixtures,
                 weight_init_value, value_inference_type,
                 learning_rate, optimizer=tf.train.AdamOptimizer):
        self._data = data
        self._num_radius_cells = data.num_radius_cells
        self._num_angle_cells = data.num_angle_cells
        self._num_vars = data.num_radius_cells * data.num_angle_cells
        self._occupancy_vals = data.occupancy_vals

        self._view_input_dist = view_input_dist
        self._view_num_decomps = view_num_decomps
        self._view_num_subsets = view_num_subsets
        self._view_num_mixtures = view_num_mixtures
        self._view_num_input_mixtures = view_num_input_mixtures
        self._view_num_top_mixtures = view_num_top_mixtures
        self._place_num_decomps = place_num_decomps
        self._place_num_subsets = place_num_subsets
        self._place_num_mixtures = place_num_mixtures
        self._place_num_input_mixtures = place_num_input_mixtures
        self._weight_init_value = weight_init_value
        self._value_inference_type = value_inference_type
        self._learning_rate = learning_rate
        self._optimizer = optimizer

        self._build_model()
        self._sess = tf.Session()
        
    def _get_indices_for_view(self, view_borders, vn):
        alli = np.arange(self._num_angle_cells *
                         self._num_radius_cells).reshape(self._num_radius_cells,
                                                         self._num_angle_cells)
        ind_var = alli[:, view_borders[vn]:view_borders[vn + 1]]
        ind_var = ind_var.ravel().tolist()
        # Now turn var indices to IV indices
        if self._occupancy_vals == Data.OccupancyVals.THREE:
            return sorted([3 * i for i in ind_var] +
                          [(3 * i) + 1 for i in ind_var] +
                          [(3 * i) + 2 for i in ind_var])
        elif self._occupancy_vals == Data.OccupancyVals.TWO:
            return sorted([2 * i for i in ind_var] +
                          [(2 * i) + 1 for i in ind_var])
        else:
            raise Exception("Incorrect value of occupancy vals.")

    def _build_model(self):
        print("\nBuilding model...")

        # IVs
        if self._occupancy_vals == Data.OccupancyVals.THREE:
            self._ivs = spn.IVs(num_vars=self._num_vars, num_vals=3)
        elif self._occupancy_vals == Data.OccupancyVals.TWO:
            self._ivs = spn.IVs(num_vars=self._num_vars, num_vals=2)
        else:
            raise Exception("Incorrect value of occupancy vals.")

        num_views = 8
        view_borders = [0] + [(i + 1) * (self._num_angle_cells // num_views)
                              for i in range(num_views)]
        if view_borders[-1] != self._num_angle_cells:
            raise Exception()
        print("* View borders %s for %s angle cells and %s views" % (
            view_borders, self._num_angle_cells, num_views))

        view_dense_gen = spn.DenseSPNGeneratorLayerNodes(
            num_decomps=self._view_num_decomps,
            num_subsets=self._view_num_subsets,
            num_mixtures=self._view_num_mixtures,
            num_input_mixtures=self._view_num_input_mixtures,
            balanced=True,
            input_dist=self._view_input_dist,
            node_type=spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK)
        
        place_dense_gen = spn.DenseSPNGenerator(
            num_decomps=self._place_num_decomps,
            num_subsets=self._place_num_subsets,
            num_mixtures=self._place_num_mixtures,
            num_input_mixtures=self._place_num_input_mixtures,
            balanced=True,
            input_dist=spn.DenseSPNGenerator.InputDist.RAW)
            # node_type=spn.DenseSPNGenerator.NodeType.BLOCK)
        
        # For each class, build a sub-SPN
        class_roots = []
        for i in range(CategoryManager.NUM_CATEGORIES):
            print("----Class %d (%s)----" % (i, CategoryManager.category_map(i, rev=True)))
            sub_root = self._build_sub_model(num_views, view_borders, view_dense_gen, place_dense_gen)
            class_roots.append(sub_root)

        self._root = spn.Sum(*class_roots)

        # It shouldn't matter which object calls convert_to_layer_nodes since the function is independent to self.
        print("Converting to Layer SPN...")
        self._root = view_dense_gen.convert_to_layer_nodes(self._root)

        # Sum up all sub SPNs
        print("* Root valid: %s" % self._root.is_valid())
        if not len(self._root.get_scope()[0]) == self._num_vars:
            import pdb; pdb.set_trace()
            raise Exception()

        self._latent = self._root.generate_ivs(name="root_Catg_IVs")


        # Print structure statistics
        all_nodes = self._root.get_nodes()
        num_sums = 0
        num_single_sums = 0
        num_weight_mults = 0
        num_prods = 0
        num_single_prods = 0
        for n in all_nodes:
            if isinstance(n, spn.Sum):
                num_sums += 1
                num_single_sums += len(n.inputs) - 1
                num_weight_mults += len(n.inputs)
            if isinstance(n, spn.Product):
                num_prods += 1
                num_single_prods += len(n.inputs) - 1
        print("* Number of sums, products: %s %s" %
              (num_sums, num_prods))
        print("* Number of single sum ops, product ops: %s %s" %
              (num_single_sums, num_single_prods))
        print("* Number of weight multiplications : %s" %
              (num_weight_mults))
        print("Done!")

        # Add weights
        print("Adding weights...")
        if self._weight_init_value == PlaceModel.WeightInitValue.ONE:
            wiv = 1
        elif self._weight_init_value == PlaceModel.WeightInitValue.RANDOM:
            wiv = spn.ValueType.RANDOM_UNIFORM(10, 11)
        else:
            raise Exception()
        spn.generate_weights(self._root, init_value=wiv, log=True)
        print("Done!")

        # Add learning ops
        print("\nAdding learning ops...")
        self._gd_learning = spn.GDLearning(
            self._root, log=True,
            value_inference_type=self._value_inference_type,
            learning_rate=self._learning_rate,
            learning_type=spn.LearningType.DISCRIMINATIVE,
            learning_inference_type=spn.LearningInferenceType.SOFT,
            use_unweighted=True)
        self._reset_accumulators = self._gd_learning.reset_accumulators()
        self._learn_spn = self._gd_learning.learn(optimizer=self._optimizer)
        self._train_likelihood = self._gd_learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)
        self._init_weights = spn.initialize_weights(self._root)
        print("Done!")
        

    def _build_sub_model(self, num_views, view_borders, view_dense_gen, place_dense_gen):
        view_roots = [None for _ in range(num_views)]
        for vi in range(num_views):
            view_start = view_borders[vi]
            view_end = view_borders[vi + 1]
            print("* Generating VIEW %d for start:%d end:%d" %
                  (vi, view_start, view_end))
            view_roots[vi] = view_dense_gen.generate(
                (self._ivs, self._get_indices_for_view(view_borders, vi)))
        print("* View roots: %s" % view_roots)
        print("* Number of nodes in each view: %s" %
              ([i.get_num_nodes() for i in view_roots]))
        print("* Scope of first view: %s" % sorted(view_roots[0].get_scope()[0]))

        # Obtain product nodes instead of roots for each view
        view_products = [None for _ in range(num_views)]
        for vi in range(num_views):
            view_products[vi] = [v.node for v in view_roots[vi].values]

        # Create sums for each view
        view_sums = [None for _ in range(num_views)]
        for vi in range(num_views):
            prods = view_products[vi]
            view_sums[vi] = [spn.Sum(*prods)
                             for _ in range(self._view_num_top_mixtures)]
        print("* View sums: %s " % (view_sums,))

        # Generate place network
        print("* Generating PLACE...")
        root = place_dense_gen.generate(*[i for v in view_sums for i in v])
        print("* Root valid: %s" % root.is_valid())
        if not len(root.get_scope()[0]) == self._num_vars:
            raise Exception()
        print("* Number of nodes in whole model: %s" %
              root.get_num_nodes())

        all_nodes = root.get_nodes()
        num_sums = 0
        num_single_sums = 0
        num_weight_mults = 0
        num_prods = 0
        num_single_prods = 0
        for n in all_nodes:
            if isinstance(n, spn.Sum):
                num_sums += 1
                num_single_sums += len(n.inputs) - 1
                num_weight_mults += len(n.inputs)
            if isinstance(n, spn.Product):
                num_prods += 1
                num_single_prods += len(n.inputs) - 1
        print("* Number of sums, products: %s %s" %
              (num_sums, num_prods))
        print("* Number of single sum ops, product ops: %s %s" %
              (num_single_sums, num_single_prods))
        print("* Number of weight multiplications : %s" %
              (num_weight_mults))
        print("Done!")
        return root


    def train(self, num_batches, num_epochs):
        np.set_printoptions(threshold=np.nan)
        self._sess.run(self._init_weights)
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(self._reset_accumulators)

        # Print weights
        print(self._sess.run(self._root.weights.node.variable))

        batch_size = self._data.training_scans.shape[0] // num_batches
        
        prev_likelihood = 100
        likelihood = 0
        epoch = 0
        # Print weights
        print(self._sess.run(self._root.weights.node.variable))
        print(self._sess.run(self._gd_learning.root_accum()))

        while epoch < num_epochs:
            prev_likelihood = likelihood
            likelihoods = []
            for batch in range(num_batches):
                start = (batch) * batch_size
                stop = (batch + 1) * batch_size
                print("- EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)

                # Run learning
                train_likelihoods_arr, avg_train_likelihood_val, _, = \
                    self._sess.run([self._train_likelihood,
                                    self._avg_train_likelihood,
                                    self._learn_spn],
                                   feed_dict={self._ivs: self._data.training_scans[start:stop],
                                              self._latent: self._data.training_labels[start:stop]})
                                              
                # Print avg likelihood of this batch data on previous batch weights
                print("  Avg likelihood (this batch data on previous weights): %s" %
                      (avg_train_likelihood_val))
                likelihoods.append(avg_train_likelihood_val)
                
            likelihood = sum(likelihoods) / len(likelihoods)
            print("- Batch avg likelihood: %s" % (likelihood))
            epoch += 1

        print("Done!")        
