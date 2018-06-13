# Single DGSM for all classes
from enum import Enum
import libspn as spn
import numpy as np
import tensorflow as tf
from deepsm.util import CategoryManager
from deepsm.dgsm.data import Data
import pprint as pp
import os, sys
import json
import random

def norm_cm(cm):
    return cm / np.sum(cm, axis=1, keepdims=True)

class PlaceModel:
    """SPN place model for multiple classes.

    Discriminative Soft Gradient-Descent Learning."""
    
    class WeightInitValue(Enum):
        ONE = 1
        RANDOM = 2

    class LearningType(Enum):
        EM = 1
        GD = 2

    def __init__(self, data,
                 view_input_dist, view_num_decomps, view_num_subsets,
                 view_num_mixtures, view_num_input_mixtures, view_num_top_mixtures,
                 place_num_decomps, place_num_subsets, place_num_mixtures,
                 place_num_input_mixtures,
                 weight_init_value, value_inference_type,
                 learning_rate,
                 init_accum_val,
                 smoothing_val, smoothing_min, smoothing_decay,
                 optimizer=tf.train.AdamOptimizer,
                 learning_type=LearningType.GD):
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
        self._smoothing_val = smoothing_val
        self._smoothing_min = smoothing_min
        self._smoothing_decay = smoothing_decay
        self._init_accum = init_accum_val
        self._learning_type = learning_type

        self._build_model()
        self._sess = tf.Session()

        self._known_classes = CategoryManager.known_categories()
        
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
        self._root = place_dense_gen.convert_to_layer_nodes(self._root)

        # Sum up all sub SPNs
        print("* Root valid: %s" % self._root.is_valid())
        if not len(self._root.get_scope()[0]) == self._num_vars:
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
        if self._learning_type == PlaceModel.LearningType.EM:
            self._additive_smoothing_var = tf.Variable(self._smoothing_val,
                                                       dtype=spn.conf.dtype)
            self._learning = spn.EMLearning(
                self._root, log=True,
                value_inference_type=self._value_inference_type,
                additive_smoothing=self._additive_smoothing_var,
                add_random=None,
                initial_accum_value=self._init_accum,
                use_unweighted=True)
            self._learn_spn = self._learning.accumulate_updates()
            self._update_spn = self._learning.update_spn()
        else:
            self._learning = spn.GDLearning(
                self._root, log=True,
                value_inference_type=self._value_inference_type,
                learning_rate=self._learning_rate,
                learning_type=spn.LearningType.DISCRIMINATIVE,
                learning_inference_type=spn.LearningInferenceType.HARD,
                use_unweighted=True)
            self._learn_spn = self._learning.learn(optimizer=self._optimizer)
        self._reset_accumulators = self._learning.reset_accumulators()
        self._train_likelihood = self._learning.value.values[self._root]
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
        """Run training"""

        # Get data
        training_scans = self._data.training_scans
        training_labels = self._data.training_labels

        print("* Num training samples: %d" % len(training_scans))

        # Run relevant ops
        np.set_printoptions(threshold=np.nan)
        self._sess.run(self._init_weights)
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(self._reset_accumulators)

        # Print weights
        print(self._sess.run(self._root.weights.node.variable))

        batch_size = training_scans.shape[0] // num_batches
        
        prev_likelihood = 100
        likelihood = 0
        epoch = 0
        # Print weights
        print(self._sess.run(self._root.weights.node.variable))
        print(self._sess.run(self._learning.root_accum()))

        while epoch < num_epochs:
            prev_likelihood = likelihood
            likelihoods = []
            for batch in range(num_batches):
                start = (batch) * batch_size
                stop = (batch + 1) * batch_size
                print("- EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)

                if self._learning_type == PlaceModel.LearningType.EM:
                    # Adjust smoothing
                    ads = max(np.exp(-epoch * self._smoothing_decay) *
                              self._smoothing_val,
                              self._smoothing_min)
                    self._sess.run(self._additive_smoothing_var.assign(ads))
                    print("  Smoothing: ", self._sess.run(self._additive_smoothing_var))

                # Run learning
                train_likelihoods_arr, avg_train_likelihood_val, _, = \
                    self._sess.run([self._train_likelihood,
                                    self._avg_train_likelihood,
                                    self._learn_spn],
                                   feed_dict={self._ivs: training_scans[start:stop],
                                              self._latent: training_labels[start:stop]})
                                              
                # Print avg likelihood of this batch data on previous batch weights
                print("  Avg likelihood (this batch data on previous weights): %s" %
                      (avg_train_likelihood_val))
                likelihoods.append(avg_train_likelihood_val)

                if self._learning_type == PlaceModel.LearningType.EM:
                    # Update weights
                    self._sess.run(self._update_spn)
                
            likelihood = sum(likelihoods) / len(likelihoods)
            print("- Batch avg likelihood: %s" % (likelihood))
            epoch += 1

        print("Done!")        


    def test(self, results_dir, batch_size=50, graph_test=True):

        # Generate states
        mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
        self._mpe_ivs, self._mpe_latent = mpe_state_gen.get_state(self._root, self._ivs, self._latent)

        # Make numpy array of test samples
        testing_scans = []
        testing_labels = []
        for d in self._data.testing_data:
            rid = d[0]
            rcat = d[1]
            scan = d[2]
            testing_scans.append(scan)
            testing_labels.append(CategoryManager.category_map(rcat))

        testing_scans = np.vstack(testing_scans)
        testing_labels = np.vstack(testing_labels)

        num_batches = testing_scans.shape[0] // batch_size
        accuracy_per_step = []
        likelihoods_per_step = [[] for k in range(CategoryManager.NUM_CATEGORIES)]  # NUM_CATEGORIES x num_test_scans
        last_batch = True
        for batch in range(num_batches):
            start = (batch) * batch_size
            if last_batch and (batch + 1) == num_batches:
                stop = testing_scans.shape[0]
            else:
                stop = (batch + 1) * batch_size

            # Session
            mpe_latent_val = self._sess.run([self._mpe_latent],
                                            feed_dict={self._ivs: testing_scans[start:stop],
                                                       self._latent: np.ones((stop - start, 1)) * -1})
            accuracy_per_step.append(np.mean(mpe_latent_val == testing_labels[start:stop]))
            
            # All marginals
            for k in range(CategoryManager.NUM_CATEGORIES):
                train_likelihoods_arr = \
                    self._sess.run([self._train_likelihood],
                                   feed_dict={self._ivs: testing_scans[start:stop],
                                              self._latent: np.full((stop - start, 1), k)})
                likelihoods_per_step[k].extend(train_likelihoods_arr[0].flatten())

        accuracy = np.mean(accuracy_per_step) * 100
        print("Classification accyracy on Test set: ", accuracy)

        # Process graph results
        
        # Confusion matrices
        cm_mpe_weighted = np.zeros((CategoryManager.NUM_CATEGORIES, CategoryManager.NUM_CATEGORIES))
        
        graph_results = {}
        likelihoods = np.transpose(np.array(likelihoods_per_step, dtype=float))
        for i, d in enumerate(self._data.testing_data):
            rid = d[0]
            rcat = d[1]
            if rid not in graph_results:  # rid is actually graph_id
                graph_results[rid] = {}

            true_class_index = CategoryManager.category_map(rcat)
            pred_class_index = np.argmax(likelihoods[i])
                
            # Record in confusion matrix
            cm_mpe_weighted[true_class_index, pred_class_index] += 1

            if graph_test:
                node_id = d[-1]
                graph_results[rid][node_id] = [rcat, self._known_classes[pred_class_index], list(likelihoods[i]), list(likelihoods[i])]

        if graph_test:
            # Overall statistics
            stats = self._compute_stats(graph_results)
            print("- Overall statistics")
            pp.pprint(stats)
        
        # Confusion matrix
        print("- Confusion matrix for MPE (weighted):")
        pp.pprint(self._known_classes)
        pp.pprint(cm_mpe_weighted)
        pp.pprint(norm_cm(cm_mpe_weighted) * 100.0)

        if graph_test:
            # Save
            os.makedirs(os.path.join(results_dir, "graphs"), exist_ok=True)
            for graph_id in graph_results:
                with open(os.path.join(results_dir, "graphs", graph_id + "_likelihoods.json"), 'w') as f:
                    json.dump(graph_results[graph_id], f)


    def _compute_stats(self, graph_results):
        stats = {}
        total_cases = 0
        total_correct = 0
        total_per_class = {}
        for graph_id in graph_results:
            graph_cases = 0
            graph_correct = 0
            graph_per_class = {}
            for nid in graph_results[graph_id]:
                groundtruth = graph_results[graph_id][nid][0]
                # We skip unknown cases
                if groundtruth not in self._known_classes:
                    continue
                if groundtruth not in total_per_class:
                    total_per_class[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                if groundtruth not in graph_per_class:
                    graph_per_class[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                
                # We have one more test case
                total_cases += 1
                graph_cases += 1
                total_per_class[groundtruth][0] += 1
                graph_per_class[groundtruth][0] += 1
                
                prediction = graph_results[graph_id][nid][1]
                if prediction == groundtruth:
                    graph_correct += 1
                    total_correct += 1
                    total_per_class[groundtruth][1] += 1
                    graph_per_class[groundtruth][1] += 1
                total_per_class[groundtruth][2] = total_per_class[groundtruth][1] / total_per_class[groundtruth][0]
                graph_per_class[groundtruth][2] = graph_per_class[groundtruth][1] / graph_per_class[groundtruth][0]
            
            stats[graph_id] = {'num_cases': graph_cases,
                               'num_correct': graph_correct,
                               'accuracy': graph_correct / graph_cases,
                               'class_results': graph_per_class}
        stats.update({'num_cases': total_cases,
                      'num_correct': total_correct,
                      'accuracy': total_correct / total_cases,
                      'class_results': total_per_class})
        return stats


    @staticmethod
    def balance_data(scans, labels, rnd=None):
        # Upsample the miniority class instances so that all class have the same number of training samples
        if rnd is None:
            rnd = random.Random()

        if len(scans) != len(labels):
            raise ValueError("Scans and labels do not have the same dimensions!")

        class_counts, scans_by_class = PlaceModel._count_class_samples(scans, labels)
        
        print("Training samples count by classes, before balancing:")
        pp.pprint(class_counts)
        
        class_with_most_samples = max(class_counts, key=lambda x: class_counts[x])
        up_limit = class_counts[class_with_most_samples]

        for catg_num in class_counts:
            if catg_num == class_with_most_samples:
                continue

            difference = up_limit - class_counts[catg_num]
            for i in range(difference):
                # Randomly pick a scan from scans
                scans.append(rnd.choice(scans_by_class[catg_num]))
                labels.append(catg_num)

        # Recompute class counts
        class_counts, _ = PlaceModel._count_class_samples(scans, labels)

        print("Training samples count by classes, after balancing:")
        pp.pprint(class_counts)


    @staticmethod
    def _count_class_samples(scans, labels):
        scans_by_class = {}
        class_counts = {}
        for i in range(len(scans)):
            class_counts[labels[i]] = class_counts.get(labels[i], 0) + 1
            if labels[i] not in scans_by_class:
                scans_by_class[labels[i]] = []
            scans_by_class[labels[i]].append(scans[i])
        return class_counts, scans_by_class
