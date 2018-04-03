from enum import Enum
import libspn as spn
from data import Data
import numpy as np
import tensorflow as tf
import os.path
import copy


class PlaceModel:
    """SPN place model that has place sub-models as children of the root."""

    def __init__(self, data, place_classes,
                 view_input_dist, view_num_decomps, view_num_subsets,
                 view_num_mixtures, view_num_input_mixtures,
                 view_num_top_mixtures,
                 place_num_decomps, place_num_subsets, place_num_mixtures,
                 place_num_input_mixtures,
                 weight_init_value, init_accum_value,
                 additive_smoothing_value, value_inference_type):
        self._data = data
        self._place_classes = place_classes
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
        self._init_accum_value = init_accum_value
        self._additive_smoothing_value = additive_smoothing_value
        self._value_inference_type = value_inference_type

        self._build_model()
        self._sess = tf.Session()

    def _build_model(self):
        # IVs
        if self._occupancy_vals == Data.OccupancyVals.THREE:
            self._ivs = spn.IVs(num_vars=self._num_vars, num_vals=3)
        elif self._occupancy_vals == Data.OccupancyVals.TWO:
            self._ivs = spn.IVs(num_vars=self._num_vars, num_vals=2)
        else:
            raise Exception("Incorrect value of occupancy vals.")

        submodels = [PlaceSubModel(
                         self._data, self._ivs,
                         self._view_input_dist, self._view_num_decomps,
                         self._view_num_subsets, self._view_num_mixtures,
                         self._view_num_input_mixtures, self._view_num_top_mixtures,
                         self._place_num_decomps, self._place_num_subsets,
                         self._place_num_mixtures, self._place_num_input_mixtures,
                         self._weight_init_value, self._init_accum_value,
                         self._additive_smoothing_value, self._value_inference_type)
                     for _ in self._place_classes]
        self._root = spn.Sum(*[submodel._root for submodel in submodels])

        # Add weights
        print("\nAdding weights...")
        if self._weight_init_value == PlaceSubModel.WeightInitValue.ONE:
            wiv = 1
        elif self._weight_init_value == PlaceSubModel.WeightInitValue.RANDOM:
            wiv = spn.ValueType.RANDOM_UNIFORM(0, 1)
        else:
            raise Exception()
        spn.generate_weights(self._root, init_value=wiv)
        print("Done!")

        self._latent = self._root.generate_ivs()

        # Add learning ops
        print("\nAdding learning ops...")
        self._additive_smoothing_var = tf.Variable(self._additive_smoothing_value,
                                                   dtype=spn.conf.dtype)
        self._em_learning = spn.EMLearning(
            self._root, log=True,
            value_inference_type=self._value_inference_type,
            additive_smoothing=self._additive_smoothing_var,
            add_random=None,
            initial_accum_value=self._init_accum_value,
            use_unweighted=True)
        self._init_weights = spn.initialize_weights(self._root)
        self._reset_accumulators = self._em_learning.reset_accumulators()
        self._accumulate_updates = self._em_learning.accumulate_updates()
        self._update_spn = self._em_learning.update_spn()
        self._train_likelihood = self._em_learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)
        print("Done!")

        print("* Root valid: %s" % self._root.is_valid())
        if not len(self._root.get_scope()[0]) == self._num_vars + 1:
            print(len(self._root.get_scope()[0]), self._num_vars)
            raise Exception()
        print("* Number of nodes in whole model: %s" %
              self._root.get_num_nodes())

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

    def train(self, stop_condition, additive_smoothing_min,
              additive_smoothing_decay, results_dir):
        np.set_printoptions(threshold=np.nan)
        self._sess.run(self._init_weights)
        self._sess.run(self._reset_accumulators)

        num_batches = 4
        batch_size = self._data.training_scans.shape[0] // num_batches
        prev_likelihood = 100
        likelihood = 0
        epoch = 0
        # Print weights
        # print(self._sess.run(self._root.weights.node.variable))
        # print(self._sess.run(self._em_learning.root_accum()))

        while abs(prev_likelihood - likelihood) > 10*stop_condition:
            prev_likelihood = likelihood
            likelihoods = []
            for batch in range(num_batches):
                start = (batch) * batch_size
                stop = (batch + 1) * batch_size
                print("- EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)
                # Adjust smoothing
                ads = max(np.exp(-epoch * additive_smoothing_decay) *
                          self._additive_smoothing_value,
                          additive_smoothing_min)
                self._sess.run(self._additive_smoothing_var.assign(ads))
                print("  Smoothing: ", self._sess.run(self._additive_smoothing_var))
                # Run accumulate_updates
                train_likelihoods_arr, avg_train_likelihood_val, _, = \
                    self._sess.run(
                        [self._train_likelihood,
                         self._avg_train_likelihood,
                         self._accumulate_updates],
                        feed_dict={
                            self._ivs: self._data.training_scans[start:stop],
                            self._latent: self._data.training_labels[start:stop]})
                # Print avg likelihood of this batch data on previous batch weights
                # print("  Likelihood array: %s" % (train_likelihoods_arr))
                print("  Avg likelihood (this batch data on previous weights): %s" %
                      (avg_train_likelihood_val))
                likelihoods.append(avg_train_likelihood_val)
                # Update weights
                self._sess.run(self._update_spn)
                # Print weights
                # print(self._sess.run(self._root.weights.node.variable))
                # print(self._sess.run(self._em_learning.root_accum()))
            likelihood = sum(likelihoods) / len(likelihoods)
            print("- Batch avg likelihood: %s" % (likelihood))
            epoch += 1

            print("Current correct ratio on the test set")
            marginal_vals = self._calculate_marginal_vals(self._data.testing_scans)
            labels = self._data.testing_labels
            kc_idx = (labels != len(self._place_classes)).squeeze()
            max_idx = np.argmax(marginal_vals, axis=1)
            max_idx = max_idx.reshape((len(max_idx), 1))
            result = (max_idx[kc_idx] == labels[kc_idx])
            print(float(np.sum(result))/len(result))

        print("Saving the trained model")
        spn.JSONSaver(os.path.join(results_dir, 'trained.spn'), pretty=True) \
           .save(self._root)

        print("Done!")

    def semisup_train(self, stop_condition, additive_smoothing_min,
                      additive_smoothing_decay, results_dir, threshold):
        # self._sess.run(self._init_weights)
        # self._sess.run(self._reset_accumulators)

        num_batches = 1
        batch_size = None  # Place holder
        prev_likelihood = 100
        likelihood = 0
        epoch = 0

        os.makedirs(os.path.join(results_dir, 'semisup'), exist_ok=True)

        # Print weights
        # print(self._sess.run(self._root.weights.node.variable))
        # print(self._sess.run(self._em_learning.root_accum()))

        while abs(prev_likelihood - likelihood) > stop_condition:
            prev_likelihood = likelihood
            likelihoods = []

            marginal_vals = self._calculate_marginal_vals(self._data.testing_scans)
            np.save(os.path.join(results_dir, 'semisup', 
                                 'testing_marginal_vals_' + str(epoch)), marginal_vals)
            np.save(os.path.join(results_dir, 'semisup', 
                                 'testing_labels_' + str(epoch)), self._data.testing_labels)

            max_idx = np.argmax(marginal_vals, axis=1)
            max_idx = max_idx.reshape((len(max_idx), 1))

            thresh_idx = np.zeros((len(max_idx)), dtype=bool)
            for i in range(len(max_idx)):
                thresh_idx[i] = \
                    (np.max(marginal_vals[i][:]) > threshold[max_idx[i][0]])
            # thresh_idx = np.max(marginal_vals, axis=1) > threshold
            print(np.sum(thresh_idx))

            kc_idx = (self._data.testing_labels != len(self._place_classes)).squeeze()
            filt_idx = np.logical_and(thresh_idx, kc_idx)

            batch_size = np.sum(filt_idx) // num_batches

            result = (max_idx[kc_idx] == self._data.testing_labels[kc_idx])
            print(np.sum(result)/len(result))
            result = (max_idx[filt_idx] == self._data.testing_labels[filt_idx])
            print(float(np.sum(result))/len(result))

            for batch in range(num_batches):
                start = (batch) * batch_size
                stop = (batch + 1) * batch_size
                print("- EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)

                # Adjust smoothing
                ads = max(np.exp(-epoch * additive_smoothing_decay) *
                          self._additive_smoothing_value,
                          additive_smoothing_min)
                self._sess.run(self._additive_smoothing_var.assign(ads))
                print("  Smoothing: ", self._sess.run(self._additive_smoothing_var))

                # Run accumulate_updates
                train_likelihoods_arr, avg_train_likelihood_val, _, = \
                    self._sess.run(
                        [self._train_likelihood,
                         self._avg_train_likelihood,
                         self._accumulate_updates],
                        feed_dict={self._ivs: self._data.testing_scans[filt_idx],
                                   self._latent: max_idx[filt_idx]})

                # Print avg likelihood of this batch data on previous batch weights
                print("  Avg likelihood (this batch data on previous weights): %s" %
                      (avg_train_likelihood_val))
                likelihoods.append(avg_train_likelihood_val)
                # Update weights
                self._sess.run(self._update_spn)

                # Print weights
                # print(self._sess.run(self._root.weights.node.variable))
                # print(self._sess.run(self._em_learning.root_accum()))

            likelihood = sum(likelihoods) / len(likelihoods)
            print("- Batch avg likelihood: %s" % (likelihood))
            epoch += 1
            
        marginal_vals = self._calculate_marginal_vals(self._data.testing_scans)
        np.save(os.path.join(results_dir, 'semisup', 
                             'testing_marginal_vals_' + str(epoch)), marginal_vals)
        np.save(os.path.join(results_dir, 'semisup', 
                             'testing_labels_' + str(epoch)), self._data.testing_labels)
        max_idx = np.argmax(marginal_vals, axis=1)
        max_idx = max_idx.reshape((len(max_idx), 1))

        # thresh_idx = np.max(marginal_vals, axis=1) > threshold
        thresh_idx = np.zeros((len(max_idx)), dtype=bool)
        for i in range(len(max_idx)):
            thresh_idx[i] = \
                (np.max(marginal_vals[i][:]) > threshold[max_idx[i][0]])
        kc_idx = (self._data.testing_labels != len(self._place_classes)).squeeze()
        filt_idx = np.logical_and(thresh_idx, kc_idx)

        result = (max_idx[kc_idx] == max_idx[kc_idx])
        print(np.sum(result)/len(result))

        result = (max_idx[filt_idx] == max_idx[filt_idx])
        print(float(np.sum(result))/len(result))

        print("Saving the semisup trained model")
        spn.JSONSaver(os.path.join(results_dir, 'semisup/trained.spn'), pretty=True) \
           .save(self._root)

        print("Done!")

    def test(self, results_dir):
        # MPE State - Moved from build_model to not speed up learning
        # TODO: Can we re-use the MPEPath from EMLearning?
        print("\nAdding testing ops:")
        self._mpe_state_gen = spn.MPEState(
            log=True, value_inference_type=spn.InferenceType.MPE)
        self._mpe_ivs, self._mpe_latent = \
            self._mpe_state_gen.get_state(self._root, self._ivs, self._latent)
        print("Done!")

        np.set_printoptions(linewidth=200, precision=2,
                            threshold=np.nan, suppress=True)

        # print("\nCalculating MPE state:")
        # # MPE state of the whole network
        # mpe_ivs_val, = self._sess.run(
        #     [self._mpe_ivs],
        #     feed_dict={self._ivs:
        #                np.ones((1, self._num_vars), dtype=int) * -1})
        # print("- Saving...")
        # self._data.plot_polar_scan(mpe_ivs_val[0],
        #                            os.path.join(results_dir, 'mpe_state.png'))
        # print("Done!")

        # print("\nCalculating all MPE vals:")
        # all_mpe_vals, = self._sess.run(
        #     [self._mpe_state_gen.mpe_path.value.values[self._root]],
        #     feed_dict={self._ivs: self._data.all_scans})
        # print("- Saving...")
        # np.save(os.path.join(results_dir, 'mpe_vals'), all_mpe_vals)
        # print("Done!")

        # print("\nCalculating all marginal vals:")
        # # Test if EM is using marginal value during upwards pass
        # if self._value_inference_type != spn.InferenceType.MARGINAL:
        #     raise Exception("EM is not using MARGINAL value inference type!"
        #                     " Calculating marginal values cannot be done"
        #                     " with the EM ops!")
        # all_marginal_vals, = self._sess.run(
        #     [self._em_learning.mpe_path.value.values[self._root]],
        #     feed_dict={self._ivs: self._data.all_scans,
        #                self._latent: self._data.all_labels})
        # print("- Saving...")
        # np.save(os.path.join(results_dir, 'marginal_vals'), all_marginal_vals)
        # print("Done!")

        # print("\nCompleting masked scans:")
        # completion_batch_size = 19  # set is divided by 19
        # if (len(self._data.masked_scans) % completion_batch_size) != 0:
        #     raise Exception("Incorrect completion batch size!")
        # filled_scans = []
        # for i in range(0, len(self._data.masked_scans), completion_batch_size):
        #     print("- BATCH for samples %s-%s/%s" %
        #           (i + 1, i + completion_batch_size, len(self._data.masked_scans)))
        #     filled_scan, = self._sess.run(
        #         [self._mpe_ivs],
        #         feed_dict={
        #             self._ivs:
        #             self._data.masked_scans[i:(i + completion_batch_size), :]})
        #     filled_scans.append(filled_scan)
        # print("- Saving...")
        # filled_scans = np.vstack(filled_scans)
        # np.save(os.path.join(results_dir, 'filled_scans'), filled_scans)
        # print("Done!")

        # print("\nCalculating MPE vals for masked scans:")
        # masked_scans_mpe_vals = []
        # for i in range(0, len(self._data.masked_scans), completion_batch_size):
        #     print("- BATCH for samples %s-%s/%s" %
        #           (i + 1, i + completion_batch_size, len(self._data.masked_scans)))
        #     masked_scans_mpe_val, = self._sess.run(
        #         [self._mpe_state_gen.mpe_path.value.values[self._root]],
        #         feed_dict={
        #             self._ivs:
        #             self._data.masked_scans[i:(i + completion_batch_size), :]})
        #     masked_scans_mpe_vals.append(masked_scans_mpe_val)
        # print("- Saving...")
        # masked_scans_mpe_vals = np.vstack(masked_scans_mpe_vals)
        # np.save(os.path.join(results_dir, 'masked_scans_mpe_vals'),
        #         masked_scans_mpe_vals)
        # print("Done!")

        # print("\nCalculating marginal vals for masked scans:")
        # masked_scans_marginal_vals = []
        # for i in range(0, len(self._data.masked_scans), 19):
        #     print("- BATCH for samples %s-%s/%s" %
        #           (i + 1, i + completion_batch_size, len(self._data.masked_scans)))
        #     masked_scans_marginal_val, = self._sess.run(
        #         [self._em_learning.mpe_path.value.values[self._root]],
        #         feed_dict={
        #             self._ivs:
        #             self._data.masked_scans[i:(i + completion_batch_size), :]})
        #     masked_scans_marginal_vals.append(masked_scans_marginal_val)
        # print("- Saving...")
        # masked_scans_marginal_vals = np.vstack(masked_scans_marginal_vals)
        # np.save(os.path.join(results_dir, 'masked_scans_marginal_vals'),
        #         masked_scans_marginal_vals)
        # print("Done!")

        print("\nCalculating marginal vals of all classes for all scans:")
        marginal_vals = self._calculate_marginal_vals(self._data.all_scans)
        labels = self._data.all_labels
        print("- Saving...")
        np.save(os.path.join(results_dir, 'all_marginal_vals'), marginal_vals)
        np.save(os.path.join(results_dir, 'all_labels'), labels)
        print("Done!")

        print("\nCalculating marginal vals of all classes for training scans:")
        marginal_vals = self._calculate_marginal_vals(self._data.training_scans)
        labels = self._data.training_labels
        print("- Saving...")
        np.save(os.path.join(results_dir, 'training_marginal_vals'), marginal_vals)
        np.save(os.path.join(results_dir, 'training_labels'), labels)
        print("Done!")

        print("\nCalculating marginal vals of all classes for testing scans:")
        marginal_vals = self._calculate_marginal_vals(self._data.testing_scans)
        labels = self._data.testing_labels
        print("- Saving...")
        np.save(os.path.join(results_dir, 'testing_marginal_vals'), marginal_vals)
        np.save(os.path.join(results_dir, 'testing_labels'), labels)
        print("Done!")

        print("\nCalculating correct classification ratio for training scans:")
        kc_idx = (self._data.training_labels != len(self._place_classes)).squeeze()
        mpe_latent_val = self._sess.run(
                [self._mpe_latent],
                feed_dict={self._ivs: self._data.training_scans[kc_idx],
                           self._latent: np.ones((np.sum(kc_idx), 1))*-1})
        result = (mpe_latent_val == self._data.training_labels[kc_idx])
        print(np.sum(result)/np.sum(kc_idx))
        print("Done!")

        print("\nCalculating correct classification ratio for testing scans:")
        kc_idx = (self._data.testing_labels != len(self._place_classes)).squeeze()
        mpe_latent_val = self._sess.run(
                [self._mpe_latent],
                feed_dict={self._ivs: self._data.testing_scans[kc_idx],
                           self._latent: np.ones((np.sum(kc_idx), 1))*-1})
        result = (mpe_latent_val == self._data.testing_labels[kc_idx])
        print(np.sum(result)/np.sum(kc_idx))
        print("Done!")
    
        # print("\nCalculating probability distribution for each place class:")
        # norm = marginal_vals - np.max(marginal_vals, axis=1)[:, None]
        # print(np.exp(norm)/np.sum(np.exp(norm), axis=1)[:, None])
        # print("Done!")

    def _calculate_marginal_vals(self, scans, num_batches=4):

        # Test if EM is using marginal value during upwards pass
        if self._value_inference_type != spn.InferenceType.MARGINAL:
            raise Exception("EM is not using MARGINAL value inference type!"
                            " Calculating marginal values cannot be done"
                            " with the EM ops!")

        batch_size = scans.shape[0] // num_batches

        marginal_vals = np.zeros((scans.shape[0], len(self._place_classes)),
                                 dtype=np.float)
        for i in range(len(self._place_classes)):
            for batch in range(num_batches):
                start = (batch) * batch_size
                stop = (batch + 1) * batch_size
                marginal_vals_col, = self._sess.run(
                        [self._em_learning.mpe_path.value.values[self._root]],
                        feed_dict={
                            self._ivs: scans[start:stop, :],
                            self._latent: np.ones((batch_size, 1))*i})
                marginal_vals[start:stop, i] = \
                        np.array(marginal_vals_col).squeeze()

        return marginal_vals


class PlaceSubModel:
    """SPN place sub-model for a single class."""

    class WeightInitValue(Enum):
        ONE = 1
        RANDOM = 2

    def __init__(self, data, ivs,
                 view_input_dist, view_num_decomps, view_num_subsets,
                 view_num_mixtures, view_num_input_mixtures,
                 view_num_top_mixtures,
                 place_num_decomps, place_num_subsets, place_num_mixtures,
                 place_num_input_mixtures,
                 weight_init_value, init_accum_value,
                 additive_smoothing_value, value_inference_type):
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
        self._init_accum_value = init_accum_value
        self._additive_smoothing_value = additive_smoothing_value
        self._value_inference_type = value_inference_type
        self._ivs = ivs

        self._build_model()
        # self._sess = tf.Session()

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

        # Generate random SPNs for each view
        num_views = 8
        view_borders = [0] + [(i + 1) * (self._num_angle_cells // num_views)
                              for i in range(num_views)]
        if view_borders[-1] != self._num_angle_cells:
            raise Exception()
        print("* View borders %s for %s angle cells and %s views" % (
            view_borders, self._num_angle_cells, num_views))

        view_dense_gen = spn.DenseSPNGenerator(
            num_decomps=self._view_num_decomps,
            num_subsets=self._view_num_subsets,
            num_mixtures=self._view_num_mixtures,
            num_input_mixtures=self._view_num_input_mixtures,
            balanced=True,
            input_dist=self._view_input_dist)

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
        place_dense_gen = spn.DenseSPNGenerator(
            num_decomps=self._place_num_decomps,
            num_subsets=self._place_num_subsets,
            num_mixtures=self._place_num_mixtures,
            num_input_mixtures=self._place_num_input_mixtures,
            balanced=True,
            input_dist=spn.DenseSPNGenerator.InputDist.RAW)
        self._root = place_dense_gen.generate(*[i for v in view_sums for i in v])

