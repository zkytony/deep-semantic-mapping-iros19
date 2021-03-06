# Single DGSM for all classesst
from enum import Enum
import libspn as spn
import numpy as np
import tensorflow as tf
from deepsm.util import CategoryManager
from deepsm.dgsm.data import Data
import pprint as pp
import os, sys
import json, yaml, csv
import random
import time
from sklearn import metrics
spn.conf.renormalize_dropconnect = True
spn.conf.rescale_dropconnect = True
# spn.config_logger(spn.DEBUG1)

def norm_cm(cm):
    return cm / np.sum(cm, axis=1, keepdims=True)


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
                 learning_rate,
                 init_accum_val,
                 smoothing_val, smoothing_min, smoothing_decay,
                 dropconnect_keep_prob=-1,
                 learning_algorithm=spn.GDLearning,
                 optimizer=tf.train.AdamOptimizer,
                 learning_type=spn.LearningTaskType.SUPERVISED,
                 learning_method=spn.LearningMethodType.DISCRIMINATIVE,
                 gradient_type=spn.GradientType.SOFT):
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
        self._learning_algorithm = learning_algorithm
        self._learning_type = learning_type
        self._learning_method = learning_method
        self._gradient_type = gradient_type
        self._dropconnect_keep_prob = dropconnect_keep_prob

        self._build_model()
        self._sess = tf.Session()

        self._known_classes = CategoryManager.known_categories()

    @staticmethod
    def get_ivs_indices(num_vals, start_var_index, end_var_index):
        """start_var_index inclusive, end_var_index exclusive"""
        return np.arange(start_var_index * num_vals,
                         end_var_index * num_vals).tolist()

    def build_sub_spn(self, gen_view, gen_place):
        print("Building sub-SPN ...")

        # Generate lower level SPN
        num_views = 8
        num_vars_per_view = self._ivs.num_vars // num_views
        roots_view = []
        for gi in range(num_views):
            roots_view.append(
                gen_view.generate((self._ivs,
                                    PlaceModel.get_ivs_indices(self._ivs.num_vals,
                                                               gi*num_vars_per_view,
                                                               (gi+1)*num_vars_per_view))))

        # Obtain product nodes instead of roots
        products_view = [None] * len(roots_view)
        for ri in range(len(roots_view)):
            products_view[ri] = [v.node for v in roots_view[ri].values]


        # Create sums for each view
        sums_view = [None] * len(roots_view)
        for ri in range(len(roots_view)):
            prods = products_view[ri]
            sums_view[ri] = spn.ParSums(*prods, num_sums=self._view_num_top_mixtures)

        # Generate Upper Level SPN
        root = gen_place.generate(*[sums for sums in sums_view])

        # Check if valid
        print("Is valid? %s" % root.is_valid())
        return root


    def _build_model(self):

        # IVS
        if self._occupancy_vals == Data.OccupancyVals.THREE:
            self._ivs = spn.IVs(num_vars=self._num_radius_cells*self._num_angle_cells, num_vals=3)
        elif self._occupancy_vals == Data.OccupancyVals.TWO:
            self._ivs = spn.IVs(num_vars=self._num_radius_cells*self._num_angle_cells, num_vals=2)
        else:
            raise Exception("Incorrect value of occupancy vals.")

        # Type of inference during upward pass of learning
        value_inference_type = self._value_inference_type

        # Create two generators
        gen_view = \
            spn.DenseSPNGeneratorLayerNodes(
                num_decomps=self._view_num_decomps,
                num_subsets=self._view_num_subsets,
                num_mixtures=self._view_num_mixtures,
                balanced=True,
                input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW)

        gen_place = \
            spn.DenseSPNGeneratorLayerNodes(
                num_decomps=self._place_num_decomps,
                num_subsets=self._place_num_subsets,
                num_mixtures=self._place_num_mixtures,
                balanced=True,
                input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW)

        # For each class, build a sub-SPN                                                                                                         
        class_roots = []
        for i in range(CategoryManager.NUM_CATEGORIES):
            print("----Class %d (%s)----" % (i, CategoryManager.category_map(i, rev=True)))
            sub_root = self.build_sub_spn(gen_view, gen_place)
            class_roots.append(sub_root)

        print("***** Full Model *****")
        self._root = spn.ParSums(*class_roots)
        self._root = gen_place.convert_to_layer_nodes(self._root)

        # Check if valid
        print("Is valid? %s" % self._root.is_valid())

        # Generate SPN weights
        print("Adding weights...")
        if self._weight_init_value == PlaceModel.WeightInitValue.ONE:
            wiv = 1
        elif self._weight_init_value == PlaceModel.WeightInitValue.RANDOM:
            wiv = spn.ValueType.RANDOM_UNIFORM(10, 11)
        else:
            raise Exception()
        spn.generate_weights(self._root, init_value=wiv, log=True)

        # Generate Ltent IVs for the root
        self._latent = self._root.generate_ivs(name="root_Catg_IVs")

        self._dropconnect_placeholder = tf.placeholder(tf.float32, shape=())

        if self._dropconnect_keep_prob > 0:
            # Change dropconnect keep prob
            print("Changing dropconnect settings to some nodes...")
            self._root.set_dropconnect_keep_prob(1.0)

            count = 0
            for node in self._root.get_nodes(skip_params=True):
                # print(node.name, type(node))

                if isinstance(node, spn.SumsLayer):
                    if isinstance(node.values[0].node, spn.IVs):
                        print("deteceted sum<-iv %s" % node)
                        node.set_dropconnect_keep_prob(1.0)
                    elif any(isinstance(c.node, spn.ProductsLayer) and
                             any(isinstance(cc.node, spn.IVs) for cc in c.node.values)
                             for c in node.values):
                        print("deteceted sum<-product<-iv %s" % node)
                    elif len(node.values) < 200:  # Cannot use dropconnect on low layers
                        print("changed dropconnect to a placeholder", node, len(node.values))
                        node.set_dropconnect_keep_prob(self._dropconnect_placeholder)
                        count += 1
            print(count)


        if self._learning_algorithm == spn.GDLearning:

            # Learning Ops
            self._gd_learning = spn.GDLearning(
                self._root,
                value_inference_type = value_inference_type,
                learning_rate=self._learning_rate,
                learning_task_type=self._learning_type,
                learning_method=self._learning_method,
                gradient_type=self._gradient_type,
                dropconnect_keep_prob=1.0)
            # self._reset_accumulators = self._learning.reset_accumulators()
            self._learn_spn, self._loss_op = self._gd_learning.learn(optimizer=self._optimizer)

            # Op for getting likelihoods. The result is an array of likelihoods produced by sub-SPNs for different classes.
            # self._likelihood_op = self._likelihood_y_given_x() #self._gd_learning._value_gen.get_value(self._root.values[0].node)
            self._likelihood_op = self._likelihood_x_given_y() #self._gd_learning._value_gen.get_value(self._root.values[0].node)

            self._init_weights = spn.initialize_weights(self._root)
            self._init_weights = [self._init_weights, tf.global_variables_initializer()]

        else:
            self._additive_smoothing_var = tf.Variable(0.0,
                                                       dtype=spn.conf.dtype)
            self._em_learning = spn.EMLearning(
                self._root, log=True,
                value_inference_type=self._value_inference_type,
                additive_smoothing=self._additive_smoothing_var,
                add_random=None,
                initial_accum_value=20,
                use_unweighted=True)

            self._init_weights = spn.initialize_weights(self._root)
            self._init_weights = [self._init_weights, tf.global_variables_initializer()]
            self._reset_accumulators = self._em_learning.reset_accumulators()
            self._accumulate_updates = self._em_learning.accumulate_updates()
            self._update_spn = self._em_learning.update_spn()
            self._train_likelihood = self._em_learning.value.values[self._root]
            self._avg_train_likelihood = tf.reduce_mean(self._train_likelihood)
            print("Done!")
        

    def train(self, batch_size, update_threshold,
              train_loss=[], test_loss=[], train_perf=[], test_perf=[],
              shuffle=True, epoch_limit=50):
        """
        Train the model.
        
        batch_size (int): size of each mini-batch.
        update_threshold (float): minimum update required between steps to continue.
        """

        train_set = self._data.training_scans
        train_labels = self._data.training_labels

        self._sess.run(self._init_weights)

        num_batches = train_set.shape[0] // batch_size
        prev_loss = 100
        loss = 0
        losses = []
        batch_losses = []
        batch = 0
        epoch = 0

        print("Start Training...")
        while (self._learning_algorithm == spn.GDLearning \
               and epoch < epoch_limit and abs(prev_loss - loss)>update_threshold) \
               or (self._learning_algorithm == spn.EMLearning \
                   and (abs(prev_loss - loss)>update_threshold)):

            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, train_set.shape[0])
            print("EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop, "  prev loss", prev_loss, "loss", loss)

            if self._learning_algorithm == spn.EMLearning:
                # Adjust smoothing
                ads = max(np.exp(-epoch * additive_smoothing_decay) *
                          self._additive_smoothing_value,
                          additive_smoothing_min)
                self._sess.run(self._additive_smoothing_var.assign(ads))
                print("  Smoothing: ", self._sess.run(self._additive_smoothing_var))
                # Run accumulate_updates
                train_likelihoods_arr, loss_train, _, = \
                    self._sess.run([self._train_likelihood,
                                    self._avg_train_likelihood,
                                    self._accumulate_updates],
                                   feed_dict={self._ivs: train_set[start:stop],
                                              self._latent: train_labels[start:stop],
                                              self._dropconnect_placeholder: self._dropconnect_keep_prob})
                # Print avg likelihood of this batch data on previous batch weights
                print("  Avg likelihood (this batch data on previous weights): %s" %
                      (loss_train))
                # Update weights
                self._sess.run(self._update_spn)

            else:
                _, loss_train, likelihoods = self._sess.run([self._learn_spn, self._loss_op, self._likelihood_op],
                                                            feed_dict={self._ivs: train_set[start:stop],
                                                                       self._latent: train_labels[start:stop],
                                                                       self._dropconnect_placeholder: self._dropconnect_keep_prob})

            batch_losses.append(loss_train)
            
            batch += 1
            if stop >= train_set.shape[0]:
                epoch += 1
                batch = 0

                # Shuffle
                if shuffle:
                    print("Shuffling...")
                    p = np.random.permutation(len(train_set))
                    train_set = train_set[p]
                    train_labels = train_labels[p]
                    
                print("Computing train, (test) losses...")
                loss_train = np.mean(batch_losses)
                train_loss.append(loss_train)
                print("Train Loss: %.3f  " % loss_train)
                if self._learning_algorithm == spn.GDLearning:
                    loss_test = self.log_loss(self._data.testing_scans, self._data.testing_labels)
                    test_loss.append(loss_test)
                    print("Test Loss: %.3f  " % loss_test)

                print("Computing train performance...")
                perf_train, _ = self.performance(train_set, train_labels, batch_size=500)
                print("Computing test performance...")
                perf_test, time_taken = self.performance(self._data.testing_scans, self._data.testing_labels, batch_size=100)
                print("Average time taken to evaluate each testing scan: %.5f" % (time_taken))
                
                for k in range(CategoryManager.NUM_CATEGORIES):
                    train_perf[k].append(perf_train[k])
                    test_perf[k].append(perf_test[k])

                losses.append(loss_train)
                prev_loss = loss
                loss = loss_train
                batch_losses = []

        return losses


    def mpe_states(self):
        """After the model is trained, return a list of mpe state (images) that are
        the model's internal representation of the class geometries. The list is
        ordered accroding to the indices of classes in CategoryManager."""
        print("Generating MPE State Ops...")
        mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
        mpe_ivs, mpe_latent = mpe_state_gen.get_state(self._root, self._ivs, self._latent)

        num_vars = self._ivs.num_vars
        num_vals = self._ivs.num_vals

        print("Performing MPE Inference for classes:")
        results = []
        for i in range(CategoryManager.NUM_CATEGORIES):
            print("  %d: %s" % (i, CategoryManager.category_map(i, rev=True)))
            mpe_ivs_val = self._sess.run([mpe_ivs],
                                         feed_dict={self._ivs: np.full((1, num_vars), -1),
                                                    self._latent: np.array([[i]])})
            results.append(mpe_ivs_val[0][0])
        return results

            
    def test(self, results_dir, batch_size=50, graph_test=True, last_batch=True):
        # Make numpy array of test samples
        testing_scans = self._data.testing_scans
        testing_labels = self._data.testing_labels

        num_batches = testing_scans.shape[0] // batch_size
        accuracy_per_step = []
        likelihoods = np.empty((0, CategoryManager.NUM_CATEGORIES))
        last_batch = True
        for batch in range(num_batches):
            start = (batch) * batch_size
            if last_batch and (batch + 1) == num_batches:
                stop = testing_scans.shape[0]
            else:
                stop = (batch + 1) * batch_size

            # Get output from sub-SPNs as likelihoods for each class. Likelihood for class i is representing P(X|Y=i)
            likelihoods_arr = self._sess.run([self._likelihood_op],
                                             feed_dict={self._ivs: testing_scans[start:stop],
                                                        self._latent: np.full((stop-start, 1), -1),
                                                        self._dropconnect_placeholder: 1.0})[0]
            likelihoods = np.vstack((likelihoods, likelihoods_arr))

        # Process graph results
        
        # Confusion matrices
        cm_weighted = np.zeros((CategoryManager.NUM_CATEGORIES, CategoryManager.NUM_CATEGORIES))
        
        graph_results = {}
        # likelihoods = np.transpose(np.array(likelihoods_per_step, dtype=float))
        for i, d in enumerate(self._data.testing_footprint):
            rid = d[0]
            rcat = d[1]
            if rid not in graph_results:  # rid is actually graph_id
                graph_results[rid] = {}

            true_class_index = CategoryManager.category_map(rcat)
            pred_class_index = np.argmax(likelihoods[i])
                
            # Record in confusion matrix
            cm_weighted[true_class_index, pred_class_index] += 1

            if graph_test:
                node_id = d[-1]
                graph_results[rid][node_id] = [rcat, self._known_classes[pred_class_index], list(likelihoods[i]), list(likelihoods[i])]

        # Confusion matrix
        print("- Confusion matrix for MPE (weighted):")
        pp.pprint(self._known_classes)
        pp.pprint(cm_weighted)
        pp.pprint(norm_cm(cm_weighted) * 100.0)

        # ROC curves
        roc_results = []
        root_weights = np.log(self._sess.run(self._root.weights.node.get_value()))[0]
        for rcat_num in range(CategoryManager.NUM_CATEGORIES):
            fpr, tpr, auc = self._roc(likelihoods, root_weights, testing_labels, rcat_num)
            roc_results.append((fpr, tpr))

        if graph_test:
            # Save
            os.makedirs(os.path.join(results_dir), exist_ok=True)
            for graph_id in graph_results:
                with open(os.path.join(results_dir, graph_id + "_likelihoods.json"), 'w') as f:
                    json.dump(graph_results[graph_id], f)

            # Overall statistics
            stats = self._compute_stats(graph_results)
            print("- Overall statistics")
            pp.pprint(stats)
            with open(os.path.join(results_dir, "results.yaml"), 'w') as f:
                yaml.dump(stats, f)
        
            return cm_weighted, norm_cm(cm_weighted) * 100.0, stats, roc_results


    def performance(self, scans, labels, batch_size=100):
        correct = np.array([0 for i in range(CategoryManager.NUM_CATEGORIES)])
        total = np.array([0 for i in range(CategoryManager.NUM_CATEGORIES)])
        batch = 0
        stop = min(batch_size, len(scans))
        time_taken = []
        while stop < len(scans):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, len(scans))
            print("    BATCH", batch, "SAMPLES", start, stop)

            start_time = time.time()
            likelihoods_arr = self._sess.run([self._likelihood_op],
                                             feed_dict={self._ivs: scans[start:stop],
                                                        self._latent: np.full((stop-start, 1), -1),
                                                        self._dropconnect_placeholder: 1.0})[0]
            time_taken.append((time.time() - start_time) / float(stop - start))
            batch += 1

            for i in range(len(likelihoods_arr)):
                pred = np.argmax(likelihoods_arr[i])
                gt = labels[start+i][0]
                if pred == gt:
                    correct[gt] += 1
                total[gt] += 1

        print(correct / total)

        return correct / total, np.mean(time_taken)
    

    def samples_exam(self, scans, labels, writer, write_row_func, batch_size=100,
                     per_lh_func=None, per_lh_func_kwargs={}):
        batch = 0
        stop = min(batch_size, len(scans))
        while stop < len(scans):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, len(scans))
            print("    BATCH", batch, "SAMPLES", start, stop)

            likelihoods_arr = self._sess.run([self._likelihood_op],
                                             feed_dict={self._ivs: scans[start:stop],
                                                        self._latent: np.full((stop-start, 1), -1),
                                                        self._dropconnect_placeholder: 1.0})[0]
            for i in range(len(likelihoods_arr)):
                write_row_func(writer, i, start, scans, likelihoods_arr, labels)
                if per_lh_func is not None:
                    per_lh_func(start, i, likelihoods_arr, **per_lh_func_kwargs)

            batch += 1
            
    def train_samples_exam(self, dirpath, trial_name):
        def write_row_func(writer, i, start, scans, likelihoods_arr, labels):
            pred = CategoryManager.category_map(np.argmax(likelihoods_arr[i]), rev=True)
            gt = CategoryManager.category_map(labels[start+i][0], rev=True)
            writer.writerow([self._data._training_footprint_graph[start+i][0]]  # graph_id
                            + [self._data._training_footprint_graph[start+i][-1]]  # node id
                            + [gt]    # groundtruth label string
                            + [pred]  # prediction label string
                            + likelihoods_arr[i].reshape(-1,).tolist() 
                            + scans[start+i].tolist())

        def per_lh_func(start, i, likelihoods_arr, lh_map={}):
            nid = self._data._training_footprint_graph[start+i][-1]
            likelihoods = likelihoods_arr[i].tolist()
            graph_id = self._data._training_footprint_graph[start+i][0]
            if graph_id not in lh_map:
                lh_map[graph_id] = {}
            lh_map[graph_id][nid] = likelihoods

        print("Writing train samples exam file...")
        train_graph_likelihoods = {}  # graph_id to nid to likelihoods array
        with open(os.path.join(dirpath, "train_samples_exam-%s.csv" % trial_name), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            self.samples_exam(self._data._training_scans_graph,
                              self._data._training_labels_graph, writer, write_row_func,
                              batch_size=200, per_lh_func=per_lh_func, per_lh_func_kwargs={'lh_map':train_graph_likelihoods})
        return train_graph_likelihoods
            

    
    def test_samples_exam(self, dirpath, trial_name):
        def write_row_func(writer, i, start, scans, likelihoods_arr, labels):
            pred = CategoryManager.category_map(np.argmax(likelihoods_arr[i]), rev=True)
            gt = CategoryManager.category_map(labels[start+i][0], rev=True)
            writer.writerow([self._data._testing_footprint[start+i][0]]  # graph_id
                            + [self._data._testing_footprint[start+i][-1]]  # node id
                            + [gt]    # groundtruth label string
                            + [pred]  # prediction label string
                            + likelihoods_arr[i].reshape(-1,).tolist() 
                            + scans[start+i].tolist())
        
        print("Writing test samples exam file...")
        with open(os.path.join(dirpath, "test_samples_exam-%s.csv" % trial_name), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            self.samples_exam(self._data._testing_scans, self._data._testing_labels, writer,
                              write_row_func)


    def log_loss(self, samples, labels, batch_size=100):
        batch = 0
        loss_values = []
        stop = min(batch_size, len(samples))
        while stop < len(samples):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, len(samples))
            print("    BATCH", batch, "SAMPLES", start, stop)

            loss_val = self._sess.run([self._loss_op],
                                      feed_dict={self._ivs: samples[start:stop],
                                                 self._latent: labels[start:stop],
                                                 self._dropconnect_placeholder: 1.0})

            loss_values.append(loss_val)
            batch += 1
        return np.mean(loss_values)


    def _roc(self, likelihoods, root_weights, labels, room_class_num):
        """
        room_class_num is an integer representation for a class.
        """
        y_true = np.copy(labels)
        y_true[labels == room_class_num] = 1
        y_true[labels != room_class_num] = 0
        y_true = y_true.reshape(-1,)
        
        # P(X) = sum_Y P(X|Y)P(Y)
        denominator = np.log(np.sum(np.exp((likelihoods + root_weights) - np.max(likelihoods)), axis=1)) + np.max(likelihoods)
        # P(Y=room_class_num|X)
        y_dist = np.array([(likelihoods[i][room_class_num] + root_weights[room_class_num] - denominator[i])
                           for i in range(len(likelihoods))])
        y_score = np.array([np.exp(y_dist[i])
                            for i in range(len(likelihoods))])
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        return fpr, tpr, metrics.auc(fpr, tpr)


    def _likelihood_y_given_x(self):
        """Follows the implementation in libspn/learning/gd.py"""
        # p(x|y)
        log_likelihood_x_given_y = spn.inference.value.LogValue(dropconnect_keep_prob=self._gd_learning._dropconnect_keep_prob,
                                                                dropprod_keep_prob=self._gd_learning._dropprod_keep_prob,
                                                                noise=self._gd_learning._noise,
                                                                batch_noise=self._gd_learning._batch_noise).get_value(self._root.values[0].node)
        # p(y)
        log_likelihood_y = self._root.weights.node.get_value()
        # p(x,y) = p(x|y)p(y)
        log_likelihood_x_and_y = log_likelihood_x_given_y + log_likelihood_y
        # p(x)
        log_likelihood_x = self._root.get_value()
        
        # p(y|x) = p(x,y)/p(x)
        return log_likelihood_x_and_y - log_likelihood_x

    
    def _likelihood_x_given_y(self):
        """Follows the implementation in libspn/learning/gd.py"""
        # p(x|y)
        log_likelihood_x_given_y = spn.inference.value.LogValue(dropconnect_keep_prob=self._gd_learning._dropconnect_keep_prob,
                                                                dropprod_keep_prob=self._gd_learning._dropprod_keep_prob,
                                                                noise=self._gd_learning._noise,
                                                                batch_noise=self._gd_learning._batch_noise).get_value(self._root.values[0].node)
        return log_likelihood_x_given_y

    def _compute_stats(self, graph_results):
        stats = {}
        total_cases = 0
        total_correct = 0
        total_correct_top2 = 0
        total_correct_top3 = 0
        total_per_class = {}
        total_per_class_top2 = {}
        total_per_class_top3 = {}
        for graph_id in graph_results:
            graph_cases = 0
            graph_correct = 0
            graph_correct_top2 = 0
            graph_correct_top3 = 0
            graph_per_class = {}
            graph_per_class_top2 = {}
            graph_per_class_top3 = {}
            for nid in graph_results[graph_id]:
                groundtruth = CategoryManager.canonical_category(graph_results[graph_id][nid][0])
                # We skip unknown cases
                if groundtruth not in self._known_classes:
                    continue
                if groundtruth not in total_per_class:
                    total_per_class[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                    total_per_class_top2[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                    total_per_class_top3[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                if groundtruth not in graph_per_class:
                    graph_per_class[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                    graph_per_class_top2[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                    graph_per_class_top3[groundtruth] = [0, 0, 0]  # total cases, correct cases, accuracy
                
                # We have one more test case
                total_cases += 1
                graph_cases += 1
                total_per_class[groundtruth][0] += 1
                graph_per_class[groundtruth][0] += 1
                total_per_class_top2[groundtruth][0] += 1
                graph_per_class_top2[groundtruth][0] += 1
                total_per_class_top3[groundtruth][0] += 1
                graph_per_class_top3[groundtruth][0] += 1

                prediction_ranking = [CategoryManager.category_map(c, rev=True)
                                      for c in np.argsort(graph_results[graph_id][nid][2])[::-1]]
                in_top_pred = groundtruth == prediction_ranking[0]
                in_top2_pred = groundtruth in prediction_ranking[:2]
                in_top3_pred = groundtruth in prediction_ranking[:3]
                
                if in_top_pred:
                    graph_correct += 1
                    total_correct += 1
                    total_per_class[groundtruth][1] += 1
                    graph_per_class[groundtruth][1] += 1
                if in_top2_pred:
                    graph_correct_top2 += 1
                    total_correct_top2 += 1
                    total_per_class_top2[groundtruth][1] += 1
                    graph_per_class_top2[groundtruth][1] += 1
                if in_top3_pred:
                    graph_correct_top3 += 1
                    total_correct_top3 += 1
                    total_per_class_top3[groundtruth][1] += 1
                    graph_per_class_top3[groundtruth][1] += 1
                    
                total_per_class[groundtruth][2] = total_per_class[groundtruth][1] / total_per_class[groundtruth][0]
                total_per_class_top2[groundtruth][2] = total_per_class_top2[groundtruth][1] / total_per_class_top2[groundtruth][0]
                total_per_class_top3[groundtruth][2] = total_per_class_top3[groundtruth][1] / total_per_class_top3[groundtruth][0]
                graph_per_class[groundtruth][2] = graph_per_class[groundtruth][1] / graph_per_class[groundtruth][0]
                graph_per_class_top2[groundtruth][2] = graph_per_class_top2[groundtruth][1] / graph_per_class_top2[groundtruth][0]
                graph_per_class_top3[groundtruth][2] = graph_per_class_top3[groundtruth][1] / graph_per_class_top3[groundtruth][0]
            
            stats[graph_id] = {'num_cases': graph_cases,
                               'num_correct': graph_correct,
                               'accuracy': graph_correct / graph_cases,
                               'accuracy_top2': graph_correct_top2 / graph_cases,
                               'accuracy_top3': graph_correct_top3 / graph_cases,
                               'class_results': graph_per_class,
                               'class_results_top2': graph_per_class_top2,
                               'class_results_top3': graph_per_class_top3}
        stats.update({'num_cases': total_cases,
                      'num_correct': total_correct,
                      'accuracy': total_correct / total_cases,
                      'accuracy_top2': total_correct_top2 / total_cases,
                      'accuracy_top3': total_correct_top3 / total_cases,
                      'class_results': total_per_class,
                      'class_results_top2': total_per_class_top2,
                      'class_results_top3': total_per_class_top3})
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
