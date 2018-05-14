import os.path
import numpy as np
import pprint
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import json
import deepsm.util as util  # TODO: REMOVE

pp = pprint.PrettyPrinter(indent=4)


def norm_cm(cm):
    return cm / np.sum(cm, axis=1, keepdims=True)


def get_class_rate(cm):
    ncm = norm_cm(cm)
    d = ncm.diagonal()
    return np.mean(d)

def interpret(likelihoods):
    """Intepret likelihoods. Returns the vector of
    normalized marginal likelihoods"""
    marginals = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                       (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return marginals


class SubModelResult:

    def __init__(self, results_dir, submodel_class):
        self._submodel_class = submodel_class
        final_dir = os.path.join(results_dir, submodel_class)
        # Load
        print("* Loading submodel result for class: %s" % submodel_class)
        self._marginal_vals = np.load(os.path.join(
            final_dir, "marginal_vals.npy"))
        self._mpe_vals = np.load(os.path.join(
            final_dir, "mpe_vals.npy"))
        # NOTE: We ignore the following results for now.
        # self._masked_scans = np.load(os.path.join(
        #     final_dir, "masked_scans.npy"))
        # self._filled_scans = np.load(os.path.join(
        #     final_dir, "filled_scans.npy"))
        # self._masked_scans_marginal = np.load(os.path.join(
        #     final_dir, "masked_scans_marginal_vals.npy"))
        # self._masked_scans_mpe = np.load(os.path.join(
        #     final_dir, "masked_scans_mpe_vals.npy"))

    @property
    def submodel_class(self):
        return self._submodel_class

    @property
    def mpe_vals(self):
        return self._mpe_vals

    @property
    def marginal_vals(self):
        return self._marginal_vals

    @property
    def filled_scans(self):
        return self._filled_scans

    @property
    def masked_scans(self):
        return self._masked_scans

    @property
    def masked_scans_marginal(self):
        return self._masked_scans_marginal

    @property
    def masked_scans_mpe(self):
        return self._masked_scans_mpe


class Results:

    def __init__(self, datas, submodel_results, results_dir):
        self._datas = datas
        self._submodel_results = submodel_results
        self._results_dir = results_dir
        # Check if data, results match
        for d, r in zip(datas, submodel_results):
            if d.submodel_class != r.submodel_class:
                raise Exception("Data and results do not match!")
        # Check if data subset is the same
        for d in datas:
            if datas[0].subset != d.subset:
                raise Exception("Datas have different subset")
        # NOTE: We do not use masked scans for now
        # Check if masked scans for all classes are the same
        # for r in self._submodel_results:
        #     if np.sum(self._submodel_results[0].masked_scans != r.masked_scans) > 0:
        #         raise Exception("Masked scans do not match across classes")
        # Extract some info from results
        self._num_classes = len(submodel_results)
        self._known_classes = [r.submodel_class for r in submodel_results]
        # Calculate class weights
        self._calculate_class_weights()

    def _calculate_class_weights(self):
        tr_data_nums = np.array([len(d.training_scans) for d in self._datas])
        self._class_weights = tr_data_nums / np.sum(tr_data_nums)
        np.set_printoptions(threshold=np.nan)
        print("- Class weights: %s" % self._class_weights)

    def _get_scan_completion_rate(self, masked_scan, true_scan, filled_scan):
        num_all_pixels = true_scan.size
        num_masked_pixels = np.sum(true_scan != masked_scan)
        if num_masked_pixels != num_all_pixels // 4:
            raise Exception()
        num_reconstructed_pixels = np.sum(filled_scan == true_scan) - \
            (num_all_pixels - num_masked_pixels)
        # Alternative way of calculating the same:
        # num_reconstructed_pixels = np.sum((filled_scan == true_scan) &
        #                                   (masked_scan == -1))
        cr = num_reconstructed_pixels / num_masked_pixels
        return cr

    def get_completion_ratios(self):
        # Get first data as representative of all data
        data = self._datas[0].data
        all_scans = self._datas[0].all_scans
        test_rooms = self._datas[0].test_rooms

        # Crete filled scan folders
        # os.makedirs(os.path.join(self._results_dir, 'filled_scans_marginal'),
        # exist_ok=True)
        os.makedirs(os.path.join(self._results_dir, 'filled_scans_marginal_weighted'),
                    exist_ok=True)
        # os.makedirs(os.path.join(self._results_dir, 'filled_scans_mpe'),
        # exist_ok=True)
        os.makedirs(os.path.join(self._results_dir, 'filled_scans_mpe_weighted'),
                    exist_ok=True)

        # Calculate
        crs_marginal = []
        crs_marginal_weighted = []
        crs_mpe = []
        crs_mpe_weighted = []
        cm_marginal = np.zeros((self._num_classes, self._num_classes))
        cm_mpe = np.zeros((self._num_classes, self._num_classes))
        cm_marginal_weighted = np.zeros((self._num_classes, self._num_classes))
        cm_mpe_weighted = np.zeros((self._num_classes, self._num_classes))
        plot_data = self._datas[0].plot_polar_scan_setup()
        for i, d in enumerate(data):
            rid = d[0]
            rclass = d[1]
            # Test sample?
            if rid in test_rooms:
                rclass_id = self._known_classes.index(rclass)
                # Get values for masked scan from each submodel
                marginal_vals = np.array([float(r.masked_scans_marginal[i])
                                          for r in self._submodel_results])
                mpe_vals = np.array([float(r.masked_scans_mpe[i])
                                     for r in self._submodel_results])
                # print(mpe_vals)

                # Get max class
                max_class_marginal = np.argmax(marginal_vals)
                max_class_marginal_weighted = np.argmax(marginal_vals +
                                                        np.log(self._class_weights))
                max_class_mpe = np.argmax(mpe_vals)
                max_class_mpe_weighted = np.argmax(mpe_vals +
                                                   np.log(self._class_weights))

                # Get confusion matrix
                cm_marginal[rclass_id, max_class_marginal] += 1
                cm_marginal_weighted[rclass_id, max_class_marginal_weighted] += 1
                cm_mpe[rclass_id, max_class_mpe] += 1
                cm_mpe_weighted[rclass_id, max_class_mpe_weighted] += 1

                # Compare true and filled scans
                cr_marginal = (
                    self._get_scan_completion_rate(
                        masked_scan=self._submodel_results[
                            max_class_marginal].masked_scans[i],
                        true_scan=all_scans[i],
                        filled_scan=self._submodel_results[
                            max_class_marginal].filled_scans[i]))
                cr_marginal_weighted = (
                    self._get_scan_completion_rate(
                        masked_scan=self._submodel_results[
                            max_class_marginal_weighted].masked_scans[i],
                        true_scan=all_scans[i],
                        filled_scan=self._submodel_results[
                            max_class_marginal_weighted].filled_scans[i]))
                cr_mpe = (
                    self._get_scan_completion_rate(
                        masked_scan=self._submodel_results[
                            max_class_mpe].masked_scans[i],
                        true_scan=all_scans[i],
                        filled_scan=self._submodel_results[
                            max_class_mpe].filled_scans[i]))
                cr_mpe_weighted = (
                    self._get_scan_completion_rate(
                        masked_scan=self._submodel_results[
                            max_class_mpe_weighted].masked_scans[i],
                        true_scan=all_scans[i],
                        filled_scan=self._submodel_results[
                            max_class_mpe_weighted].filled_scans[i]))
                crs_marginal.append(cr_marginal)
                crs_marginal_weighted.append(cr_marginal_weighted)
                crs_mpe.append(cr_mpe)
                crs_mpe_weighted.append(cr_mpe_weighted)

                # Save filled scans
                # self._datas[0].plot_polar_scan_plot(
                #     plot_data,
                #     self._submodel_results[max_class_marginal].filled_scans[i],
                #     os.path.join(self._results_dir, 'filled_scans_marginal',
                #                  "%04d-%s-%s.png" % (i, rid, rclass)))
                self._datas[0].plot_polar_scan_plot(
                    plot_data,
                    self._submodel_results[max_class_marginal_weighted].filled_scans[i],
                    os.path.join(self._results_dir, 'filled_scans_marginal_weighted',
                                 "%04d-%s-%s.png" % (i, rid, rclass)))
                # self._datas[0].plot_polar_scan_plot(
                #     plot_data,
                #     self._submodel_results[max_class_mpe].filled_scans[i],
                #     os.path.join(self._results_dir, 'filled_scans_mpe',
                #                  "%04d-%s-%s.png" % (i, rid, rclass)))
                self._datas[0].plot_polar_scan_plot(
                    plot_data,
                    self._submodel_results[max_class_mpe_weighted].filled_scans[i],
                    os.path.join(self._results_dir, 'filled_scans_mpe_weighted',
                                 "%04d-%s-%s.png" % (i, rid, rclass)))
                sys.stdout.write('.')
                sys.stdout.flush()

        plt.close()
        print("")
        print("- Masked confusion matrix for marginal (unweighted):")
        pp.pprint(cm_marginal)
        pp.pprint(norm_cm(cm_marginal) * 100.0)
        print("- Masked confusion matrix for marginal (weighted):")
        pp.pprint(cm_marginal_weighted)
        pp.pprint(norm_cm(cm_marginal_weighted) * 100.0)
        print("- Masked confusion matrix for MPE (unweighted):")
        pp.pprint(cm_mpe)
        pp.pprint(norm_cm(cm_mpe) * 100.0)
        print("- Masked confusion matrix for MPE (weighted):")
        pp.pprint(cm_mpe_weighted)
        pp.pprint(norm_cm(cm_mpe_weighted) * 100.0)

        print("- Masked classification rate for marginal (unweighted): %s" %
              (get_class_rate(cm_marginal) * 100.0))
        print("- Masked classification rate for marginal (weighted): %s" %
              (get_class_rate(cm_marginal_weighted) * 100.0))
        print("- Masked classification rate for MPE (unweighted): %s" %
              (get_class_rate(cm_mpe) * 100.0))
        print("- Masked classification rate for MPE (weighted): %s" %
              (get_class_rate(cm_mpe_weighted) * 100.0))

        print("- Avg completion ratio for marginal (unweighted): %s" %
              (np.mean(crs_marginal)))
        print("- Avg completion ratio for marginal (weighted): %s" %
              (np.mean(crs_marginal_weighted)))
        print("- Avg completion ratio for MPE (unweighted): %s" %
              (np.mean(crs_mpe)))
        print("- Avg completion ratio for MPE (weighted): %s" %
              (np.mean(crs_mpe_weighted)))

        # Save
        np.save(os.path.join(self._results_dir, 'complation_ratios_marginal'),
                crs_marginal)
        np.save(os.path.join(self._results_dir, 'complation_ratios_marginal_weighted'),
                crs_marginal_weighted)
        np.save(os.path.join(self._results_dir, 'complation_ratios_mpe'),
                crs_mpe)
        np.save(os.path.join(self._results_dir, 'complation_ratios_mpe_weighted'),
                crs_mpe_weighted)

        np.save(os.path.join(self._results_dir, 'masked_cm_marginal'),
                cm_marginal)
        np.save(os.path.join(self._results_dir, 'masked_cm_marginal_weighted'),
                cm_marginal_weighted)
        np.save(os.path.join(self._results_dir, 'masked_cm_mpe'),
                cm_mpe)
        np.save(os.path.join(self._results_dir, 'masked_cm_mpe_weighted'),
                cm_mpe_weighted)

    def get_classification_results(self):
        # Get first data as representative of all data
        data = self._datas[0].data
        test_rooms = self._datas[0].test_rooms

        # Calculate
        cm_marginal = np.zeros((self._num_classes, self._num_classes))
        cm_mpe = np.zeros((self._num_classes, self._num_classes))
        cm_marginal_weighted = np.zeros((self._num_classes, self._num_classes))
        cm_mpe_weighted = np.zeros((self._num_classes, self._num_classes))
        for i, d in enumerate(data):
            rid = d[0]
            rclass = d[1]
            # Test sample?
            if rid in test_rooms:
                rclass_id = self._known_classes.index(rclass)
                # Get values for masked scan from each submodel
                marginal_vals = np.array([float(r.marginal_vals[i])
                                          for r in self._submodel_results])
                mpe_vals = np.array([float(r.mpe_vals[i])
                                     for r in self._submodel_results])

                # Get max class
                max_class_marginal = np.argmax(marginal_vals)
                max_class_marginal_weighted = np.argmax(marginal_vals +
                                                        np.log(self._class_weights))
                max_class_mpe = np.argmax(mpe_vals)
                max_class_mpe_weighted = np.argmax(mpe_vals +
                                                   np.log(self._class_weights))

                # Get confusion matrix
                cm_marginal[rclass_id, max_class_marginal] += 1
                cm_marginal_weighted[rclass_id, max_class_marginal_weighted] += 1
                cm_mpe[rclass_id, max_class_mpe] += 1
                cm_mpe_weighted[rclass_id, max_class_mpe_weighted] += 1

        print("- Number of test samples: %s" % int(np.sum(cm_marginal)))

        print("- Confusion matrix for marginal (unweighted):")
        pp.pprint(cm_marginal)
        pp.pprint(norm_cm(cm_marginal) * 100.0)
        print("- Confusion matrix for marginal (weighted):")
        pp.pprint(cm_marginal_weighted)
        pp.pprint(norm_cm(cm_marginal_weighted) * 100.0)
        print("- Confusion matrix for MPE (unweighted):")
        pp.pprint(cm_mpe)
        pp.pprint(norm_cm(cm_mpe) * 100.0)
        print("- Confusion matrix for MPE (weighted):")
        pp.pprint(cm_mpe_weighted)
        pp.pprint(norm_cm(cm_mpe_weighted) * 100.0)

        print("- Classification rate for marginal (unweighted): %s" %
              (get_class_rate(cm_marginal) * 100.0))
        print("- Classification rate for marginal (weighted): %s" %
              (get_class_rate(cm_marginal_weighted) * 100.0))
        print("- Classification rate for MPE (unweighted): %s" %
              (get_class_rate(cm_mpe) * 100.0))
        print("- Classification rate for MPE (weighted): %s" %
              (get_class_rate(cm_mpe_weighted) * 100.0))

        # Save
        np.save(os.path.join(self._results_dir, 'cm_marginal'),
                cm_marginal)
        np.save(os.path.join(self._results_dir, 'cm_marginal_weighted'),
                cm_marginal_weighted)
        np.save(os.path.join(self._results_dir, 'cm_mpe'),
                cm_mpe)
        np.save(os.path.join(self._results_dir, 'cm_mpe_weighted'),
                cm_mpe_weighted)

    def get_novelty_results(self):
        # Get first data as representative of all data
        data = self._datas[0].data
        test_rooms = self._datas[0].test_rooms
        novel_classes = self._datas[0].novel_classes

        all_y = []
        all_pred = []
        num_indist = 0
        num_novel = 0
        num_indist_rejected = 0
        num_novel_rejected = 0
        threshold = -596
        for i, d in enumerate(data):
            rid = d[0]
            rclass = d[1]
            marginal_vals = np.array([float(r.marginal_vals[i])
                                      for r in self._submodel_results])
            marginal_vals_weighted = marginal_vals + np.log(self._class_weights)
            max_val = np.max(marginal_vals_weighted)
            val = max_val + np.log(np.sum(np.exp(marginal_vals_weighted - max_val)))
            # Test sample?
            if rid in test_rooms:
                # Test sample
                num_indist += 1
                all_pred.append(val)
                all_y.append(1)
                if val < threshold:
                    num_indist_rejected += 1
            elif rclass in novel_classes:
                # Novel sample
                num_novel += 1
                all_pred.append(val)
                all_y.append(-1)
                if val < threshold:
                    num_novel_rejected += 1

        print("- Number of test samples: %s" % num_indist)
        print("- Number of novel samples: %s" % num_novel)
        all_together = np.c_[np.array(all_y), np.array(all_pred)]

        # Get AUC
        fpr, tpr, _ = sklearn.metrics.roc_curve(all_together[:, 0].astype(int),
                                                all_together[:, 1])
        self._plot_roc(fpr, tpr, os.path.join(self._results_dir, 'novelty_roc.png'))
        aucval = sklearn.metrics.auc(fpr, tpr)
        print("- AUC for novelty detection: %s" % aucval)

        print("- Test for threshold %s: %s %s" %
              (threshold,
               num_indist_rejected / num_indist, num_novel_rejected / num_novel))

        # Save
        np.save(os.path.join(self._results_dir, 'novelty_results'),
                all_together)


    def get_classification_results_for_graphs(self):
        # Get first data as representative of all data
        data = self._datas[0].data
        test_rooms = self._datas[0].test_rooms

        graph_results = {}  # rid (i.e. graph id) -> {node id -> [groundtruth, prediction, likelihoods(normalized)]}
        for i, d in enumerate(data):
            rid = d[0]
            rclass = util.CategoryManager.canonical_category(d[1], checking=True) ## TODO: WE DON"T NEED CATEGORYMANAGER HERE
            # Test sample?
            if rid in test_rooms:

                if rid not in graph_results:
                    graph_results[rid] = {}

                node_id = d[-1]
                # Get values for masked scan from each submodel
                # NOTE: use MPE result for now.
                # marginal_vals = np.array([float(r.marginal_vals[i])
                #                           for r in self._submodel_results])
                mpe_vals = np.array([float(r.mpe_vals[i])
                                     for r in self._submodel_results])
                # Get max class (weighted)
                max_class_mpe_weighted = self._known_classes[np.argmax(mpe_vals +
                                                                       np.log(self._class_weights))]
                # # Normalize the likelihoods; The normalization basically turns the likelihoods
                # # from log space into normal space by taking exp(); These likelihoods can be
                # # fed into GraphSPN directly; All values fed into an SPN will be taken log().
                # likelihoods = list(interpret(mpe_vals))
                likelihoods = list(mpe_vals)

                graph_results[rid][node_id] = [rclass, max_class_mpe_weighted, likelihoods, list(mpe_vals)]

        # Computing statistics
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

        print("- Overall statistics")
        pp.pprint(stats)

        # Save
        os.makedirs(os.path.join(self._results_dir, "graphs"), exist_ok=True)
        for graph_id in graph_results:
            with open(os.path.join(self._results_dir, "graphs", graph_id + "_likelihoods.json"), 'w') as f:
                json.dump(graph_results[graph_id], f)


    def _plot_roc(self, fpr, tpr, filename):
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %
                 sklearn.metrics.auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()


