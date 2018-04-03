#!/usr/bin/env python3

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse
import numpy as np
import sklearn.metrics


num_subsets = 4


def parse_args():
    parser = argparse.ArgumentParser(description='Generate final results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('results_dir', type=str)

    # Parse
    args = parser.parse_args()
    return args


def print_args(args):
    print("---------")
    print("Arguments")
    print("---------")
    print("* Results dir: %s" % args.results_dir)


class SetResults:

    def __init__(self, subset):
        self._subset = subset

    def load(self, res_dir):
        # Classification results
        self.cm_marginal = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'cm_marginal.npy'))
        self.cm_marginal_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'cm_marginal_weighted.npy'))
        self.cm_mpe = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'cm_mpe.npy'))
        self.cm_mpe_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'cm_mpe_weighted.npy'))

        # Novelty results
        self.novelty_results = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'novelty_results.npy'))

        # Completion results
        self.masked_crs_marginal = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'complation_ratios_marginal.npy'))
        self.masked_crs_marginal_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'complation_ratios_marginal_weighted.npy'))
        self.masked_crs_mpe = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'complation_ratios_mpe.npy'))
        self.masked_crs_mpe_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'complation_ratios_mpe_weighted.npy'))

        self.masked_cm_marginal = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'masked_cm_marginal.npy'))
        self.masked_cm_marginal_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'masked_cm_marginal_weighted.npy'))
        self.masked_cm_mpe = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'masked_cm_mpe.npy'))
        self.masked_cm_mpe_weighted = np.load(
            os.path.join(res_dir, 'set%d' % self._subset, 'masked_cm_mpe_weighted.npy'))


def norm_cm(cm):
    return cm / np.sum(cm, axis=1, keepdims=True)


def get_class_rate(cm):
    ncm = norm_cm(cm)
    d = ncm.diagonal()
    return np.mean(d)


def get_class_rate_naive(cm):
    """Calculate a naive CR where each sample (not class) has equal weight."""
    return np.sum(np.diag(cm)) / np.sum(cm)


def plot_conf_mat(conf_mat_a, img_path):
    conf_mat = norm_cm(conf_mat_a)
    conf_mat = conf_mat * 100

    plt.rc('ytick', labelsize=10)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_mat), cmap=plt.cm.Oranges,
                    interpolation='nearest')
    width, height = conf_mat.shape
    plt.grid('off')
    for x in range(width):
        for y in range(height):
            ax.annotate("%.1f" % (conf_mat[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    fig.set_size_inches(5.5, 3.5)
    fig.colorbar(res)
    plt.xticks(range(width), ['Corridor', 'Doorway', 'Small Office', 'Large Ofice'])
    plt.yticks(range(height), ['Corridor', 'Doorway', 'Small Office', 'Large Ofice'])
    plt.ylabel("True Class", fontsize=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.show()
    plt.savefig(img_path, dpi=200, bbox_inches='tight')


def classification_results(set_results, res_dir):
    print("\n\n------------------")
    print("CLASSIFICATION")
    print("------------------")

    # Set classification rates
    print("\nClassification rate (marginal, unweighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.cm_marginal) * 100.0))
    print("\nClassification rate (marginal, weighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.cm_marginal_weighted) * 100.0))
    print("\nClassification rate (MPE, unweighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.cm_mpe) * 100.0))
    print("\nClassification rate (MPE, weighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.cm_mpe_weighted) * 100.0))

    # For comparison plot SVM confmat
    svm_cm = np.array([[451., 7., 2., 1.],
                       [2., 41., 3., 0.],
                       [7., 3., 260., 55.],
                       [7., 3., 73., 398.]])
    plot_conf_mat(svm_cm,
                  os.path.join(res_dir, 'svm_cm.png'))

    # Cumulative confusion matrices
    print("\nCumulative confusion matrix (marginal, unweighted):")
    cum_cm_marginal = np.sum([sr.cm_marginal for sr in set_results], axis=0)
    print(cum_cm_marginal)
    print(norm_cm(cum_cm_marginal) * 100.0)
    plot_conf_mat(cum_cm_marginal,
                  os.path.join(res_dir, 'cum_cm_marginal.png'))

    print("\nCumulative confusion matrix (marginal, weighted):")
    cum_cm_marginal_weighted = np.sum([sr.cm_marginal_weighted
                                       for sr in set_results], axis=0)
    print(cum_cm_marginal_weighted)
    print(norm_cm(cum_cm_marginal_weighted) * 100.0)
    plot_conf_mat(cum_cm_marginal_weighted,
                  os.path.join(res_dir, 'cum_cm_marginal_weighted.png'))

    print("\nCumulative confusion matrix (mpe, unweighted):")
    cum_cm_mpe = np.sum([sr.cm_mpe for sr in set_results], axis=0)
    print(cum_cm_mpe)
    print(norm_cm(cum_cm_mpe) * 100.0)
    plot_conf_mat(cum_cm_mpe,
                  os.path.join(res_dir, 'cum_cm_mpe.png'))

    print("\nCumulative confusion matrix (mpe, weighted):")
    cum_cm_mpe_weighted = np.sum([sr.cm_mpe_weighted
                                  for sr in set_results], axis=0)
    print(cum_cm_mpe_weighted)
    print(norm_cm(cum_cm_mpe_weighted) * 100.0)
    plot_conf_mat(cum_cm_mpe_weighted,
                  os.path.join(res_dir, 'cum_cm_mpe_weighted.png'))

    # Cumulative classification rates (avg over samples)
    print("\nCumulative classification rate (avg over samples) (marginal, unweighted): %.3f" %
          (get_class_rate(cum_cm_marginal) * 100.0))
    print("\nCumulative classification rate (avg over samples) (marginal, weighted): %.3f" %
          (get_class_rate(cum_cm_marginal_weighted) * 100.0))
    print("\nCumulative classification rate (avg over samples) (mpe, unweighted): %.3f" %
          (get_class_rate(cum_cm_mpe) * 100.0))
    print("\nCumulative classification rate (avg over samples) (mpe, weighted): %.3f" %
          (get_class_rate(cum_cm_mpe_weighted) * 100.0))

    # Cumulative classification rates (avg over sets)
    print("\nCumulative classification rate (avg over sets) (marginal, unweighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.cm_marginal) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.cm_marginal) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (marginal, weighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.cm_marginal_weighted) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.cm_marginal_weighted) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (mpe, unweighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.cm_mpe) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.cm_mpe) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (mpe, weighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.cm_mpe_weighted) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.cm_mpe_weighted) * 100.0
                   for sr in set_results])))


def plot_roc(fpr, tpr, geomfeat_fpr, geomfeat_tpr, filename):
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label='DGSM (area = %0.2f)' %
             sklearn.metrics.auc(fpr, tpr))
    plt.plot(geomfeat_fpr, geomfeat_tpr, label='SVM (area = %0.2f)' %
             sklearn.metrics.auc(geomfeat_fpr, geomfeat_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def novelty_results(set_results, res_dir):
    print("\n\n------------------")
    print("NOVELTY")
    print("------------------")

    # Single set stuff
    print("\nNovelty AUC per set:")
    set_auc_vals = []
    set_weights = []
    for i, sr in enumerate(set_results):
        fpr, tpr, _ = sklearn.metrics.roc_curve(sr.novelty_results[:, 0].astype(int),
                                                sr.novelty_results[:, 1])
        aucval = sklearn.metrics.auc(fpr, tpr)
        set_auc_vals.append(aucval)
        set_weights.append(sr.novelty_results.shape[0])
        print(" - set %i: %.4f" % (i + 1, aucval))
    set_weights /= np.sum(set_weights)

    # Cumulative (assuming same threshold for all sets)
    all_novelty_results = np.vstack([sr.novelty_results for sr in set_results])
    fpr, tpr, _ = sklearn.metrics.roc_curve(all_novelty_results[:, 0].astype(int),
                                            all_novelty_results[:, 1])
    aucval = sklearn.metrics.auc(fpr, tpr)
    print("\nCumulative novelty AUC (assuming same threshold for all sets): %.4f" %
          aucval)

    # Cumulative (avg over sets)
    print("\nCumulative novelty AUC (avg over sets): %.4f +/- %.4f" %
          (np.mean(set_auc_vals), np.std(set_auc_vals)))

    # Load SVM ROC
    geomfeat_fpr, geomfeat_tpr = np.load('./geomfeat-novelty_results.npy')

    # Store cumulative ROC
    plot_roc(fpr, tpr, geomfeat_fpr, geomfeat_tpr,
             os.path.join(res_dir, 'cum_novelty_roc.png'))


def masked_classification_results(set_results, res_dir):
    print("\n\n------------------")
    print("MASKED CLASSIFICATION")
    print("------------------")

    # Set classification rates
    print("\nClassification rate (marginal, unweighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.masked_cm_marginal) * 100.0))
    print("\nClassification rate (marginal, weighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.masked_cm_marginal_weighted) * 100.0))
    print("\nClassification rate (MPE, unweighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.masked_cm_mpe) * 100.0))
    print("\nClassification rate (MPE, weighted):")
    for i, sr in enumerate(set_results):
        print("- Set %d: %.3f" %
              (i + 1, get_class_rate(sr.masked_cm_mpe_weighted) * 100.0))

    # Cumulative confusion matrices
    print("\nCumulative confusion matrix (marginal, unweighted):")
    cum_cm_marginal = np.sum([sr.masked_cm_marginal for sr in set_results], axis=0)
    print(cum_cm_marginal)
    print(norm_cm(cum_cm_marginal) * 100.0)
    plot_conf_mat(cum_cm_marginal,
                  os.path.join(res_dir, 'cum_masked_cm_marginal.png'))

    print("\nCumulative confusion matrix (marginal, weighted):")
    cum_cm_marginal_weighted = np.sum([sr.masked_cm_marginal_weighted
                                       for sr in set_results], axis=0)
    print(cum_cm_marginal_weighted)
    print(norm_cm(cum_cm_marginal_weighted) * 100.0)
    plot_conf_mat(cum_cm_marginal_weighted,
                  os.path.join(res_dir, 'cum_masked_cm_marginal_weighted.png'))

    print("\nCumulative confusion matrix (mpe, unweighted):")
    cum_cm_mpe = np.sum([sr.masked_cm_mpe for sr in set_results], axis=0)
    print(cum_cm_mpe)
    print(norm_cm(cum_cm_mpe) * 100.0)
    plot_conf_mat(cum_cm_mpe,
                  os.path.join(res_dir, 'cum_masked_cm_mpe.png'))

    print("\nCumulative confusion matrix (mpe, weighted):")
    cum_cm_mpe_weighted = np.sum([sr.masked_cm_mpe_weighted
                                  for sr in set_results], axis=0)
    print(cum_cm_mpe_weighted)
    print(norm_cm(cum_cm_mpe_weighted) * 100.0)
    plot_conf_mat(cum_cm_mpe_weighted,
                  os.path.join(res_dir, 'cum_masked_cm_mpe_weighted.png'))

    # Cumulative classification rates (avg over samples)
    print("\nCumulative classification rate (avg over samples) (marginal, unweighted): %.3f" %
          (get_class_rate(cum_cm_marginal) * 100.0))
    print("\nCumulative classification rate (avg over samples) (marginal, weighted): %.3f" %
          (get_class_rate(cum_cm_marginal_weighted) * 100.0))
    print("\nCumulative classification rate (avg over samples) (mpe, unweighted): %.3f" %
          (get_class_rate(cum_cm_mpe) * 100.0))
    print("\nCumulative classification rate (avg over samples) (mpe, weighted): %.3f" %
          (get_class_rate(cum_cm_mpe_weighted) * 100.0))

    # Cumulative classification rates (avg over sets)
    print("\nCumulative classification rate (avg over sets) (marginal, unweighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.masked_cm_marginal) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.masked_cm_marginal) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (marginal, weighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.masked_cm_marginal_weighted) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.masked_cm_marginal_weighted) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (mpe, unweighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.masked_cm_mpe) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.masked_cm_mpe) * 100.0
                   for sr in set_results])))
    print("\nCumulative classification rate (avg over sets) (mpe, weighted): %.4f +/- %.4f" %
          (np.mean([get_class_rate(sr.masked_cm_mpe_weighted) * 100.0
                    for sr in set_results]),
           np.std([get_class_rate(sr.masked_cm_mpe_weighted) * 100.0
                   for sr in set_results])))


def completion_results(set_results, res_dir):
    print("\n\n------------------")
    print("COMPLETION")
    print("------------------")

    # Set results
    print("\nAverage completion ratios per set (marginal, unweighted):")
    for i, sr in enumerate(set_results):
        print(" - set %i: %.4f" % (i + 1, np.mean(sr.masked_crs_marginal) * 100))
    print("\nAverage completion ratios per set (marginal, weighted):")
    for i, sr in enumerate(set_results):
        print(" - set %i: %.4f" % (i + 1, np.mean(sr.masked_crs_marginal_weighted) * 100))
    print("\nAverage completion ratios per set (mpe, unweighted):")
    for i, sr in enumerate(set_results):
        print(" - set %i: %.4f" % (i + 1, np.mean(sr.masked_crs_mpe) * 100))
    print("\nAverage completion ratios per set (mpe, weighted):")
    for i, sr in enumerate(set_results):
        print(" - set %i: %.4f" % (i + 1, np.mean(sr.masked_crs_mpe_weighted) * 100))

    # Cumulative
    print("\nAverage (over samples) cumulative completion ratio (marginal, unweighted): %.4f" %
          (np.mean(np.concatenate([sr.masked_crs_marginal for sr in set_results])) * 100))
    print("\nAverage (over samples) cumulative completion ratio (marginal, weighted): %.4f" %
          (np.mean(np.concatenate([sr.masked_crs_marginal_weighted for sr in set_results])) * 100))
    print("\nAverage (over samples) cumulative completion ratio (mpe, unweighted): %.4f" %
          (np.mean(np.concatenate([sr.masked_crs_mpe for sr in set_results])) * 100))
    print("\nAverage (over samples) cumulative completion ratio (mpe, weighted): %.4f" %
          (np.mean(np.concatenate([sr.masked_crs_mpe_weighted for sr in set_results])) * 100))

    # Cumulative
    print("\nAverage (over sets) cumulative completion ratio (marginal, unweighted): %.4f +/- %.4f" %
          (np.mean([np.mean(sr.masked_crs_marginal * 100) for sr in set_results]),
           np.std([np.mean(sr.masked_crs_marginal * 100) for sr in set_results])))
    print("\nAverage (over sets) cumulative completion ratio (marginal, weighted): %.4f +/- %.4f" %
          (np.mean([np.mean(sr.masked_crs_marginal_weighted * 100) for sr in set_results]),
           np.std([np.mean(sr.masked_crs_marginal_weighted * 100) for sr in set_results])))
    print("\nAverage (over sets) cumulative completion ratio (mpe, unweighted): %.4f +/- %.4f" %
          (np.mean([np.mean(sr.masked_crs_mpe * 100) for sr in set_results]),
           np.std([np.mean(sr.masked_crs_mpe * 100) for sr in set_results])))
    print("\nAverage (over sets) cumulative completion ratio (mpe, weighted): %.4f +/- %.4f" %
          (np.mean([np.mean(sr.masked_crs_mpe_weighted * 100) for sr in set_results]),
           np.std([np.mean(sr.masked_crs_mpe_weighted * 100) for sr in set_results])))


def main():
    args = parse_args()
    print_args(args)

    # Load results
    print("\nLoading results...")
    set_results = [SetResults(i + 1)
                   for i in range(num_subsets)]
    for i, sr in enumerate(set_results):
        print("- Set %i" % (i + 1))
        sr.load(args.results_dir)
    print("Done!")

    print("\nObtaining results...")
    classification_results(set_results, args.results_dir)
    novelty_results(set_results, args.results_dir)
    masked_classification_results(set_results, args.results_dir)
    completion_results(set_results, args.results_dir)
    print("Done!")


if __name__ == '__main__':
    main()
