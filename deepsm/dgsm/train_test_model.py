#!/usr/bin/env python3

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import os
import sys
import argparse
import random
import libspn as spn
import tensorflow as tf
import numpy as np
from deepsm.dgsm.place_model import PlaceModel
from deepsm.dgsm.data import Data
from deepsm.util import CategoryManager, plot_to_file, plot_roc
import deepsm.experiments.common as common
from pprint import pprint
import math
import csv

def create_parser():
    parser = argparse.ArgumentParser(description='Train and test an SPN place model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is stored')
    parser.add_argument('results_dir', type=str,
                        help='Directory where results are saved')
    parser.add_argument('subset', type=int,
                        help='Data subset')

    # Data params
    data_params = parser.add_argument_group(title="data parameters")
    data_params.add_argument('--occupancy-vals', type=str, default='three',
                             help='How many occupancy values to consider: ' +
                             ', '.join([a.name.lower()
                                        for a in Data.OccupancyVals]))
    data_params.add_argument('--resolution', type=float, default=common.resolution,
                             help='')
    data_params.add_argument('--angle-cells', type=float, default=common.num_angle_cells,
                             help='Number of angular cells')
    data_params.add_argument('--radius-min', type=float, default=common.min_radius,
                             help='')
    data_params.add_argument('--radius-max', type=float, default=common.max_radius,
                             help='')
    data_params.add_argument('--radius-factor', type=float, default=common.radius_factor,
                             help='')
    data_params.add_argument('--data-seed', type=int, default=547,
                             help='Seed used for shuffling training data')
    data_params.add_argument('--balance-data', action='store_true',
                             help='Upsample the miniority class instances so that all class have the same number of training samples')

    # Model params
    model_params = parser.add_argument_group(title="model parameters")
    model_params.add_argument('--view-input-dist', type=str, default='raw',
                              help='Input distributions: ' +
                              ', '.join([a.name.lower()
                                         for a in spn.DenseSPNGenerator.InputDist]))
    model_params.add_argument('--view-decomps', type=int, default=1,
                              help='Number of decompositions for a view')
    model_params.add_argument('--view-subsets', type=int, default=2,
                              help='Number of subsets for a view')
    model_params.add_argument('--view-mixtures', type=int, default=4,
                              help='Number of mixtures for a view')
    model_params.add_argument('--view-input-mixtures', type=int, default=3,
                              help='Number of input mixtures for a view')
    model_params.add_argument('--view-top-mixtures', type=int, default=14,
                              help='Number of top mixtures for a view')
    model_params.add_argument('--place-decomps', type=int, default=4,
                              help='Number of decompositions for a place')
    model_params.add_argument('--place-subsets', type=int, default=4,
                              help='Number of subsets for a place')
    model_params.add_argument('--place-mixtures', type=int, default=5,
                              help='Number of mixtures for a place')
    model_params.add_argument('--place-input-mixtures', type=int, default=5,
                              help='Number of input mixtures for a place')

    # Learning params
    learn_params = parser.add_argument_group(title="learning parameters")
    learn_params.add_argument('--update-threshold', type=float, default=0.00001,
                              help='Threshold of likelihood update')
    learn_params.add_argument('--batch-size', type=int, default=10,
                              help='Size of each batch for training')
    learn_params.add_argument('--epoch-limit', type=int, default=50,
                              help='limit on epochs')
    learn_params.add_argument('--weight-init', type=str, default='random',
                              help='Weight init value: ' +
                              ', '.join([a.name.lower()
                                         for a in PlaceModel.WeightInitValue]))
    learn_params.add_argument('--value-inference', type=str, default='marginal',
                              help='Type of inference during EM upwards pass: ' +
                              ', '.join([a.name.lower()
                                        for a in spn.InferenceType]))
    learn_params.add_argument('--dropconnect-keep-prob', type=float, default=-1,
                              help='drop-connect probability parameter for gd learning')
    # GDLearning
    learn_params.add_argument('--learning-rate', type=float, default=0.001,
                              help='Learning rate for gradient descent')
    # EMLearning
    learn_params.add_argument('--init-accum', type=int, default=20,
                              help='Initial accumulator value')
    learn_params.add_argument('--smoothing-val', type=float, default=0.0,
                              help='Additive smoothing value')
    learn_params.add_argument('--smoothing-min', type=float, default=0.0,
                              help='Additive smoothing min value')
    learn_params.add_argument('--smoothing-decay', type=float, default=0.2,
                              help='Additive smoothing decay')


    # Testing params
    test_params = parser.add_argument_group(title="testing parameters")
    test_params.add_argument('--mask-seed', type=int, default=100,
                             help='Seed used for randomizing masks')
    test_params.add_argument('--graph-test', action="store_true",
                             help='Graph-scale test, producing results to feed into GraphSPN.')
    

    # Other
    other_params = parser.add_argument_group(title="other")
    other_params.add_argument('--save-masked', action='store_true',
                              help='Save masked scans')
    other_params.add_argument('--building', type=str, default='default',
                              help='Building identifier for this run.')
    return parser

    
def parse_args(parser=None, args_list=None):
    if parser is None:
        parser = create_parser()
    
    # Parse
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    # Check
    if args.subset not in [1, 2, 3, 4]:
        print("ERROR: Incorrect subset ('%s')." % args.subset)
        sys.exit(1)
    try:
        args.occupancy_vals = Data.OccupancyVals[
            args.occupancy_vals.upper()]
    except KeyError:
        print("ERROR: Incorrect occupancy-vals ('%s')." % args.occupancy_vals)
        sys.exit(1)
    try:
        args.view_input_dist = spn.DenseSPNGeneratorLayerNodes.InputDist[
            args.view_input_dist.upper()]
    except KeyError:
        print("ERROR: Incorrect view-input-dist ('%s')." % args.view_input_dist)
        sys.exit(1)
    try:
        args.weight_init = PlaceModel.WeightInitValue[args.weight_init.upper()]
    except KeyError:
        print("ERROR: Incorrect weight-init ('%s')." % args.weight_init)
        sys.exit(1)
    try:
        args.value_inference = spn.InferenceType[args.value_inference.upper()]
    except KeyError:
        print("ERROR: Incorrect value-inference ('%s')." % args.value_inference)
        sys.exit(1)

    return args


def print_args(args):
    print("---------")
    print("Arguments")
    print("---------")
    print("* Data dir: %s" % args.data_dir)
    print("* Results dir: %s" % args.results_dir)
    print("* Classes: %s" % str(CategoryManager.known_categories()))
    print("* Subset: %s" % args.subset)

    print("\nData parameters:")
    print("* Occupancy values: %s" % args.occupancy_vals)
    print("* Resolution: %s" % args.resolution)
    print("* Angle cells: %s" % args.angle_cells)
    print("* Min radius: %s" % args.radius_min)
    print("* Max radius: %s" % args.radius_max)
    print("* Radius factor: %s" % args.radius_factor)
    print("* Training data shuffling seed: %s" % args.data_seed)
    print("* Balance data: %s" % args.balance_data)

    print("\nModel parameters:")
    print("* View input distributions: %s" % args.view_input_dist)
    print("* View decompositions: %s" % args.view_decomps)
    print("* View subsets: %s" % args.view_subsets)
    print("* View mixtures: %s" % args.view_mixtures)
    print("* View input mixtures: %s" % args.view_input_mixtures)
    print("* View top mixtures: %s" % args.view_top_mixtures)
    print("* Place decompositions: %s" % args.place_decomps)
    print("* Place subsets: %s" % args.place_subsets)
    print("* Place mixtures: %s" % args.place_mixtures)
    print("* Place input mixtures: %s" % args.place_input_mixtures)

    print("\nLearning parameters:")
    print("* Weight initialization: %s" % args.weight_init)
    print("* Learning rate: %s" % args.learning_rate)
    print("* Drop-connect keep prob: %s" % args.dropconnect_keep_prob)
    print("* Epoch limit: %s" % args.epoch_limit)
    print("* Likelihood update threshold: %s" % args.update_threshold)
    print("* Batch size: %s" % args.batch_size)
    print("* Value Inference: %s"  % args.value_inference)

    print("\nTesting parameters:")
    print("* Mask seed: %s" % args.mask_seed)

    print("\nOther:")
    print("* Save masked scans: %s" % args.save_masked)


def create_directories(args):
    os.makedirs(args.results_dir, exist_ok=True)
    if not args.graph_test:
        os.makedirs(os.path.join(args.results_dir, 'mpe_states'), exist_ok=True)
        if args.save_masked:
            os.makedirs(os.path.join(args.results_dir, 'masked_scans'), exist_ok=True)

def make_trial_name(args, model):
    trial_name = ""
    if args.balance_data:
        trial_name = "balanced"
    else:
        trial_name = "unbalanced"

    trial_name += "_lr" + str(abs(round(math.log(args.learning_rate, 10))))
    trial_name += "_b" + str(args.batch_size)
    if args.update_threshold != 0:
        trial_name += "_uc" + str(abs(round(math.log(args.update_threshold, 10))))
    else:
        trial_name += "_ucZero"
    trial_name += "_d" + str(args.dropconnect_keep_prob)
    trial_name += "_mpe" if args.value_inference == spn.InferenceType.MPE else "_marginal"
    trial_name += "_k" + str(CategoryManager.NUM_CATEGORIES)
    trial_name += "_E" + str(args.epoch_limit)
    trial_name += "_" + str(model._learning_method)
    trial_name += "_" + args.building
    return trial_name

        
def main(args=None):
    if args is None:
        args = parse_args()

    create_directories(args)

    data = Data(args.angle_cells, args.radius_min,
                args.radius_max, args.radius_factor)
    data.load(args.data_dir)
    data.process(args.subset, args.occupancy_vals)
    data.print_info()
    if not args.graph_test:
        data.generate_masked_scans(args.mask_seed)
        data.save_masked_scans(args.results_dir)
        if args.save_masked:
            data.visualize_masked_scans(
                os.path.join(args.results_dir, 'masked_scans'))

    rnd = random.Random()
    rnd.seed(567)
    shuffle = True

    # Get data
    training_scans = list(data.training_scans)
    training_labels = list(data.training_labels.flatten())

    # Balance data
    if args.balance_data:
        PlaceModel.balance_data(training_scans, training_labels, rnd=rnd)

        sys.stdout.write("Verifying data balance...")
        class_counts, _ = PlaceModel._count_class_samples(training_scans, training_labels)
        count = class_counts[rnd.choice(list(class_counts.keys()))]
        for catg_num in class_counts:
            if class_counts[catg_num] != count:
                raise ValueError("Class %s has %d samples but needs %d" % (CategoryManager.category_map(catg_num, rev=True),
                                                                           class_counts[catg_num], count))
        sys.stdout.write("OK\n")

    print_args(args)

    # Model
    model = PlaceModel(data=data,
                       view_input_dist=args.view_input_dist,
                       view_num_decomps=args.view_decomps,
                       view_num_subsets=args.view_subsets,
                       view_num_mixtures=args.view_mixtures,
                       view_num_input_mixtures=args.view_input_mixtures,
                       view_num_top_mixtures=args.view_top_mixtures,
                       place_num_decomps=args.place_decomps,
                       place_num_subsets=args.place_subsets,
                       place_num_mixtures=args.place_mixtures,
                       place_num_input_mixtures=args.place_input_mixtures,
                       weight_init_value=args.weight_init,
                       learning_rate=args.learning_rate,
                       init_accum_val=args.init_accum,
                       smoothing_val=args.smoothing_val,
                       smoothing_min=args.smoothing_min,
                       smoothing_decay=args.smoothing_decay,
                       value_inference_type=args.value_inference,
                       optimizer=tf.train.AdamOptimizer,
                       dropconnect_keep_prob=args.dropconnect_keep_prob)
    trial_name = make_trial_name(args, model)
    train_loss, test_loss, train_perf, test_perf = [], [], [], []
    epoch = 0
    try:
        model.train(args.batch_size, args.update_threshold, train_loss=train_loss, test_loss=test_loss,
                    train_perf=train_perf, test_perf=test_perf,
                    shuffle=shuffle, epoch_limit=args.epoch_limit)
        epoch = len(train_loss)
    except KeyboardInterrupt:
        print("Stop training...")
    finally:
        dirpath = os.path.join("analysis", "dgsm")

        trial_name = make_trial_name(args, model)

        loss_plot_path = os.path.join(dirpath, 'loss-%s.png' % trial_name)
        plot_to_file(train_loss, test_loss, train_perf, test_perf,
                     labels=['train loss', 'test loss', 'train accuracy', 'test accuracy'],
                     xlabel='epochs',
                     ylabel='Cross Entropy Loss', path=loss_plot_path)
        np.savetxt(os.path.join(dirpath, 'loss-train-%s.txt' % trial_name), train_loss, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(dirpath, 'loss-test-%s.txt' % trial_name), test_loss, delimiter=',', fmt='%.4f')

        cm_weighted, cm_weighted_norm, stats, roc_results = model.test(args.results_dir, graph_test=args.graph_test)
        model.train_test_samples_exam(dirpath, trial_name)

        # Report cm
        with open(os.path.join(dirpath, 'cm-%s.txt' % trial_name), 'w') as f:
            pprint(cm_weighted, stream=f)
            pprint(cm_weighted_norm, stream=f)
            pprint(stats, stream=f)

        # Plot roc
        roc_plot_path = os.path.join(dirpath, 'roc-%s.png' % trial_name)
        plot_roc(roc_results, savepath=roc_plot_path,
                 names=CategoryManager.known_categories())

        # Report file
        with open(os.path.join(dirpath, 'full-report-%s.csv' % trial_name), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')

            # Header
            writer.writerow(['trial_name', 'learning_rate', 'batch_size', 'stopping_condition', 'inference_type',
                             'epochs', 'class_rate', 'class_rate_top2', 'class_rate-top3',
                             'cm_diagonal', 'loss_plot', 'ROC_curve'])
            writer.writerow([trial_name, args.learning_rate, args.batch_size, args.update_threshold, args.value_inference,
                             epoch, stats['accuracy'], stats['accuracy_top2'], stats['accuracy_top3'],
                             str([cm_weighted_norm[i,i] for i in range(cm_weighted.shape[0])]),
                             os.path.join("https://github.com/pronobis/deep-semantic-mapping/blob/master/deepsm/experiments/analysis/dgsm",
                                          os.path.basename(loss_plot_path)),
                             os.path.join("https://github.com/pronobis/deep-semantic-mapping/blob/master/deepsm/experiments/analysis/dgsm",
                                          os.path.basename(roc_plot_path))])

if __name__ == '__main__':
    main()
