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
from deepsm.util import CategoryManager
import deepsm.experiments.common as common

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
    learn_params.add_argument('--num-epochs', type=float, default=100,
                              help='Total number of epochs')
    learn_params.add_argument('--num-batches', type=float, default=20,
                              help='Number of batches to divide the data for training')
    learn_params.add_argument('--weight-init', type=str, default='random',
                              help='Weight init value: ' +
                              ', '.join([a.name.lower()
                                         for a in PlaceModel.WeightInitValue]))
    learn_params.add_argument('--learning-rate', type=int, default=0.001,
                              help='Learning rate for gradient descent')
    learn_params.add_argument('--value-inference', type=str, default='marginal',
                              help='Type of inference during EM upwards pass: ' +
                              ', '.join([a.name.lower()
                                         for a in spn.InferenceType]))

    # Testing params
    test_params = parser.add_argument_group(title="testing parameters")
    test_params.add_argument('--mask-seed', type=int, default=100,
                             help='Seed used for randomizing masks')

    # Other
    other_params = parser.add_argument_group(title="other")
    other_params.add_argument('--save-masked', action='store_true',
                              help='Save masked scans')
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
    print("* Num Epocs: %s" % args.num_epochs)
    print("* Num Batches: %s" % args.num_batches)

    print("\nTesting parameters:")
    print("* Mask seed: %s" % args.mask_seed)

    print("\nOther:")
    print("* Save masked scans: %s" % args.save_masked)


def create_directories(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'mpe_states'), exist_ok=True)
    if args.save_masked:
        os.makedirs(os.path.join(args.results_dir, 'masked_scans'), exist_ok=True)

        
def main(args=None):
    if args is None:
        args = parse_args()

    create_directories(args)

    data = Data(args.angle_cells, args.radius_min,
                args.radius_max, args.radius_factor)
    data.load(args.data_dir)
    data.process(args.subset, args.occupancy_vals)
    data.print_info()
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
    PlaceModel.balance_data(training_scans, training_labels, rnd=rnd)

    sys.stdout.write("Verifying data balance...")
    class_counts, _ = PlaceModel._count_class_samples(training_scans, training_labels)
    count = class_counts[rnd.choice(list(class_counts.keys()))]
    for catg_num in class_counts:
        if class_counts[catg_num] != count:
            raise ValueError("Class %s has %d samples but needs %d" % (CategoryManager.category_map(catg_num, rev=True),
                                                                       class_counts[catg_num], count))
    sys.stdout.write("OK\n")

    # Shuffle data
    data._training_scans = np.array(training_scans, dtype=int)
    data._training_labels = np.array(training_labels, dtype=int).reshape(-1,1)
    if shuffle:
        print("Shuffling...")
        p = np.random.permutation(len(data.training_scans))
        data._training_scans = data.training_scans[p]
        data._training_labels = data.training_labels[p]

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
                       value_inference_type=args.value_inference,
                       optimizer=tf.train.AdamOptimizer)
    try:
        model.train(args.num_batches, args.num_epochs)
    except KeyboardInterrupt:
        print("Stop training...")
    finally:
        model.test(args.results_dir, graph_test=True)


if __name__ == '__main__':
    main()
