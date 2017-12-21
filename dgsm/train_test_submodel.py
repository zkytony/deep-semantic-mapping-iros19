#!/usr/bin/env python3

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import os
import sys
import argparse
import libspn as spn
from model import PlaceSubModel
from data import Data

KNOWN_CLASSES = ['corridor', 'door', 'small_office', 'large_office']


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test an SPN place model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is stored')
    parser.add_argument('results_dir', type=str,
                        help='Directory where results are saved')
    parser.add_argument('submodel_class', type=str,
                        help='Class of this sub-model')
    parser.add_argument('subset', type=int,
                        help='Data subset')

    # Data params
    data_params = parser.add_argument_group(title="data parameters")
    data_params.add_argument('--occupancy-vals', type=str, default='three',
                             help='How many occupancy values to consider: ' +
                             ', '.join([a.name.lower()
                                        for a in Data.OccupancyVals]))
    data_params.add_argument('--resolution', type=float, default=0.02,
                             help='')
    data_params.add_argument('--angle-cells', type=float, default=56,
                             help='Number of angular cells')
    data_params.add_argument('--radius-min', type=float, default=0.3,
                             help='')
    data_params.add_argument('--radius-max', type=float, default=5.0,
                             help='')
    data_params.add_argument('--radius-factor', type=float, default=1.15,
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
    model_params.add_argument('--view-top-mixtures', type=int, default=8,
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
    learn_params.add_argument('--stop-condition', type=float, default=0.05,
                              help='Min likelihood change between epochs')
    learn_params.add_argument('--weight-init', type=str, default='random',
                              help='Weight init value: ' +
                              ', '.join([a.name.lower()
                                         for a in PlaceSubModel.WeightInitValue]))
    learn_params.add_argument('--init-accum', type=int, default=20,
                              help='Initial accumulator value')
    learn_params.add_argument('--smoothing-val', type=float, default=0.0,
                              help='Additive smoothing value')
    learn_params.add_argument('--smoothing-min', type=float, default=0.0,
                              help='Additive smoothing min value')
    learn_params.add_argument('--smoothing-decay', type=float, default=0.2,
                              help='Additive smoothing decay')
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

    # Parse
    args = parser.parse_args()

    # Check
    if args.submodel_class not in KNOWN_CLASSES:
        print("ERROR: Incorrect class ('%s')." % args.submodel_class)
        sys.exit(1)
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
        args.view_input_dist = spn.DenseSPNGenerator.InputDist[
            args.view_input_dist.upper()]
    except KeyError:
        print("ERROR: Incorrect view-input-dist ('%s')." % args.view_input_dist)
        sys.exit(1)
    try:
        args.weight_init = PlaceSubModel.WeightInitValue[args.weight_init.upper()]
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
    print("* Class: %s" % args.submodel_class)
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
    print("* Stopping condition: %s" % args.stop_condition)
    print("* Weight initialization: %s" % args.weight_init)
    print("* Initial accumulator value: %s" % args.init_accum)
    print("* Smoothing value: %s" % args.smoothing_val)
    print("* Smoothing min value: %s" % args.smoothing_min)
    print("* Smoothing decay: %s" % args.smoothing_decay)
    print("* EM upwards pass inference: %s" % args.value_inference)

    print("\nTesting parameters:")
    print("* Mask seed: %s" % args.mask_seed)

    print("\nOther:")
    print("* Save masked scans: %s" % args.save_masked)


def create_directories(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'mpe_states'), exist_ok=True)
    if args.save_masked:
        os.makedirs(os.path.join(args.results_dir, 'masked_scans'), exist_ok=True)


def main():
    args = parse_args()
    print_args(args)
    create_directories(args)

    data = Data(args.angle_cells, args.radius_min,
                args.radius_max, args.radius_factor)
    data.load(args.data_dir)
    data.process(args.submodel_class, args.subset, args.occupancy_vals)
    data.print_info()
    data.generate_masked_scans(args.mask_seed)
    data.save_masked_scans(args.results_dir)
    if args.save_masked:
        data.visualize_masked_scans(
            os.path.join(args.results_dir, 'masked_scans'))

    # Model
    model = PlaceSubModel(data=data,
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
                          init_accum_value=args.init_accum,
                          additive_smoothing_value=args.smoothing_val,
                          value_inference_type=args.value_inference)
    model.train(stop_condition=args.stop_condition,
                additive_smoothing_min=args.smoothing_min,
                additive_smoothing_decay=args.smoothing_decay,
                results_dir=args.results_dir)
    model.test(args.results_dir)


if __name__ == '__main__':
    main()
