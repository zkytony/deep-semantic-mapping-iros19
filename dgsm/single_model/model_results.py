#!/usr/bin/env python3

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import os
import sys
import argparse
from data import Data
from result import SubModelResult, Results

KNOWN_CLASSES = ['corridor', 'door', 'small_office', 'large_office']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate result stats.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is stored')
    parser.add_argument('results_dir', type=str,
                        help='Directory where all results are saved')
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

    # Parse
    args = parser.parse_args()

    # Check
    try:
        args.occupancy_vals = Data.OccupancyVals[
            args.occupancy_vals.upper()]
    except KeyError:
        print("ERROR: Incorrect occupancy-vals ('%s')." % args.occupancy_vals)
        sys.exit(1)

    return args


def print_args(args):
    print("---------")
    print("Arguments")
    print("---------")
    print("* Data dir: %s" % args.data_dir)
    print("* Results dir: %s" % args.results_dir)
    print("* Subset: %s" % args.subset)

    print("\nData parameters:")
    print("* Occupancy values: %s" % args.occupancy_vals)
    print("* Resolution: %s" % args.resolution)
    print("* Angle cells: %s" % args.angle_cells)
    print("* Min radius: %s" % args.radius_min)
    print("* Max radius: %s" % args.radius_max)
    print("* Radius factor: %s" % args.radius_factor)


def main():
    args = parse_args()
    print_args(args)

    # Load and process data for each class
    print("\nLoading data:")
    datas = []
    for kc in KNOWN_CLASSES:
        data = Data(args.angle_cells, args.radius_min,
                    args.radius_max, args.radius_factor)
        data.load(args.data_dir)
        data.process(kc, args.subset, args.occupancy_vals)
        datas.append(data)
    print("Done!")

    # Print number of training scans for each class
    print("\nNumber of training scans per class:")
    for data in datas:
        print("- %s: %s" % (data.submodel_class, len(data.training_scans)))

    # Load all subclass results
    print("\nLoading results:")
    submodel_results = []
    for kc in KNOWN_CLASSES:
        submodel_result = SubModelResult(args.results_dir, kc)
        submodel_results.append(submodel_result)
    print("Done!")

    # Process all results
    print("\nProcessing results:")
    results = Results(datas, submodel_results, args.results_dir)
    results.get_completion_ratios()
    results.get_classification_results()
    results.get_clustering_results()
    results.get_novelty_results()
    print("Done!")


if __name__ == '__main__':
    main()
