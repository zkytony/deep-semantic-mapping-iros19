#!/usr/bin/env python3

import os
import argparse
from data import Data


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize scans.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is stored')
    parser.add_argument('output_dir', type=str,
                        help='Directory where visualizations are saved')

    # Data params
    data_params = parser.add_argument_group(title="data parameters")
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
    return args


def print_args(args):
    print("---------")
    print("Arguments")
    print("---------")
    print("* Data dir: %s" % args.data_dir)
    print("* Output dir: %s" % args.output_dir)


def create_directories(args):
    os.makedirs(args.output_dir, exist_ok=True)


def main():
    args = parse_args()
    print_args(args)
    create_directories(args)

    data = Data(args.angle_cells, args.radius_min,
                args.radius_max, args.radius_factor)
    data.load(args.data_dir)
    data.visualize_data(args.output_dir)


if __name__ == '__main__':
    main()
