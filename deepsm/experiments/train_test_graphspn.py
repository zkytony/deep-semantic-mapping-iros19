#!/usr/bin/env python3

import argparse
import sys

from deepsm.experiments.custom_graph_experiment import custom_graph
from deepsm.experiments.cold_database_experiment import same_building, across_buildings
from deepsm.experiments.synthetic_experiment import synthetic

def main():
    available_commands = {
        'samebuilding': same_building,
        'acrossbuildings': across_buildings,
        'customgraph': custom_graph,
        'synthetic': synthetic
    }
    parser = argparse.ArgumentParser(description="Run GraphSPN experiments in"\
                                     "the full spatial knowledge framework",
                                     usage="%s <command> [<args>]]" % sys.argv[0])
    parser.add_argument("command", help="What command to run. Commands: %s" % sorted(available_commands.keys()))
    args = parser.parse_args(sys.argv[1:2])  # Exclude the rest of args to focus only on <command>.
    
    if args.command not in available_commands:
        print("Unrecognized command %s" % args.command)
        parser.print_help()
        sys.exit(1)

    # Run command
    available_commands[args.command]()


if __name__ == "__main__":
    main()
