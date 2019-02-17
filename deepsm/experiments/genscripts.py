#!/usr/bin/env python3
# Generate shell scripts. This avoids a lot of error and effort when
# manually creating scripts.

import yaml
import argparse
import sys, os
import copy

def read_paramcfg_file(filepath):
    with open(filepath) as f:
        cfg = yaml.load(f)
    globalcfg = {cfg['_global']['options'][i]:cfg['_global']['default_values'][i]
                 for i in range(len(cfg['_global']['param_names']))}
    # Returns a list of parameter settings for the command line program.
    settings = []

    for exp_name in cfg:
        if exp_name.startswith("_"):
            continue
        expcfg = cfg[exp_name]
        for value in expcfg['values']:
            setting = copy.deepcopy(globalcfg)  # a setting maps from an option to a value
            for i in range(len(value)):
                param_indx = expcfg['param_indices'][i]
                option = cfg['_global']['options'][param_indx]
                setting[option] = value[i]
            # convert setting into a string and append it to `settings`
            options_str = " "
            for option in setting:
                if type(setting[option]) is bool:
                    options_str += option + " "
                else:
                    options_str += option + " " + str(setting[option]) + " "
            settings.append(options_str)
    return settings


def handle_dgsm():
    parser = argparse.ArgumentParser("Generate shell scripts to run DGSM experiments")
    parser.add_argument("db", type=str, help="Building. e.g .Stockholm")
    parser.add_argument("paramcfg_file", type=str, help="Path to a file that defines what parameters"\
                        "you want to vary and what values you want to set them to. A controlled experiment"\
                        "will be performed for those parameters. You can also vary multiple parameters"\
                        "together. See more in code. Parameters are for training.")
    parser.add_argument("save_dir", type=str, help="Path to directory to save the shell script.")
    parser.add_argument("exppy_dir", type=str, help="Path to directory where experiment python code is stored, relative to save_dir")
    parser.add_argument("cases", type=str, nargs="+", help="Case, e.g. 456-7. If 'full' will"\
                        "enumerate all cases for the given building")
    parser.add_argument("--category-type", type=str, help="either SIMPLE, FULL, or BINARY", default="SIMPLE")
    parser.add_argument("--gpus", type=int, nargs="+", help="Integer ids for gpus. The commands"\
                        "will be split evenly among them")
    args = parser.parse_args(sys.argv[2:])

    cases = {
        "Stockholm": ["456-7", "457-6", "567-4"],
        "Freiburg": ["12-3", "23-1", "13-2"],
        "Saarbrucken": ["123-4", "124-3", "134-2", "234-1"]
    }
    for c in args.cases:
        if c not in cases[args.db]:
            print("Case %s not recognized for %s" % (args.cases, args.db))

    settings = read_paramcfg_file(args.paramcfg_file)
    commands = []
    for options_str in settings:
        for test_case in args.cases:
            command = "%s DGSM_SAME_BUILDING -d %s --config \"{'test_case':'%s',"\
                      "'category_type':'%s', 'training_params':'%s'}\""\
                      % (os.path.join(args.exppy_dir, "train_test_dgsm_full_model.py"),
                         args.db,
                         test_case,
                         args.category_type,
                         options_str)
            commands.append(command)

    comands_by_gpu = [commands]
    if args.gpus is not None:
        commands_by_gpu = []
        n = int(len(commands) / len(args.gpus))
        for i, gpu in enumerate(args.gpus):
            if i < len(args.gpus) - 1:
                commands_by_gpu.append(commands[i*n:(i+1)*n])
            else:
                commands_by_gpu.append(commands[i*n:])
    for i in range(len(commands_by_gpu)):
        commands_str = "\n".join(commands_by_gpu[i])
        filename = os.path.join(args.save_dir, "run_%s_gpu_%d.sh"
                                % (os.path.basename(args.paramcfg_file),
                                   args.gpus[i]))
        with open(filename, "w") as f:
            f.write("set -x\n")
            f.write(commands_str)
            f.write("set +x\n")
            os.chmod(filename, 0o777)
        print("Written commands to %s" % filename)
    

def handle_graphspn():
    parser = argparse.ArgumentParser("Generate shell scripts to run GraphSPN experiments")
    args = parser.parse_args(sys.argv[2:])


def main():
    available_commands = {
        'dgsm': handle_dgsm,
        'graphspn': handle_graphspn
    }
    
    parser = argparse.ArgumentParser(description="Generate shell scripts to run experiments",
                                     usage="genscripts.py <command> [<args>]")
    parser.add_argument("command",
                        help="What command to run.. Commands: %s" % sorted(available_commands.keys()))
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in available_commands:
        print("Unrecognized command %s" % args.command)
        parser.print_help()
        sys.exit(1)

    # Run command
    print(args.command)
    available_commands[args.command]()

if __name__ == "__main__":
    main()
