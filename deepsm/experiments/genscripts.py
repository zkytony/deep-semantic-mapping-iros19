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
    globalcfg = {cfg['_global']['param_names'][i]:(cfg['_global']['default_values'][i],
                                                   cfg['_global']['options'][i])
                 for i in range(len(cfg['_global']['param_names']))}
    # Returns a list of parameter settings for the command line program.
    settings = []
    if "_run_global_case" in cfg and cfg["_run_global_case"]:
        settings.append(copy.deepcopy(globalcfg))

    for exp_name in cfg:
        if exp_name.startswith("_"):
            continue
        expcfg = cfg[exp_name]
        for value in expcfg['values']:
            setting = copy.deepcopy(globalcfg)  # a setting maps from an option to a value
            for i in range(len(value)):
                param_indx = expcfg['param_indices'][i]
                param_name = cfg['_global']['param_names'][param_indx]
                setting[param_name] = (value[i], cfg['_global']['options'][param_indx])
            settings.append(setting)
    return settings


def proc_cases(arg_cases, arg_db):
    casesdb = {
        "Stockholm": ["456-7", "457-6", "467-5", "567-4"],
        "Freiburg": ["12-3", "23-1", "13-2"],
        "Saarbrucken": ["123-4", "124-3", "134-2", "234-1"]
    }
        
    cases = []
    for c in arg_cases:
        if c == "full":
            cases = casesdb[arg_db]
            break
        elif c not in casesdb[arg_db]:
            print("Case %s not recognized for %s" % (arg_cases, arg_db))
            return
        else:
            cases.append(c)
    return cases

def divide_commands_by_gpu(commands, gpus):
    commands_by_gpu = []
    n = int(round(len(commands) / float(len(gpus)) +  0.001))
    for i, gpu in enumerate(gpus):
        if i < len(gpus) - 1:
            commands_by_gpu.append(commands[i*n:(i+1)*n])
        else:
            commands_by_gpu.append(commands[i*n:])
    return commands_by_gpu

def save_commands_to_files(filename_prefix, commands_by_gpu, gpus):
    for i in range(len(commands_by_gpu)):
        commands_str = "\n".join(commands_by_gpu[i])
        filename = "%s_gpu_%d.sh" % (filename_prefix, gpus[i])
        with open(filename, "w") as f:
            f.write("set -x\n")
            f.write(commands_str)
            f.write("\nset +x\n")
            os.chmod(filename, 0o777)
        print("Written commands to %s" % filename)




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
    cases = proc_cases(args.cases, args.db)

    # Generate commands
    settings = read_paramcfg_file(args.paramcfg_file)
    commands = []
    for setting in settings:
        for test_case in cases:
            option_str = ""
            for param_name in setting:
                value = setting[param_name][0]
                option = setting[param_name][1]
                if type(value) is bool:
                    option_str += option + " "
                else:
                    option_str += option + " " + str(value) + " "
            
            command = "%s DGSM_SAME_BUILDING -d %s --config \"{'test_case':'%s',"\
                      "'category_type':'%s', 'training_params':'%s'}\""\
                      % (os.path.join(args.exppy_dir, "train_test_dgsm_full_model.py"),
                         args.db,
                         test_case,
                         args.category_type,
                         option_str)
            commands.append(command)

    if args.gpus is None:
        commands_by_gpu = [commands]
    else:
        commands_by_gpu = divide_commands_by_gpu(commands, args.gpus)
        
    filename = os.path.join(args.save_dir, "run_%s"
                            % (os.path.basename(args.paramcfg_file)))
    save_commands_to_files(filename, commands_by_gpu, args.gpus)
    

def handle_graphspn():
    parser = argparse.ArgumentParser("Generate shell scripts to run GraphSPN experiments")
    parser.add_argument("db", type=str, help="Building. e.g .Stockholm")
    parser.add_argument("exp_name", type=str, help="Experiment name. Should be unique for grouping results")
    parser.add_argument("exp_case", type=str, help="Experiment type. For example Classification, Novelty.")
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
    cases = proc_cases(args.cases, args.db)

    # Generate commands
    settings = read_paramcfg_file(args.paramcfg_file)
    commands = []
    for setting in settings:
        for test_case in cases:
            test_floor = test_case.split("-")[1]

            template = setting["template"][0]
            num_partitions = setting["num_partitions"][0]

            # batch_size, epoch, likelihood_thresholds (fmt: 0001 means 0.001), sampling method, dgsm_lh
            trial_cfg_str = "B%dE%dlh%sm%s"\
                            % (setting["batch_size"][0], setting["epochs"][0],
                               str(setting["likelihood_thres"][0]).replace(".",""),
                               "RANDOM" if setting["random_sampling"][0] else "")
            trial_cfg_str += "dgsmLh" if setting["train_with_likelihoods"][0] else ""
            test_name = "full%s%sP%dT0%strain%s" % (template, args.exp_case,
                                                    num_partitions,
                                                    args.db, trial_cfg_str)
                                                    
            option_str = "--expr-case %s " % args.exp_case\
                         + "-d %s " % args.db\
                         + "-e %s " % args.exp_name\
                         + "-t %s " % test_name\
                         + "--test-floor %s " % test_floor\
                         + " "
            for param_name in setting:
                value = setting[param_name][0]
                option = setting[param_name][1]
                if type(value) is bool:
                    option_str += option + " "
                else:
                    option_str += option + " " + str(value) + " "            
            
            command = "%s DGSM_SAME_BUILDING %s"\
                      % (os.path.join(args.exppy_dir, "train_test_graphspn_colddb_multiple_sequences.py"),
                         option_str)
            commands.append(command)

    # save commands to files
    if args.gpus is None:
        commands_by_gpu = [commands]
    else:
        commands_by_gpu = divide_commands_by_gpu(commands, args.gpus)
        
    filename = os.path.join(args.save_dir, "run_%s"
                            % (os.path.basename(args.paramcfg_file)))
    save_commands_to_files(filename, commands_by_gpu, args.gpus)


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
