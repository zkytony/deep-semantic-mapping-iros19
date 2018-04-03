#!/usr/bin/env python

from spn_topo.tests.tbm.factor_graph.run_factor_graph_tests import FactorGraphTest, test_one_sequence
from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate
from spn_topo.tests.tbm.runner import TbmExperiment, random_policy, random_fixed_policy, get_noisification_level
from spn_topo.util import CategoryManager, ColdDatabaseManager, print_banner, print_in_box

import argparse
import time
from pprint import pprint
import random
import os
import json
import re

from spn_topo.tests.constants import COLD_ROOT, TOPO_MAP_DB_ROOT, BP_EXEC_PATH, BP_RESULT_DIR
 
# HIGH = (0.99999, 0.9999995)#(0.5, 0.7)
# LOW = (0.05, 0.09)#(0.01, 0.02)

def read_test_case_file(path):
    """Reads the test case file from `path`. Returns a dictionary of
    three keys:
       'likelihoods': likelihoods setting for the nodes.
       'query': the masked graph.
       'true': the groundtruth.
    Note that the dictionary returned is created by JSON. So keys are
    """
    def add_quotes(matchobj):
        return '"' + matchobj.group(0)[:-1] + '":'
    
    with open(path) as f:
        string = ""
        for line in f.readlines():
            string += line
        string = string.replace("'", '"')
        string = string.replace("\n", "")
        string = string.replace("(", "[")
        string = string.replace(")", "]")
        string = re.sub(r'[0-9]+:', add_quotes, string)
        test_case = json.loads(string)
    return test_case

def record_stats(stats, overall_stats):
    for k in stats:
        if type(k) == str and not k.startswith('_'):
            if k not in overall_stats:
                overall_stats[k] = [0,0,0]
            # k is a class abbreviation
            overall_stats[k][0] += stats[k][0]
            overall_stats[k][1] += stats[k][1]
            overall_stats['_total_correct_'] += stats[k][0]
            overall_stats['_total_inferred_'] += stats[k][1]
            if overall_stats[k][1] != 0:
                overall_stats[k][2] = overall_stats[k][0] / overall_stats[k][1]
    overall_stats['_cases_'].append(stats)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run factor graph on one sequence')
    parser.add_argument('db_abrv', type=str, help="Database name. sa=Saarbrucken, fr=Freiburg, st=Stockholm")
    parser.add_argument('seq_id', type=str, help="sequence id")
    parser.add_argument('-n', '--num-runs', type=int, help="number of rounds of tests on this sequence. Default: 10.", default=10)
    parser.add_argument('-v', '--visualize', action="store_true", help="Visualize groundtruth, query, and result")
    parser.add_argument('-r', '--rate-occluded', type=float, help="Percentage of randomly occluded nodes. Default is 0.2", default=0.2)
    parser.add_argument('-infph', '--inferplaceholder-exp', action="store_true", help="Run infer placeholder experiment.")
    parser.add_argument('-i', '--inference-type', type=str, help="Inference type. Can be 'mpe' or 'marginal'. Default is 'marginal'. Must be lower-case.", default="marginal")
    parser.add_argument('-l', '--likelihoods', action="store_true", help="Use likelihood or not.")
    parser.add_argument('-lo-c', '--low-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-c', '--high-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-lo-ic', '--low-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-ic', '--high-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-t', '--test-case-path', type=str, help="Path to test case. The test case file has the same format as `test_instance.log`. Overrides the -n option.")
    parser.add_argument('-unif', '--uniform-for-incorrect', action="store_true", help="Set likelihoods for nodes labeled incorrect to be uniform")
    parser.add_argument('-seg', '--segment', action="store_true", help="Segment the graph such that each node is a room, instead of a place.")
    parser.add_argument('-tr', '--triplet', action="store_true", help="Use 3-element factors instead of pairwise factors for room connectivity.")
    parser.add_argument('-nov', '--novelty', action="store_true", help="Runs novelty detection task.")

    
    db_name = None
    db_count = []
    args = parser.parse_args()
    if args.db_abrv == 'fr':
        db_name = "Freiburg"
        db_count = ["Stockholm", "Saarbrucken"]
    elif args.db_abrv == 'st':
        db_name = "Stockholm"
        db_count = ["Freiburg", "Saarbrucken"]
    elif args.db_abrv == 'sa':
        db_name = "Saarbrucken"
        db_count = ["Stockholm", "Freiburg"]
    # Train on two. Test on one of trained buildings.
    elif args.db_abrv == 'sa-st':
        db_count = ['Freiburg', 'Stockholm']
        db_name = 'Stockholm'
    elif args.db_abrv == 'sa-fr':
        db_count = ['Freiburg', 'Stockholm']
        db_name = 'Freiburg'
    elif args.db_abrv == 'st-sa':
        db_count = ['Freiburg', 'Saarbrucken']
        db_name = 'Saarbrucken'
    elif args.db_abrv == 'st-fr':
        db_count = ['Freiburg', 'Saarbrucken']
        db_name = 'Freiburg'
    elif args.db_abrv == 'fr-st':
        db_count = ['Stockholm', 'Saarbrucken']
        db_name = 'Stockholm'
    elif args.db_abrv == 'fr-sa':
        db_count = ['Stockholm', 'Saarbrucken']
        db_name = 'Saarbrucken'
    else:
        raise ValueError("Unrecognized database abbreviation %s" % args.db_abrv)

    seq_id = args.seq_id
    print("loading data...")
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load(db_name="Stockholm", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment)
    dataset.load(db_name="Saarbrucken", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment)
    dataset.load(db_name="Freiburg", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment)

    ft = FactorGraphTest(BP_EXEC_PATH, dataset, db_count, db_name, result_dir=BP_RESULT_DIR,
                         use_triplet=args.triplet, infer_type=args.inference_type)
    
    overall_stats = {'_overall_':0, '_total_correct_': 0, '_total_inferred_': 0,
                     '_failed_instances_': 0, '_total_instances_': 0, '_no_prob_': 0, '_cases_':[]}

    if args.test_case_path:
        topo_map = dataset.get_topo_maps(db_name=db_name, seq_id=seq_id)[seq_id]
        test_case = read_test_case_file(args.test_case_path)
        groundtruth = {int(k):test_case['true'][k] for k in test_case['true']}
        assert groundtruth == topo_map.current_category_map()  # assert usability of this test case.
        masked = {int(k):test_case['query'][k] for k in test_case['query']}
        likelihoods = None
        if args.likelihoods:
            likelihoods = {int(k):tuple(test_case['likelihoods'][k]) for k in test_case['likelihoods']}
        if not os.path.exists(os.path.join(ft._result_dir, seq_id)):
            os.makedirs(os.path.join(ft._result_dir, seq_id))
        result_path = os.path.join(ft._result_dir, seq_id, "instance_%s_%s_Inf.json" % (db_name, seq_id))
        result, stats = ft.run_instance(seq_id, topo_map, masked, groundtruth, likelihoods=likelihoods,
                                        visualize=args.visualize, result_path=result_path, consider_placeholders=args.inferplaceholder_exp)
        overall_stats['_total_instances_'] += 1
        if stats is None:
            overall_stats['_failed_instances_'] += 1
        record_stats(stats, overall_stats)

    else:
        # Parameters.
        test_kwargs = {
            'visualize': args.visualize,
            'uniform_for_incorrect': args.uniform_for_incorrect,
        }
        if args.likelihoods or args.inferplaceholder_exp:
            test_kwargs.update({
                'high_likelihood_correct': eval(args.high_likelihood_correct),
                'low_likelihood_correct': eval(args.low_likelihood_correct),
                'high_likelihood_incorrect': eval(args.high_likelihood_incorrect),
                'low_likelihood_incorrect': eval(args.low_likelihood_incorrect),    
            })
        for i in range(args.num_runs):
            stats = test_one_sequence(db_name+"-"+seq_id,
                                      ft, random_fixed_policy, policy_args={'rate_occluded': args.rate_occluded},
                                      run_correction_task=args.likelihoods and not args.inferplaceholder_exp,
                                      novelty=args.novelty, **test_kwargs)
            overall_stats['_total_instances_'] += 1
            if stats is None:
                overall_stats['_failed_instances_'] += 1
                continue
            record_stats(stats, overall_stats)

    overall_stats['_overall_'] = overall_stats['_total_correct_'] / max(1, overall_stats['_total_inferred_'])
            
    timestamp = time.strftime("%Y%m%d-%H%M")
    with open(os.path.join(BP_RESULT_DIR, "overall_report_%s_%s.log" % (seq_id, timestamp)), "w") as f:
        pprint(overall_stats, stream=f)
    pprint(overall_stats)

    # Print noisification level (if involving likelihoods)
    if args.likelihoods:
        print("--- Noisifcation Level ---")
        pprint(get_noisification_level(eval(args.high_likelihood_correct), eval(args.low_likelihood_correct),
                                       eval(args.high_likelihood_incorrect), eval(args.low_likelihood_incorrect),
                                       uniform_for_incorrect=args.uniform_for_incorrect))
    print("Done.")
    
