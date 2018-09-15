#!/usr/bin/env python
#
# Generate appropriate .fg files and run C++ factor graph executive
# and parse the output.
#
# author: Kaiyu Zheng

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
#matplotlib.use('Agg')

from deepsm.graphspn.tbm.dataset import TopoMapDataset
from deepsm.graphspn.tbm.topo_map import Edge
from deepsm.graphspn.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate
from deepsm.graphspn.tests.tbm.runner import TbmExperiment, doorway_policy, random_policy, random_fixed_policy, random_fixed_plus_placeholders, get_noisification_level
from deepsm.util import CategoryManager, ColdDatabaseManager, print_banner, print_in_box
import csv
import matplotlib.pyplot as plt
import os
import pprint
import random
import subprocess
import time
import json
from pprint import pprint
import argparse

from deepsm.experiments.common import COLD_ROOT, TOPO_MAP_DB_ROOT, BP_EXEC_PATH, BP_RESULTS_ROOT, GROUNDTRUTH_ROOT

COUNT = 0

def get_category_map_from_lh(lh):
    """lh is a dictionary { nid -> [ ... likelihood for class N... ]}"""
    category_map = {}   # nid -> numerical value of the category with highest likelihood
    for nid in lh:
        class_index = np.argmax(lh[nid])
        category_map[nid] = class_index
    return category_map

class FactorGraphTest:

    COUNT = 0

    def __init__(self, bp_exec_path, dataset, db_count, db_test, result_dir=None,
                 tiny_dataset=None, ftable_path=None, use_triplet=False, infer_type="marginal"):
        """
        bp_exec_path (str): Path to the factor graph executable file, which takes
                            two arguments a .fg file and a .tab file, and optionally
                            an output path, and it will save a JSON file for the
                            MAP inference on the factor graph given the evidence.
                            The result is a map from variable label to category number.
        dataset (TopoMapDataset): a loaded database.
        db_count (list): List of database names (e.g. Stockholm) to load topological
                         map data for computing factor values based on counting.
        db_test (str): A string of database name (e.g. Stockholm) to load topological
                        map data for actually creating factor graphs and run
                        loopy BP and MAP inference.
        result_dir (str): Path to directory to save results. If None, use "./fg_results".
        ftable_path (str): See doc of _create_room_connectivity_factor().
        """
        self._bp_exec_path = bp_exec_path
        self._dataset = dataset
        self._tiny_dataset = tiny_dataset
        self._db_count = db_count
        self._db_test = db_test
        self._result_dir = result_dir if result_dir is not None else "./r"
        self._infer_type = infer_type
        if not os.path.exists(self._result_dir):
            os.makedirs(self._result_dir)

        self._use_triplet = use_triplet
        if not use_triplet:
            self._rc_factor = self._create_room_connectivity_factor(ftable_path=ftable_path)
        else:
            self._tr_factor = self._create_room_triplet_factor(ftable_path=ftable_path)


    def get_test_seq_ids(self, amount=-1):
        if self._tiny_dataset is not None:
            return self._tiny_dataset.get_topo_maps(db_name=self._db_test, amount=amount).keys()
        else:
            return self._dataset.get_topo_maps(db_name=self._db_test, amount=amount).keys()

    def _get_topo_maps(self, seqs=None):
        topo_maps = {}
        if seqs is not None:
            for db_seq_id in seqs:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = list(self._dataset.get_topo_maps(db_name=db_name, seq_id=seq_id).values())[0]
        else: # Load from db_count
            for db_name in self._db_count:
                for seq_id in self._dataset.get_topo_maps(db_name=db_name, amount=-1):
                    topo_maps[db_name+"-"+seq_id] = list(self._dataset.get_topo_maps(db_name=db_name, seq_id=seq_id).values())[0]
        return topo_maps

    @staticmethod
    def normalize_factor(factor_orig):
        # Normalize
        factor = dict(factor_orig)
        total = sum(factor.values())
        for entry in factor:
            factor[entry] = factor[entry] / total
        return factor
    

    def _create_room_connectivity_factor(self, ftable_path=None, uniform=False, seqs=None):
        """
        Creates a of factor table between any two room categories, using the training (counting) database.
        
        If `ftable_path` is provided, instead of counting from the database, will use the factors provided
        in the table instead. The row format for this factor table file is:

           x, y, f(X=x, Y=y)

        where X, Y are variables, and x, y are stats, and f(.) is the factor function that outputs a real
        number.
        """
        factor = {}

        # Check if we want to load factors from file.
        if ftable_path is not None:
            with open(ftable_path) as f:
                csvreader = csv.reader(f, delimiter=',', quotechar='"')
                for row in csvreader:
                    pair = (int(row[0]), int(row[1]))
                    likelihood = float(row[2])
                    factor[pair] = likelihood
                return factor

        if uniform:
            # We just want uniform factor
            for a in range(CategoryManager.NUM_CATEGORIES):
                for b in range(CategoryManager.NUM_CATEGORIES):
                    if a <= b:  # agree with the requirement: a <= b
                        factor[(a,b)] = 1
            return FactorGraphTest.normalize_factor(factor)

        # Count occurrences of each possible pair
        # First, get all topo maps.
        topo_maps = self._get_topo_maps(seqs=seqs)
                
        # Iterate over every topo map - every edge
        for db_seq_id in topo_maps:
            topo_map = topo_maps[db_seq_id]
            for eid in topo_map.edges:
                edge = topo_map.edges[eid]
                # Get pair of category number
                pair = (CategoryManager.category_map(edge.nodes[0].label),
                        CategoryManager.category_map(edge.nodes[1].label))
                if pair in factor:
                    factor[pair] += 1
                elif tuple(reversed(pair)) in factor:
                    factor[tuple(reversed(pair))] += 1
                else:
                    # We insert a new pair combination (a, b) into the factor. Note we want a <= b.
                    if pair[0] > pair[1]:
                        factor[tuple(reversed(pair))] = 1
                    else:
                        factor[pair] = 1    
                        
        for a in range(CategoryManager.NUM_CATEGORIES):
            for b in range(CategoryManager.NUM_CATEGORIES):
                if a <= b:  # agree with the requirement above: a <= b
                    if (a,b) not in factor:
                        factor[(a,b)] = 1  # Smoothing. Avoid 0 factor.

        return FactorGraphTest.normalize_factor(factor)
                        
    def _create_room_triplet_factor(self, ftable_path=None, uniform=False, seqs=None):
        """
        Similar to _create_room_connectivity_factor, but each row in the factor table
        has three variables. This is created by counting occurrance of 3-node structure
        identified by a pair of edge connected to one node. Symmetry is enforced.
        """
        factor = {}

        # Check if we want to load factors from file.
        if ftable_path is not None:
            with open(ftable_path) as f:
                csvreader = csv.reader(f, delimiter=',', quotechar='"')
                for row in csvreader:
                    triplet = (int(row[0]), int(row[1]), int(row[2]))
                    likelihood = float(row[3])
                    factor[triplet] = likelihood
                return factor

        if uniform:
            # We just want uniform factor
            for a in range(CategoryManager.NUM_CATEGORIES):
                for b in range(CategoryManager.NUM_CATEGORIES):
                    for c in range(CategoryManager.NUM_CATEGORIES):
                        factor[(a, b, c)] = 1  # Smoothing.
            return FactorGraphTest.normalize_factor(factor)

        # Iterate over every topo map used for counting
        topo_maps = self._get_topo_maps(seqs=seqs)
        for db_seq_id in topo_maps:
            topo_map = topo_maps[db_seq_id]
            node_edge_pairs = topo_map.connected_edge_pairs()
            for nid in node_edge_pairs:
                for edge_pair in node_edge_pairs[nid]:
                    triplet = Edge.get_triplet_from_edge_pair(topo_map, edge_pair, nid, catg=True)
                    rev_triplet = tuple(reversed(triplet))
                    if triplet not in factor:
                        factor[triplet] = 0
                        factor[rev_triplet] = 0
                    factor[triplet] += 1
                    factor[rev_triplet] += 1
        for a in range(CategoryManager.NUM_CATEGORIES):
            for b in range(CategoryManager.NUM_CATEGORIES):
                for c in range(CategoryManager.NUM_CATEGORIES):
                    if (a, b, c) not in factor:
                        factor[(a, b, c)] = 1  # Smoothing.
        return FactorGraphTest.normalize_factor(factor)
                        

    def _parse_marginal_results(self, result_path, varlabels):
        # Marginal inference:
        with open(result_path) as rf:
            strings = rf.read().replace('-nan', '\"NaN\"')
            marginal_result = json.loads(strings)
        res_catg_map = {}
        rev_varlabels = self._reverse_var_labels_map(varlabels)
        for label in marginal_result:
            n_label = int(label[1:])
            nid = rev_varlabels[n_label]
            if 'NaN' in marginal_result[label]:
                res_catg_map[nid] = -1  # Failed to converge to a probability distribution for this variable.
            else:
                res_catg_map[nid] = marginal_result[label].index(max(marginal_result[label]))  # Get the index of the max-valued probability
        return res_catg_map


    def _parse_mpe_results(self, result_path, varlabels, return_logscore=False):
        # MPE Inference
        # Read the result file. Return a category map from nid to category number
        with open(result_path) as rf:
            map_result = json.load(rf)
        rev_varlabels = self._reverse_var_labels_map(varlabels)
        res_catg_map = {}
        for label in map_result:
            if label[0] != "_":
                n_label = int(label[1:])
                nid = rev_varlabels[n_label]
                res_catg_map[nid] = map_result[label]
        if return_logscore:
            return res_catg_map, float(map_result['_logScore_'])
        return res_catg_map

    def _visualize_test_case(self, seq_id, topo_map, groundtruth=None, masked=None,
                             result=None, consider_placeholders=False, save_path=None):
        """Visualize"""
        def save_vis(topo_map, category_map, db_name, seq_id, save_path, name,  consider_placeholders):
            ColdMgr = ColdDatabaseManager(db_name, COLD_ROOT, gt_root=GROUNDTRUTH_ROOT)
            topo_map.assign_categories(category_map)
            rcParams['figure.figsize'] = 22, 14
            topo_map.visualize(plt.gca(), ColdMgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), consider_placeholders=consider_placeholders)
            if not os.path.exists(os.path.join(save_path, seq_id)):
                os.makedirs(os.path.join(save_path, seq_id))
            plt.savefig(os.path.join(save_path, seq_id, '%s_%s_%s.png' % (db_name, seq_id, name)))
            plt.clf()
            print("Saved %s visualization for %s:%s." % (name, db_name, seq_id))

        if save_path is None:
            save_path = self._result_dir
        if groundtruth:
            save_vis(topo_map, groundtruth, self._db_test, seq_id, save_path, 'groundtruth', False)
        if masked:
            save_vis(topo_map, masked, self._db_test, seq_id, save_path, 'query', consider_placeholders)
        if result:
            save_vis(topo_map, result, self._db_test, seq_id, save_path, 'result', consider_placeholders)


    def run_instance(self, seq_id, topo_map, masked, groundtruth, likelihoods=None,
                     result_path=None, visualize=False, consider_placeholders=False,
                     avoid_placeholders=False):
        """
        Runs on the given test case for `seq_id`, described by `masked`, and `likelihoods`(if provided).

        Args:
        masked (dict): a dictionary from node id to category number. May contain -1.
        groundtruth (dict): a dictionary from node id to category number. Does not contain -1.
        likelihoods (dict): a dictionary {K:V} where K is node id, and V is a tuple of N
                            elements where N is the number of semantic classes, and a value
                            is the likelihood that the corresponding node belongs to this class.
                            Assume that the order of the class in the tuple is from 0 - NUM_CATEGORIES-1.
        consider_placeholder is True if we want to only compute the accuracy over the
            placeholder nodes.
        avoid_placeholders is True if we want to exclude placeholder nodes when computing
            the accuracy.
        Return:
        A tuple:
            1st element: Resulting category map by BP.
            2nd element: Statistics of accuracy
        """
        global COUNT
        varlabels = self._create_var_labels_map(topo_map)

        # Save the varlabels mapping for reference
        with open(os.path.join(result_path, "%s_%s_VarL_%d.json"
                               % (self._db_test, seq_id, COUNT)), 'w') as f:
            json.dump(varlabels, f, indent=4, sort_keys=True)
        
        # Create a .fg file
        if likelihoods:
            fg_path = self._create_fg(topo_map, seq_id, varlabels, likelihoods=likelihoods)
            tab_path = self._create_tab(topo_map, seq_id, varlabels, masked, likelihoods=likelihoods)

        else:
            fg_path = self._create_fg(topo_map, seq_id, varlabels)
            tab_path = self._create_tab(topo_map, seq_id, varlabels, masked)
        

        if result_path is None:
            result_path = self._result_dir

        COUNT += 1
        infer_result_path = os.path.join(result_path, "%s_%s_Inf_%d.json" % (self._db_test, seq_id, COUNT))
        
        # Run the binary for BP
        bp_proc = subprocess.Popen([self._bp_exec_path, fg_path, tab_path, self._infer_type, infer_result_path])
        bp_proc.wait()

        if self._infer_type == "marginal":
            res_catg_map = self._parse_marginal_results(infer_result_path, varlabels)  # (Using marginal inference)
        else:
            res_catg_map = self._parse_mpe_results(infer_result_path, varlabels)  # (Using MPE inference)
        stats = self.compute_accuracy(groundtruth, masked, res_catg_map, used_likelihoods=likelihoods is not None,
                                      consider_placeholders=consider_placeholders,
                                      avoid_placeholders=avoid_placeholders, topo_map=topo_map)
        # If visualize: save some images
        if visualize:
            self._visualize_test_case(seq_id, topo_map, groundtruth=groundtruth, masked=get_category_map_from_lh(likelihoods),
                                      result=res_catg_map,
                                      save_path=result_path,
                                      consider_placeholders=avoid_placeholders or consider_placeholders)
        topo_map.reset_categories()
        return res_catg_map, stats


    def run_novelty_detection(self, db_seq_id=None, mix_ratio=0.8):
        """
        Runs the novelty detection task on "1PO-2PO", "1PO-CR", "DW-CR"

        db_seq_id (str) sequence id to run novelty detection task on. If None, will randomly select a sequence.
        """
        def run_case(self, topo_map, result_path, seq_id, varlabels, factor):
            fg_path = self._create_fg(topo_map, seq_id, varlabels, factor=factor)  # use default path
            tab_path = self._create_tab(topo_map, seq_id, varlabels, topo_map.current_category_map())  # use default path

            # Run the binary for BP
            bp_proc = subprocess.Popen([self._bp_exec_path, fg_path, tab_path, "mpe", result_path])
            bp_proc.wait()
            res_catg_map, logscore = self._parse_mpe_results(result_path, varlabels, return_logscore=True)
            return res_catg_map, logscore

        RUN_TRAIN=False
        
        # Count from 80% sequences. Test on a sequence randomly selected from the remaining 20%
        if db_seq_id:
            db_name = db_seq_id.split("-")[0]
            other_dbs = {'Freiburg', 'Saarbrucken', 'Stockholm'} - {db_name}
            if RUN_TRAIN:
                # Count on sequence's building and another building.
                count_seqs, tt = self._dataset.split(0.8, *([db_name, random.choice(list(other_dbs))])) # lazy way
                try: # We don't want to count db_seq_id. Move it out by swapping with a sequence in tt
                    idx = count_seqs.index(db_seq_id)
                    tt[0], count_seqs[idx] = count_seqs[idx], tt[0]
                except ValueError:
                    # Ok. The given sequence is in test.
                    pass
            else:
                # Count on other two buildings
                count_seqs, _ = self._dataset.split(1.0, *(list(other_dbs))) # lazy way
                        
        else:
            count_seqs, test_seqs = self._dataset.split(mix_ratio, *self._db_count)
            db_seq_id = test_seqs[0]
        db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
        
        if not self._use_triplet:
            self._rc_factor = self._create_room_connectivity_factor(seqs=count_seqs)
        else:
            self._tr_factor = self._create_room_triplet_factor(seqs=count_seqs)

        # Get test instance
        topo_map = list(self._dataset.get_topo_maps(db_name=db_name, seq_id=seq_id).values())[0]
        
        varlabels = self._create_var_labels_map(topo_map)
        stats = {}
        
        # Uniform
        if not self._use_triplet:
            factor_unif = self._create_room_connectivity_factor(uniform=True)
        else:
            factor_unif = self._create_room_triplet_factor(uniform=True)
        result_path = os.path.join(self._result_dir, "%s_%s_Inf_%s.json" % (db_name, seq_id, "uniform"))
        _, logscore = run_case(self, topo_map, result_path, seq_id, varlabels, factor_unif)
        stats['_uniform_'] = logscore

        # Cases
        for swapped_classes in [('1PO', '2PO'), ('1PO', 'CR'), ('DW', 'CR')]:
            c1, c2 = swapped_classes
            print("Swap %s and %s" % (c1, c2))
            topo_map.swap_classes(swapped_classes)
            query = topo_map.current_category_map()

            result_path = os.path.join(self._result_dir, "%s_%s_Inf_%s.json" % (db_name, seq_id, "-".join(swapped_classes)))
            res_catg_map, logscore = run_case(self, topo_map, result_path, seq_id, varlabels, None)
            stats[swapped_classes] = {'raw': logscore, 'normalized': -(logscore / stats['_uniform_'])}
            topo_map.reset_categories()

        # Groundtruth
        result_path = os.path.join(self._result_dir, "%s_%s_Inf_%s.json" % (db_name, seq_id, "groundtruth"))
        _, logscore = run_case(self, topo_map, result_path, seq_id, varlabels, None)
        stats['_groundtruth_'] = {'raw': logscore, 'normalized': -(logscore / stats['_uniform_'])}
        return stats, db_seq_id
            
    
    def run_task(self, seq_id=None, result_path=None, mask_policy=None, run_correction_task=False,
                 visualize=False, policy_args={},
                 # settings for test involving likelihoods:
                 high_likelihood_correct=(0.5, 0.7), low_likelihood_correct=(0.3, 0.5),
                 high_likelihood_incorrect=(0.5, 0.7), low_likelihood_incorrect=(0.3, 0.5),
                 uniform_for_incorrect=False, novelty=False):
        """
        Runs a test on factor graph for one instance of topological map.

        seq_id (str) if provided, use this sequence, provided it is in the
                     test db. Other wise, will run on a random sequence
        run_correction_task (bool) if True, run classification correction task. If false, run infer
                                   placeholders task.
        novelty (bool) if True, run novelty detection task. Will run on the given seq_id if the seq_id
                       has a db prefix.

        Returns:
           A tuple:
            1st element: Resulting category map by BP.
            2nd element: Statistics of accuracy
            3rd element: the seq_id used to do this experiment as well. (Has db prefix if running novelty task)
        """
        global COUNT
        # Novelty
        if novelty:
            db_seq_id = None
            if seq_id is not None:
                if "-" in seq_id:
                    db_seq_id = seq_id
            stats, db_seq_id = self.run_novelty_detection(db_seq_id=db_seq_id)
            return None, stats, db_seq_id

        # Get seq_id
        if seq_id is None:
            seq_id = random.sample(self._dataset.topo_maps[self._db_test].keys(), 1)[0]
            db_seq_id = self._db_test+"-"+seq_id
        else:
            if "-" in seq_id:    # seq_id has db prefix
                db_seq_id = seq_id
            else:
                db_seq_id = self._db_test+"-"+seq_id
        db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]

        # Get groundtruth
        if self._tiny_dataset:
            topo_map = self._tiny_dataset.get_topo_maps(db_name=db_name, seq_id=seq_id)[seq_id]
        else:
            topo_map = self._dataset.get_topo_maps(db_name=db_name, seq_id=seq_id)[seq_id]
        groundtruth = topo_map.current_category_map()

        # Get masked
        likelihoods = None
        consider_placeholders = False
        if not run_correction_task:
            print("Infer Placeholder task. Also noisify the graph.")
            topo_map.mask_by_policy(random_fixed_plus_placeholders, **policy_args)
            consider_placeholders = True
        else:
            topo_map.mask_by_policy(mask_policy, **policy_args)
        likelihoods = TbmExperiment.create_instance_spn_likelihoods(0, topo_map, groundtruth,
                                                                    high_likelihood_correct, low_likelihood_correct,
                                                                    high_likelihood_incorrect, low_likelihood_incorrect,
                                                                    uniform_for_incorrect=uniform_for_incorrect,
                                                                    consider_placeholders=consider_placeholders)
        masked = topo_map.current_category_map()

        if result_path is None:
            suffix = ""
            if likelihoods:
                suffix = "_lh"
            result_path = os.path.join(self._result_dir, "%s_%s_Inf_%d%s.json" % (self._db_test, seq_id, COUNT, suffix))
            COUNT += 1

        result, stats = self.run_instance(seq_id, topo_map, masked, groundtruth, likelihoods=likelihoods,
                                          result_path=result_path, visualize=visualize, consider_placeholders=consider_placeholders)
        return result, stats, seq_id
    

    def compute_accuracy(self, groundtruth, masked, result, used_likelihoods=False,
                         consider_placeholders=False,
                         avoid_placeholders=False, topo_map=None):
        """
        `groundtruth`, `masked`, `result` are all dictionaries mapping from node id to
        category number. `masked` may contain -1 as category numbers to indicate missing
        information.

        consider_placeholder is True if we want to only compute the accuracy over the
            placeholder nodes. If False, then accuracy is computed over all nodes.
        avoid_placeholders is True if we want to exclude placeholder nodes when computing
            the accuracy. If False, it depends on the value of consider_placeholder to
            determine which nodes are the accuracy computed over. TODO: This logic is too complex.

        Returns statistics as a dictionary:
        {
          <CAT_ABRV> : [#correct, #total, float (accuracy)],
          ...
        "_total_correct_": int
        "_total_inferred_": int
          "_overall_": float (accuracy)
        }
        """
        stats = {
            CategoryManager.category_map(c, rev=True): [0, 0, 0]
            for c in range(CategoryManager.NUM_CATEGORIES)
        }
        num_total_correct = 0
        num_total_cases = 0
        num_no_prob = 0  # number of cases where there is no valid probability distribution.
        try:
            for nid in result:
                # Determine if we will consider the node-inference case for nid.
                count_case = False
                if avoid_placeholders:
                    if topo_map.nodes[nid].placeholder:
                        continue # we do not count placeholders since avoid_placeholder is True
                if consider_placeholders:
                    if topo_map.nodes[nid].placeholder:
                        assert masked[nid] == -1
                        count_case = True
                elif used_likelihoods or masked[nid] == -1:
                    count_case = True
                if count_case:
                    catg_abrv = CategoryManager.category_map(groundtruth[nid], rev=True)
                    if groundtruth[nid] == result[nid]:
                        stats[catg_abrv][0] += 1
                        num_total_correct += 1
                    else:
                        if result[nid] == -1:  # indicates invalid prob. distr.
                            num_no_prob += 1
                    stats[catg_abrv][1] += 1
                    stats[catg_abrv][2] = stats[catg_abrv][0] / stats[catg_abrv][1]
                    num_total_cases += 1
            if num_total_cases != 0:
                stats['_overall_'] = num_total_correct / num_total_cases
            else:
                stats['_overall_'] = 0
            stats['_total_correct_'] = num_total_correct
            stats['_total_inferred_'] = num_total_cases

            # Compute accuracy by class
            accuracy_per_catg = []
            for catg in stats:
                if not catg.startswith("_"):
                    accuracy_per_catg.append(stats[catg][2])
            stats['_overall_by_class_'] = float(np.mean(accuracy_per_catg))
            stats['_stdev_by_class_'] = float(np.std(accuracy_per_catg))
                
            stats['_no_prob_'] = num_no_prob
        except Exception as ex:
            print(ex)
        return stats
                
        

    def _create_var_labels_map(self, topo_map):
        """
        varlabels (dict): dictionary that maps from nid to factor graph variable label.
        """
        varlabels = {}  # map from node id to variable label used in the fg
        _label = 0   # increment whenever sees a new nid
        # Map nid to a variable label
        for nid in topo_map.nodes:
            # increment __label
            if nid not in varlabels:
                varlabels[nid] = _label
                _label += 1
        return varlabels

    
    def _reverse_var_labels_map(self, varlabels):
        """
        Returns a dictionary from variable label to graph node id, by reversing
        the nid-label map created from _create_var_labels_map().
        """
        return {varlabels[k]:k for k in varlabels}


    def _create_fg(self, topo_map, seq_id, varlabels, likelihoods=None, path=None, factor=None):
        """
        Creates a .fg file and save it to `path` (if provided, otherwise by convention).
        User can optionally supply likelihoods to indicate noise on the topological map.
        
        topo_map (TopologicalMap): topo map to work with.
        varlabels (dict): dictionary that maps from nid to variable label (used in .fg file)
        likelihoods (dict): a dictionary {K:V} where K is node id, and V is a tuple of N
                            elements where N is the number of semantic classes, and a value
                            is the likelihood that the corresponding node belongs to this class.
                            Assume that the order of the class in the tuple is from 0 - NUM_CATEGORIES-1.
        path (str): Path to save the produced .fg file. If None, save the file to '{result_dir}/{seq_id}.fg'.
                    If likelihoods is supplied, then save it to '{result_dir}}/{seq_id}_lh.fg'
        factor (dict): If provided, will use this factor instead of the default one.
        """
        if path is None:
            if likelihoods is None:
                path = os.path.join(self._result_dir, self._db_test + "_" + seq_id + ".fg")
            else:
                path = os.path.join(self._result_dir, self._db_test + "_" + seq_id + "_lh.fg")
                
        ### Example .fg file format
        ### ----------------
        ### 4                   |  number of factors
        ###                     |  empty line
        ### 2                   |  number of variables in this factor (start of factor block)
        ### 0 2                 |  labels for variables (0 .. |X|-1)
        ### 2 2                 |  number of possible values for variables
        ### 4                   |  number of non-zero entries in the factor table
        ### 0  0.5              |  entry [x1=0, x2=0, value=0.5]
        ### 1  0.5              |  entry [x1=1, x2=0, value=0.5]
        ### 2  0.2              |  entry [x1=0, x2=1, value=0.5]
        ### 3  0.8              |  entry [x1=1, x2=1, value=0.5]
        ###
        ### ... (3 more factor blocks)

        # For every pair of nodes, we have a factor f_rc(.|.) which is computed using the
        # _create_room_connectivity_factor() function. The keys of self._rc_factor should
        # be {(0, 0), (0, 1), (0, 2), ...}
        with open(path, "w") as of:
            # First write the number of factors (i.e. number of edges)
            if not self._use_triplet:
                num_factors = len(topo_map.edges)
            else:
                node_edge_pairs = topo_map.connected_edge_pairs()
                num_factors = self._count_num_triplet_factors(topo_map, node_edge_pairs)
            if likelihoods:
                num_factors += len(topo_map.nodes)
                
            of.write("%d\n\n" % num_factors)
            
            if not self._use_triplet:
                self._create_pairwise_fg(of, topo_map, varlabels, factor=factor)
            else:
                self._create_triplet_fg(of, topo_map, varlabels, node_edge_pairs, factor=factor)
            
            if likelihoods is not None:
                # noisify topo map. Add factor block for each (masked) node's factor for prior class likelihoods.
                for nid in topo_map.nodes:
                    label = varlabels[nid]
                    of.write("1\n")                                     # only one variable
                    of.write("%d\n" % label)                            # node label
                    of.write("%d\n" % CategoryManager.NUM_CATEGORIES)   # number of possible values
                    of.write("%d\n" % CategoryManager.NUM_CATEGORIES)   # number of non-zero entries in the factor table.
                    # Assert validity of likelihoods dict
                    assert len(likelihoods[nid]) == CategoryManager.NUM_CATEGORIES
                    for _i in range(len(likelihoods[nid])):
                        of.write("%d %f\n" % (_i, likelihoods[nid][_i]))
                    of.write("\n")
            print("Factor graph for %s saved to %s." % (seq_id, path))
        return path

    
    def _add_pairwise_factor(self, of, node1, node2, varlabels, factor=None):
        """
        Writes a pairwise factor for the edge between node1 and node2.

        varlabels (dict): dictionary that maps from nid to variable label (used in .fg file)
        """
        label1, label2 = varlabels[node1.id], varlabels[node2.id]
        of.write("2\n")  # number of variables in this faactor
        of.write("%d %d\n" % (label1, label2)) # variable labels
        of.write("%d %d\n" % (CategoryManager.NUM_CATEGORIES, CategoryManager.NUM_CATEGORIES))

        if factor is None:
            factor = self._rc_factor

        to_write = []
        for valpair in sorted(factor):
            a, b = valpair
            table_index = b*CategoryManager.NUM_CATEGORIES + a
            to_write.append((table_index, factor[valpair]))
            if a != b:
                rev_table_index = a*CategoryManager.NUM_CATEGORIES + b
                to_write.append((rev_table_index, factor[valpair]))

        of.write("%d\n" % (len(to_write)))  # number of non-zero entries
        for table_index, val in sorted(to_write, key=lambda x: x[0]):
            of.write("%d %f\n" % (table_index, val))

        of.write("\n")  # ends a factor block

        
    def _create_pairwise_fg(self, of, topo_map, varlabels, factor=None):
        """
        Appends pairwise factors to the file `of`.

        of (file): file opened for the factor graph file
        varlabels (dict): dictionary that maps from nid to variable label (used in .fg file)
        """
        # For each edge, add a factor
        for eid in topo_map.edges:
            node1, node2 = topo_map.edges[eid].nodes
            self._add_pairwise_factor(of, node1, node2, varlabels, factor=factor)


    def _count_num_triplet_factors(self, topo_map, node_edge_pairs):
        covered = set({})
        for nid in node_edge_pairs:
            for edge_pair in node_edge_pairs[nid]:
                # We don't want to duplicate factors
                if not (edge_pair in covered and reversed(edge_pair) in covered):
                    covered.add(edge_pair)
        return len(covered)
    

    def _add_triplet_factor(self, of, node1, nodeC, node2, varlabels, factor=None):
        """
        nodeC (Node): center node
        """
        label1, labelC, label2 = varlabels[node1.id], varlabels[nodeC.id], varlabels[node2.id]
        of.write("3\n")  # number of variables in this faactor
        of.write("%d %d %d\n" % (label1, labelC, label2)) # variable labels
        of.write("%d %d %d\n" % (CategoryManager.NUM_CATEGORIES,
                                 CategoryManager.NUM_CATEGORIES, CategoryManager.NUM_CATEGORIES))

        if factor is None:
            factor = self._tr_factor

        to_write = []
        for triplet in factor:
            a, b, c = triplet
            table_index = c*CategoryManager.NUM_CATEGORIES**2 + b*CategoryManager.NUM_CATEGORIES + a
            to_write.append((table_index, factor[triplet]))

        of.write("%d\n" % (len(to_write)))  # number of non-zero entries
        for table_index, val in sorted(to_write, key=lambda x: x[0]):
            of.write("%d %f\n" % (table_index, val))

        of.write("\n")  # ends a factor block

    def _create_triplet_fg(self, of, topo_map, varlabels, node_edge_pairs, factor=None):
        # For each edge pair, add a factor. All edge pais should be added since topo_map is
        # (assumed to be) a connected graph.
        covered = set({})
        for nid in node_edge_pairs:
            for edge_pair in node_edge_pairs[nid]:
                # We don't want to duplicate factors
                if not (edge_pair in covered and reversed(edge_pair) in covered):
                    nid1, nidC, nid2 = Edge.get_triplet_from_edge_pair(topo_map, edge_pair, nid, catg=False)
                    self._add_triplet_factor(of, topo_map.nodes[nid1], topo_map.nodes[nidC], topo_map.nodes[nid2], varlabels, factor=factor)
                    covered.add(edge_pair)
                

    def _create_tab(self, topo_map, seq_id, varlabels, masked_catg_map, path=None, likelihoods=None):
        """
        Creates an evidence file (.tab) to feed observation to the factor graph in order to
        perform MAP inference. Save the file to path, if provided.

        varlabels (dict): dictionary that maps from node id to variable label (used in .fg file)
        masked_catg_map (dict): dictionary from node id to category number. -1 if masked.

        likelihoods (dict): Optional. If provided, the observations will be based on it. This is a
        map from node id to list of likelihoods, one for each class.
        """
        if path is None:
            path = os.path.join(self._result_dir, "%s_%s.tab" % (self._db_test, seq_id))

        ### Example .tab file format
        ### ------------------------
        ### 0\t1\t2\t3   | variable labels separated by tabs
        ###              | empty line
        ### 1\t0\t1\t1   | observation 1. [x0=1, x1=0, x2=1, x3=1]
        ### 1\t1\t1\t0   | observation 2. [x0=1, x1=1, x2=1, x3=0]
        ### 1\t\t1\t0    | observation 3. [x0=1, x2=1, x3=1], unobserved [x1].
        ### ... (more)

        # We just create one row, i.e. one observation from the masked_catg_map. Simply
        # skip the masked_catg_map.

        # Make two rows
        labels = []
        obsrvs = []
        for nid in masked_catg_map:
            labels.append(str(varlabels[nid]))
            if likelihoods is None:
                if masked_catg_map[nid] == -1:
                    obsrvs.append('')
                else:
                    obsrvs.append(str(masked_catg_map[nid]))
            else:
                #catg_num = likelihoods[nid].index(max(likelihoods[nid]))
                obsrvs.append('')#str(catg_num))
                
        with open(path, 'w') as of:
            # Write the two linesx
            of.write('\t'.join(labels) + "\n\n")
            of.write('\t'.join(obsrvs) + "\n")
            print("Evidence for %s saved to %s." % (seq_id, path))
        return path


def test_one_sequence(seq_id, ft, mask_policy, policy_args,
                      visualize=False, run_correction_task=True,
                      high_likelihood_correct=(0.5, 0.7), low_likelihood_correct=(0.3, 0.5),
                      high_likelihood_incorrect=(0.5, 0.7), low_likelihood_incorrect=(0.3, 0.5),
                      uniform_for_incorrect=False, novelty=False):
    stats = {}
    try:
        result, stats, seq_id = ft.run_task(mask_policy=mask_policy, seq_id=seq_id,
                                            run_correction_task=run_correction_task, visualize=visualize, policy_args=policy_args,
                                            high_likelihood_correct=high_likelihood_correct, low_likelihood_correct=low_likelihood_correct,
                                            high_likelihood_incorrect=high_likelihood_incorrect, low_likelihood_incorrect=low_likelihood_incorrect,
                                            uniform_for_incorrect=uniform_for_incorrect, novelty=novelty)
        print(seq_id + "\n#############\n")
        pprint(stats)
        print("\n")
    except Exception as ex:
        # Do nothing. Probably unnormalizable probability. Ignore erros.
        print("Failed! No result. Exception: %s" % ex)
        if not isinstance(ex, FileNotFoundError):
            raise ex
        return None
    return stats


def run_test_suite(mask_policy, dataset, amount=-1, tiny_dataset=None, policy_args={},
                   visualize=False, run_correction_task=True,
                   high_likelihood_correct=(0.5, 0.7), low_likelihood_correct=(0.3, 0.5),
                   high_likelihood_incorrect=(0.5, 0.7), low_likelihood_incorrect=(0.3, 0.5),
                   uniform_for_incorrect=False, use_triplet=False, novelty=False, infer_type="marginal"):
    """
    Runs the test suite.

    tiny_dataset contains seqs of tiny graphs
    """
    overall_stats = {'_overall_':0, '_total_correct_': 0, '_total_inferred_': 0,
                     '_failed_instances_': 0, '_total_instances_': 0, '_no_prob_': 0}
    dbs = ['Freiburg', 'Stockholm', 'Saarbrucken']
    for db in dbs:
        overall_stats[db] = {'_overall_':0, '_total_correct_': 0, '_total_inferred_': 0,
                             '_failed_instances_': 0, '_total_instances_': 0, '_no_prob_': 0, '_cases_':[]}

        # Test amount sequences
        ft = FactorGraphTest(BP_EXEC_PATH, dataset, list(set(dbs) - {db}),  db, result_dir=BP_RESULTS_ROOT,
                             tiny_dataset=tiny_dataset, use_triplet=use_triplet, infer_type=infer_type)
        test_seqs = ft.get_test_seq_ids(amount=amount)
        for seq_id in test_seqs:
            stats = test_one_sequence(seq_id, ft, mask_policy, policy_args,
                                      run_correction_task=run_correction_task,
                                      high_likelihood_correct=high_likelihood_correct, low_likelihood_correct=low_likelihood_correct,
                                      high_likelihood_incorrect=high_likelihood_incorrect, low_likelihood_incorrect=low_likelihood_incorrect,
                                      uniform_for_incorrect=uniform_for_incorrect, novelty=novelty)
            if novelty:
                overall_stats[db]['_cases_'].append(stats)
                continue
            overall_stats[db]['_total_instances_'] += 1
            if stats is None:
                overall_stats[db]['_failed_instances_'] += 1
                continue
            seq_correct, seq_total_inferred = 0, 0
            for k in stats:
                if not k.startswith('_'):
                    if k not in overall_stats[db]:
                        overall_stats[db][k] = [0,0,0]
                    # k is a class abbreviation
                    overall_stats[db][k][0] += stats[k][0]
                    overall_stats[db][k][1] += stats[k][1]
                    seq_correct += stats[k][0]
                    seq_total_inferred += stats[k][1]
                    overall_stats[db]['_total_correct_'] += stats[k][0]
                    overall_stats[db]['_total_inferred_'] += stats[k][1]
                    if overall_stats[db][k][1] != 0:
                        overall_stats[db][k][2] = overall_stats[db][k][0] / overall_stats[db][k][1]
            overall_stats[db]['_no_prob_'] += stats['_no_prob_']
            if seq_total_inferred != 0:
                overall_stats[db]['_cases_'].append(seq_correct / seq_total_inferred)
        if novelty:
            continue
        if overall_stats[db]['_total_inferred_'] != 0:
            overall_stats[db]['_overall_'] = overall_stats[db]['_total_correct_'] / overall_stats[db]['_total_inferred_']

        overall_stats['_total_correct_'] += overall_stats[db]['_total_correct_']
        overall_stats['_total_inferred_'] += overall_stats[db]['_total_inferred_']
        overall_stats['_failed_instances_'] += overall_stats[db]['_failed_instances_']
        overall_stats['_total_instances_'] += overall_stats[db]['_total_instances_']
        overall_stats['_no_prob_'] += overall_stats[db]['_no_prob_']
         
    if overall_stats['_total_inferred_'] != 0:
        overall_stats['_overall_'] = overall_stats['_total_correct_'] / overall_stats['_total_inferred_']
    return overall_stats
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run factor graph test.')
    parser.add_argument('-t', '--tiny-size', type=int, help="Tiny size. If you want to run experiments on tiny graphs, give me a size, such as 4. If run this test, won't run others (with full map).", default=0)
    parser.add_argument('-n', '--num-runs', type=int, help="number of rounds of tests.. Default: 1.", default=1)
    parser.add_argument('-N', '--amount-seqs', type=int, help="number of sequences to run test on per round. Default: 5.", default=5)
    parser.add_argument('-d', '--doorway-exp', action="store_true", help="Run doorway noisification test, i.e. only occlude or noisify doorways.")
    parser.add_argument('-crct', '--classif-correction', action="store_true", help="Run classification correction experiment.")
    parser.add_argument('-infph', '--inferplaceholder-exp', action="store_true", help="Run infer placeholder experiment.")
    parser.add_argument('-i', '--inference-type', type=str, help="Inference type. Can be 'mpe' or 'marginal'. Default is 'marginal'. Must be lower-case.", default="marginal")
    parser.add_argument('-l', '--likelihoods', action="store_true", help="Use likelihood or not.")
    parser.add_argument('-lo-c', '--low-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-c', '--high-likelihood-correct', type=str, help="For nodes that take groundtruth class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-lo-ic', '--low-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. Low likelihood range. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-hi-ic', '--high-likelihood-incorrect', type=str, help="For nodes that take incorrect class: Tuple-like string. High likelihood ramge. If supplied, override what is in the test_kwargs file.")
    parser.add_argument('-unif', '--uniform-for-incorrect', action="store_true", help="Set likelihoods for nodes labeled incorrect to be uniform")
    parser.add_argument('-v', '--visualize', action="store_true", help="Visualize groundtruth, query, and result")
    parser.add_argument('-seg', '--segment', action="store_true", help="Segment the graph such that each node is a room, instead of a place.")
    parser.add_argument('-tr', '--triplet', action="store_true", help="Use 3-element factors instead of pairwise factors for room connectivity.")
    parser.add_argument('-nov', '--novelty', action="store_true", help="Runs novelty detection task.")
    parser.add_argument("--skip-placeholders", help='Skip placeholders. Placeholders will not be part of the graph.', action='store_true')
    args = parser.parse_args()

    CategoryManager.TYPE = "SIMPLE"
    CategoryManager.init()
    
    print("loading data...")
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load(db_name="Stockholm", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment, skip_placeholders=args.skip_placeholders)
    dataset.load(db_name="Saarbrucken", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment, skip_placeholders=args.skip_placeholders)
    dataset.load(db_name="Freiburg", skip_unknown=CategoryManager.SKIP_UNKNOWN, segment=args.segment, skip_placeholders=args.skip_placeholders)

    if args.tiny_size > 0:
        tiny_dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
        tiny_size = args.tiny_size
        tiny_dataset.load_tiny_graphs(db_name="Stockholm", skip_unknown=CategoryManager.SKIP_UNKNOWN, num_nodes=tiny_size)
        tiny_dataset.load_tiny_graphs(db_name="Saarbrucken", skip_unknown=CategoryManager.SKIP_UNKNOWN, num_nodes=tiny_size)
        tiny_dataset.load_tiny_graphs(db_name="Freiburg", skip_unknown=CategoryManager.SKIP_UNKNOWN, num_nodes=tiny_size)

    # Parameters.
    test_kwargs = {
        'amount': args.amount_seqs,
        'visualize': args.visualize,
        'uniform_for_incorrect': args.uniform_for_incorrect,
        'use_triplet': args.triplet,
        'infer_type': args.inference_type
    }
    if args.likelihoods or args.inferplaceholder_exp:
        test_kwargs.update({
            'high_likelihood_correct': eval(args.high_likelihood_correct),
            'low_likelihood_correct': eval(args.low_likelihood_correct),
            'high_likelihood_incorrect': eval(args.high_likelihood_incorrect),
            'low_likelihood_incorrect': eval(args.low_likelihood_incorrect),    
        })
    
    outcome = {}

    for i in range(args.num_runs):
        print_in_box(["Run #%d" % (i+1)])
        outcome[i] = {}
        if args.classif_correction:
            if args.tiny_size > 0:
                print_banner("Tiny NOISE", length=40)
                outcome[i]['t_tiny'] = run_test_suite(random_fixed_policy, dataset, tiny_dataset=tiny_dataset,
                                                      policy_args={'rate_occluded': 0.2}, run_correction_task=args.likelihoods,
                                                      **test_kwargs)
            else:
                print_banner("Random NOISE", length=40)
                outcome[i]['t_rand'] = run_test_suite(random_policy, dataset,
                                                      run_correction_task=args.likelihoods, policy_args={'rand_rate': 0.2},
                                                      **test_kwargs)

        if args.doorway_exp:
            print_banner("Doorway NOISE", length=40)
            outcome[i]['1_door'] = run_test_suite(doorway_policy, dataset,
                                                  run_correction_task=args.likelihoods,
                                                  **test_kwargs)
            
        if args.inferplaceholder_exp:
            print_banner("Placeholder NOISE", length=40)
            outcome[i]['2_infr'] = run_test_suite(None, dataset,
                                                  run_correction_task=False, policy_args={'rate_occlude': 0.2},
                                                  **test_kwargs)


        if args.novelty:
            print_banner("Novelty", length=40)
            outcome[i]['3_novl'] = run_test_suite(None, dataset,
                                                  run_correction_task=False, novelty=True,
                                                  **test_kwargs)

        timestamp = time.strftime("%Y%m%d-%H%M")
        with open(os.path.join("r/report_%s.log" % timestamp), "w") as f:
            pprint(outcome[i], stream=f)
        print("Done! [%d]" % (i+1))

    # Compute overall average and standard deviation
    print("Gathering statistics...")
    overall = {} 
    for i in outcome:
        print("--- Run #%d ---" % (i+1))
        pprint(outcome[i])
        for test_name in outcome[i]:
            if test_name not in overall:
                # two lists: 1st=Level 0: overall across 3 databases, 2nd=Level 1: overall within one database
                overall[test_name] = {'_cases_': [], '_total_correct_': 0, '_total_inferred_': 0}  
            for key in outcome[i][test_name]:
                if not key.startswith("_"):  # key is a db
                    if key not in overall[test_name]:
                        overall[test_name][key] = {'_cases_':[], '_total_correct_': 0, '_total_inferred_': 0}
                    overall[test_name][key]['_cases_'].extend(outcome[i][test_name][key]['_cases_'])
                    overall[test_name][key]['_total_correct_'] += outcome[i][test_name][key]['_total_correct_']
                    overall[test_name][key]['_total_inferred_'] += outcome[i][test_name][key]['_total_inferred_']
            overall[test_name]['_cases_'].append(outcome[i][test_name]['_overall_'])
            overall[test_name]['_total_correct_'] += outcome[i][test_name]['_total_correct_']
            overall[test_name]['_total_inferred_'] += outcome[i][test_name]['_total_inferred_']

    # Compute statistics
    for test_name in overall:
        for db_name in overall[test_name]:
            if not db_name.startswith("_"):
                overall[test_name][db_name]['_mean_'] = np.mean(overall[test_name][db_name]['_cases_'])
                overall[test_name][db_name]['_std_'] = np.std(overall[test_name][db_name]['_cases_'])
        overall[test_name]['_mean_'] = np.mean(overall[test_name]['_cases_'])
        overall[test_name]['_std_'] = np.std(overall[test_name]['_cases_'])
    
    timestamp = time.strftime("%Y%m%d-%H%M")
    with open(os.path.join("r/overall_report_%s.log" % timestamp), "w") as f:
        pprint(overall, stream=f)
    pprint(overall)

    # Print noisification level (if involving likelihoods)
    if args.likelihoods:
        print("--- Noisifcation Level ---")
        pprint(get_noisification_level(eval(args.high_likelihood_correct), eval(args.low_likelihood_correct),
                                       eval(args.high_likelihood_incorrect), eval(args.low_likelihood_incorrect),
                                       uniform_for_incorrect=args.uniform_for_incorrect))
    print("All done.")
