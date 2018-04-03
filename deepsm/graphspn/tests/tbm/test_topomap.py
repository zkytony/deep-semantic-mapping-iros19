#!/usr/bin/env python
#
# Tests both TopoMapDataset and TopologicalMap class.
#
# author: Kaiyu Zheng

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
#matplotlib.use('Agg')

from spn_topo.tbm.dataset import TopoMapDataset
from spn_topo.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate
#from spn_topo.tests.tbm.runner import TbmExperiment
from spn_topo.util import CategoryManager, ColdDatabaseManager
import csv
import matplotlib.pyplot as plt
import os, sys
from pprint import pprint
import random
import numpy as np

from spn_topo.tests.constants import COLD_ROOT, RESULTS_DIR, TOPO_MAP_DB_ROOT


def doorway_policy(topo_map, node):
    return node.label == "DW"

def random_policy(topo_map, node, rand_rate=0.2):
    return random.uniform(0, 1.0) <= rand_rate


def TEST_refine_partition(dataset, coldmgr, extra_partitions_multiplyer=3):
    rcParams['figure.figsize'] = 22, 14
    coverages = []
    size = 10
    differences = []  # differences between the top `size` and bottom `size` partition attempts.
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=1)
    num_partitions = 5
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]

        print("Partitioning the graph...")
        partitioned_results = {}
        main_template = ThreeNodeTemplate
        for i in range(extra_partitions_multiplyer*num_partitions):
            """Note: here, we only partition with the main template. The results (i.e. supergraph, unused graph) are stored                                                                       
            and will be used later. """
            supergraph, unused_graph = topo_map.partition(main_template, get_unused=True)
            coverage = len(supergraph.nodes)*main_template.size() / len(topo_map.nodes)
            partitioned_results[(i,coverage)] = (supergraph, unused_graph)
        used_partitions = []
        used_coverages = set({})
        _i = 0
        for i,coverage in sorted(partitioned_results, reverse=True, key=lambda x:x[1]):
            used_partitions.append(partitioned_results[(i,coverage)])
            sys.stdout.write(str(coverage) + "  ")
            sys.stdout.flush()
            _i += 1
            if len(used_partitions) >= num_partitions:
                break
        sys.stdout.write("\n")
        """Delete unused partitions"""
        for coverage in list(partitioned_results.keys()):
            if coverage not in used_coverages:
                del partitioned_results[coverage]

        _i = 0
        for supergraph, unused_graph in used_partitions:
            node_ids = []
            for snid in supergraph.nodes:
                node_ids.append(supergraph.nodes[snid].to_place_id_list())
            topo_map.visualize_partition(plt.gca(), node_ids, coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'))
            fname = 'gtiny-%s_%d.png' % (seq_id, _i+1)
            plt.savefig(fname)
            plt.clf()
            sys.stdout.write("%s." % fname)
            sys.stdout.flush()
            _i += 1
        sys.stdout.write("\n")


def TEST_topo_map_copy(dataset):
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=1)
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        copy_map = topo_map.copy()

        # Assert all nodes have the same information
        for nid in topo_map.nodes:
            porig = topo_map.nodes[nid]
            pcpy = copy_map.nodes[nid]
            assert porig.label == pcpy.label
            assert porig.anchor_pose == pcpy.anchor_pose
            assert porig.pose == pcpy.pose
            assert porig.placeholder == pcpy.placeholder
                
        # Assert all nodes have the same connectivity
        for nid in topo_map.nodes:
            neiborig = topo_map.neighbors(nid)
            neibcpy = copy_map.neighbors(nid)
            assert neiborig == neibcpy
            for nnid in neiborig:
                assert topo_map.edge_between(nid, nnid) == copy_map.edge_between(nid, nnid)


        # Assert modifying copy_map information does not change topo_map
        for i in range(1000):
            nid = random.sample(topo_map.nodes.keys(), 1)[0]
            copy_map.nodes[nid].label = 100
            assert topo_map.nodes[nid].label != copy_map.nodes[nid].label


def TEST_partition_by_edge_relations(dataset):
    rcParams['figure.figsize'] = 22, 14
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=1)
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]

        ert_map1 = topo_map.partition_by_edge_relations()
        ert_map2 = topo_map.partition_by_edge_relations()

        total1 = 0
        total2 = 0
        for key in ert_map1:
            print("[1] %s: %d" % (key, len(ert_map1[key])))
            for template in ert_map1[key]:
                total1 += template.num_vars
        print(total1)
        for key in ert_map2:
            print("[2] %s: %d" % (key, len(ert_map2[key])))
            for template in ert_map2[key]:
                total2 += template.num_vars
        print(total2)
        assert total1 == total2

        
def TEST_load_edgetemplate_samples(dataset):
    samples, stats = dataset.create_edge_template_dataset(PairEdgeTemplate, return_stats=True)
    for db in samples:
        print("Loaded %d from %s." % (len(samples[db]), db))
        print("____It looks like: %s" % str(random.sample(samples[db], 1)[0]))

    print(stats)



def TEST_segmentation(dataset, coldmgr):
    rcParams['figure.figsize'] = 22, 14
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=1)
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        seg_graph = topo_map.segment()

        topo_map.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'))
        plt.savefig('seg_full.png')
        plt.clf()
        print("Saved seg_full.png")
        
        seg_graph.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), dotsize=18)
        plt.savefig('seg_no_doorway.png')
        plt.clf()
        print("Saved seg_no_doorway.png")
        
        seg_graph = topo_map.segment(remove_doorway=False)
        seg_graph.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), dotsize=18)
        plt.savefig('seg_doorway.png')
        plt.clf()
        print("Saved seg_doorway.png")
    

if __name__ == "__main__":

    coldmgr = ColdDatabaseManager("Stockholm", COLD_ROOT)

    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
    dataset.load("Stockholm", skip_unknown=True)
    #TEST_refine_partition(dataset, coldmgr)
    #TEST_topo_map_copy(dataset)
    TEST_segmentation(dataset, coldmgr)
