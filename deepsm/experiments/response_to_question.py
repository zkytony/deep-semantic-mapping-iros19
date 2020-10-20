# Response to the question in this email about number of nodes in an SPN
#
# Dear Kaiyu Zheng and Andrzej Pronobis,
#
# I am a PhD student in KU Leuven, Belgium designing an ASIC for SPNs. In
# one of my papers, I am using TopoNets as a real-life application
# example. For this purpose, I want to roughly estimate the computational
# cost of a TopoNet for any example building.
#
# In particular, I want to check the following 2 things with you:
#     1) Do you have such an estimate of the cost of a TopoNet for any
# example building (e.g. in terms of number of sum and product nodes)
#     2) Are there any trained TopoNets that you can share with us?
#
# Thank you very much for your time!
#
# Best regards,
# Nimish Shah
# PhD student, KU Leuven

import libspn as spn
import tensorflow as tf
from libspn.graph.algorithms import traverse_graph

total_sums = 0
total_products = 0

def fun(node):
    global total_sums, total_products
    if node.is_op:
        if isinstance(node, spn.Sums):
            total_sums += node.num_sums
        elif isinstance(node, spn.Sum):
            total_sums += 1
        elif isinstance(node, spn.ParSums):
            total_sums += node.num_sums
        elif isinstance(node, spn.Product):
            total_products += 1
        elif isinstance(node, spn.PermProducts):
            total_products += node.num_prods

def count_nodes(root):
    global total_sums, total_products
    total_sums = 0
    total_products = 0
    traverse_graph(root, fun, skip_params=True)
    print("Total sums: %d" % total_sums)
    print("Total products: %d" % total_products)

def count_polar_scan_spn():
    print("Count polar scan spn:")
    ivs = spn.IVs(num_vars=1175, num_vals=3)
    dg_layered = spn.DenseSPNGeneratorLayerNodes(num_decomps=1, num_subsets=3,
                                                 num_mixtures=5, input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW)

    root = dg_layered.generate(ivs)
    count_nodes(root)

def count_topo_graph_spn(num_templates=30, num_partitions=40):
    global total_sums, total_products
    print("Count topo graph spn:")
    ivs = spn.IVs(num_vars=3, num_vals=10)
    dg_layered = spn.DenseSPNGeneratorLayerNodes(num_decomps=1, num_subsets=3,
                                                 num_mixtures=5, input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW)
    root = dg_layered.generate(ivs)
    count_nodes(root)
    total_sums *= (num_templates + num_partitions)
    total_products *= (num_templates + num_partitions)
    print("Total sums: %d" % total_sums)
    print("Total products: %d" % total_products)

if __name__ == "__main__":
    count_polar_scan_spn()
    count_topo_graph_spn()
