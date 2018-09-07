import numpy as np

import os, sys
from deepsm.graphspn.tbm.topo_map import PlaceNode, TopologicalMap
from deepsm.graphspn.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate, SingletonTemplate
import deepsm.util as util
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


class PartitionSampler:

    """
    A partition sampler samples partitions for a particular topological
    map according to a distribution described by an energy function.

    - The energy function should be exp(-alpha * factor1) * exp(-beta * factor2) etc.
    - Factors to consider:
      - similarity: % of nodes with identical templates
      - complexity: graded by number of variables, should be equal to 1 for
                    a graph completely covered by most complex templates, should
                    assign highest weight to most complex template, then to the
                    second most complex etc.
      - template variable values: assigns energy to particular values of template
                    variables, e.g. view value (promote straigh template), degree
                    of middle node, or ability of the middle node of template to
                    split graph into independent components.
    """

    def __init__(self, topo_map, **kwargs):
        self._topo_map = topo_map
        self._similarity_coeff        = kwargs.get('similarity_coeff', 0.02)
        self._complexity_coeff        = kwargs.get('complexity_coeff', 3)
        self._straight_template_coeff = kwargs.get('straight_template_coeff', 10)
        self._dom_coeff               = kwargs.get('dom_coeff', 3)
        self._separable_coeff         = kwargs.get('separable_coeff', 10)


    def set_params(self, **kwargs):
        self._similarity_coeff        = kwargs.get('similarity_coeff', self._similarity_coeff)
        self._complexity_coeff        = kwargs.get('complexity_coeff', self._complexity_coeff)
        self._straight_template_coeff = kwargs.get('straight_template_coeff', self._straight_template_coeff)
        self._dom_coeff               = kwargs.get('dom_coeff', self._dom_coeff)
        self._separable_coeff         = kwargs.get('separable_coeff', self._separable_coeff)


    def sample_partition_set(self, num_rounds, num_partitions, accept_threshold=float('inf')):
        """
        Pick the best set of partitions (with size `num_partitions`) among
        `num_rounds` of rounds.
        """
        attributes = []
        partition_sets = []

        for i in range(num_rounds):
            sys.stdout.write("round %d/%d\r" % (i+1, num_rounds))
            sys.stdout.flush()
            partition_sets[i], attributes[i] = sampler.sample_partitions(num_partitions, accept_threshold=accept_threshold)

        indices = sorted(range(len(attributes)), key=lambda i:np.median(attributes[i]['energies']))
        top_partition_set = partition_sets[indices[0]]
        energies = attributes[indices[0]]['energies']
        factors = attributes[indices[0]]['factors']

        return top_partition_set, {'energies': energies,
                                   'factors': factors}


    def sample_partitions(self, amount,
                          current_partitions=[],
                          accept_threshold=float('inf'),
                          debug=False):
        """Sample `amount` number of  partitions. Take into account `current_partitions`
        (if not None) when computing similarity. """
        def sample_one(current_partitions=[], debug=False):
            p = self.random_partition()
            e, f = self.energy(p, partitions=current_partitions, debug=debug)
            while e > accept_threshold:
                p = self.random_partition()
                e, f = self.energy(p, partitions=current_partitions, debug=debug)
            return p, e, f

        current_partitions = current_partitions.copy()
        partitions = []
        attributes = {'energies':[],'factors':[]}

        while len(partitions) < amount:
            p, e, f = sample_one(current_partitions=current_partitions, debug=debug)
            partitions.append(p)
            attributes['energies'].append(e)
            attributes['factors'].append(f)
            current_partitions.append(p)
        return partitions, attributes


    def energy(self, partition, partitions=[], debug=False):

        factors = {
            'complexity': self._complexity(partition),
            'straight': self._straight_template(partition),
            'dom': self._degree_of_middle_node(partition),
            'separable': self._separable_by_middle_node(partition),
        }
        score = np.exp(-self._complexity_coeff*factors['complexity']) \
                * np.exp(-self._straight_coeff*factors['straight']) \
                * np.exp(-self._dom_coeff*factors['dom']) \
                * np.exp(-self._separable_coeff*factors['separable'])

        if len(partitions) > 0:
            factors['similarity'] = self._similarity(partition, partitions)
            score *= np.exp(-self._similarity_coeff * factors['similarity'])

        if debug:
            print("   complexity: %.4f" % np.log(factors['complexity']))
            print("     straight: %.4f" % np.log(factors['straight']))
            print("   similarity: %.4f" % np.log(factors['similarity']))
            print("deg_of_middle: %.4f" % np.log(factors['deg_of_middle']))
            print("    separable: %.4f" % np.log(factors['separable']))
            print("      [score]  %.4f" % score)
            print("---------------------")

        return score, factors

    

class NodeTemplatePartitionSampler(PartitionSampler):

    """
    A partition is a dictionary that maps from template class to a TopologicalMap
    object that represents the subgraph resulted from partitioning the graph using
    that template.
    """

    def __init__(self, topo_map,
                 templates=[ThreeNodeTemplate, PairTemplate, SingletonTemplate],
                 **kwargs):
        """
        templates (list) list of templates ordered by descending complexity

        kwargs:
            similarity_coeff
            complexity_coeff
            straight_template_coeff
            straight_template_coeff
            dom_coeff (degree_of_middle_node_coefficient)
            separable_coeff (separable by middle node coefficient)
        """
        super().__init__(topo_map, **kwargs)
        self._templates = templates


    def random_partition(self):
        partition = {}
        tmp_graph = self._topo_map
        for template in self._templates:
            supergraph, unused_graph = tmp_graph.partition(template, get_unused=True)
            partition[template] = supergraph
            tmp_graph = unused_graph
        return partition
            

    def visualize_partition(self, partition, groundtruth_file):
        """presult is the returned object from partition()"""

        rcParams['figure.figsize'] = 22, 14

        ctype = 2
        for template in self._templates:
            sgraph = partition[template]
            node_ids = []
            for snid in sgraph.nodes:
                node_ids.append(sgraph.nodes[snid].to_place_id_list())
            self._topo_map.visualize_partition(plt.gca(), node_ids, groundtruth_file,  ctype=ctype)
            ctype += 1

    def _complexity(self, partition):
        """
        should be equal to 1 for a graph completely covered by most complex templates
        """
        score = 0
        template_weight = 0.5
        for i, template in enumerate(self._templates):
            score += template_weight * len(partition[template].nodes) / len(self._topo_map.nodes)
            template_weight /= 2
        return score

    def _similarity(self, partition, partitions):
        """
        % of nodes with identical templates
        """
        count = 0
        
        for template in self._templates:
            supergraph = partition[template]
            for i in supergraph.nodes:
                num_contained = 0
                for q in partitions:
                    contained = False
                    for j in q[template].nodes:
                        if supergraph.nodes[i].to_place_id_list() == q[template].nodes[j].to_place_id_list() \
                           or list(reversed(supergraph.nodes[i].to_place_id_list())) == q[template].nodes[j].to_place_id_list():
                            contained = True  # in identical templates
                            break
                    if contained is True:
                        num_contained += 1
                if num_contained > (len(partitions) // 2):
                    count += template.size()
                            
        return count / len(self._topo_map.nodes)

    def _straight_template(self, partition, dist_func=util.abs_view_distance):
        """
        Percentage of nodes in straight templates
        """
        if self._templates[0] != ThreeNodeTemplate:
            raise ValueError("straight_template factor only evaluated when "\
                             "ThreeNodeTemplate as main template")
        count = 0
        supergraph = partition[self._templates[0]]
        for i in supergraph.nodes:
            nodes = supergraph.nodes[i].nodes  # nodes in template

            view_1 = util.compute_view_number(nodes[1], nodes[0])
            view_2 = util.compute_view_number(nodes[1], nodes[2])
            dist = dist_func(view_1, view_2)
            if dist == 4:  # straight line
                count += self._templates[0].size()
        return count / len(self._topo_map.nodes)
                        
    def _degree_of_middle_node(self, partition, divisions=8):
        """
        Computes average degree of middle node, normalized by divisions (total number
        of possible degrees).
        """
        count = 0
        supergraph = partition[self._templates[0]]
        for i in supergraph.nodes:
            nodes = supergraph.nodes[i].nodes
            middle_node = nodes[len(nodes)//2]
            count += len(self._topo_map.neighbors(middle_node.id))

        return count / len(supergraph.nodes) / 8


    def _separable_by_middle_node(self, partition):
        """
        Returns percentage of nodes which are both center at some templates,
        and can also separate the graph into two components, if removed.
        """
        count = 0
        supergraph = partition[self._templates[0]]
        for i in supergraph.nodes:
            nodes = supergraph.nodes[i].nodes
            middle_node = nodes[len(nodes)//2]

            # Ignores if the degree of middle node is only 2. Otherwise, the middle node could be
            # a room node in a thin change of room nodes. We hope middle node to be doorway node.
            if len(self._topo_map.neighbors(middle_node.id)) <= 2:
                continue
            
            single_node_graph = TopologicalMap({middle_node.id:middle_node}, {middle_node.id:set()})
            remainder = self._topo_map.subtract(single_node_graph)

            if len(remainder.connected_components()) > 1:
                # separates, and non-trivial.
                count += 1

        return count / len(self._topo_map.nodes)
        
