import numpy as np

from deepsm.graphspn.tbm.topo_map import PlaceNode, TopologicalMap
from deepsm.graphspn.tbm.template import SingleEdgeTemplate, PairEdgeTemplate, ThreeNodeTemplate, PairTemplate
from deepsm.util as util
import numpy as np


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

    def sample_partitions(self, amount, current_partitions=None):
        pass

    def energy(self, partition, partitions=None):
        pass

    

class NodeTemplatePartitionSampler:

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

        self._similarity_coeff        = kwargs.get('similarity_coeff', 0.1)
        self._complexity_coeff        = kwargs.get('complexity_coeff', 0.1)
        self._straight_template_coeff = kwargs.get('straight_template_coeff', 0.1)
        self._dom_coeff               = kwargs.get('dom_coeff', 0.1)
        self._separable_coeff         = kwargs.get('separable_coeff', 0.1)
        

    def sample_partitions(self, amount,
                          current_partitions=[],
                          partition_threshold=0.5):
        """Sample `amount` number of  partitions. Take into account `current_partitions`
        (if not None) when computing similarity. """
        partitions = []

        # Sample one partition
        if len(current_partitions) == 0:
            p = self.random_partition()
            while energy(p) < partition_threshold:
                p = self.random_partition()
            partitions.add(p)

        while len(partitions) < amount:
            p = self.random_partition()
            current_partitions += partitions
            while energy(p, current_partitions) < partition_threshold:
                p = self.random_partition()
            partitions.add(p)
        return partitions
        

    def random_partition(self):
        partition = []
        tmp_graph = self._topo_map
        for template in self._templates:
            supergraph, unused_graph = tmp_graph.partition(template, get_unused=True)
            partition[template] = supergraph
            tmp_graph = unused_graph
        return partition
            

    def energy(self, partition, partitions=None):

        score = np.exp(self._complexity_coeff * self._complexity(partition)) \
                * np.exp(self._straight_template_coeff * self._straight_template(partition)) \
                * np.exp(self._dom_coeff * self._degree_of_middle_node(partition)) \
                * np.exp(self._separable_coeff * self._separable_by_middle_node(partition))
        
        if partitions is not None:
            score *= np.exp(self._similarity_coeff * self._similarity(partition, partitions))

        return score
        

    def _complexity(self, partition):
        """
        should be equal to 1 for a graph completely covered by most complex templates
        """
        return len(partition[self._templates[0]].nodes) / len(self._topo_map.nodes)

    def _similarity(self, partition, partitions):
        """
        % of nodes with identical templates
        """
        count = 0
        
        for template in self._templates:
            supergraph = partition[template]
            for i in supergraph.nodes:
                for q in partitions:
                    for j in q[template].nodes:
                        if supergraph.nodes[i].to_place_id_list() == q[template].nodes[j].to_place_id_list() \
                           or list(reversed(supergraph.nodes[i].to_place_id_list())) == q[template].nodes[j].to_place_id_list():
                            # in identical templates
                            count += template.size
        return count / len(self._topo_map.nodes)

    def _straight_template(self, partition, dist_func=util.abs_view_distance):
        """
        Percentage of nodes in straight templates
        """
        if self._templates[0] != ThreeNodeTemplate:
            raise ValueError("straight_template factor only evaluated when "\
                             "ThreeNodeTemplate as main template")
        count = 0
        supergraph = partition[self._templates[0]]:
        for i in supergraph.nodes:
            nodes = supergraph.nodes[i].nodes  # nodes in template

            view_1 = util.compute_view_number(nodes[1], nodes[0])
            view_2 = util.compute_view_number(nodes[1], nodes[2])
            dist = dist_func(view_1, view_2)
            if dist == 4:  # straight line
                count += self._templates[0].size
        return count / len(self._topo_map.nodes)
                        
    def _degree_of_middle_node(self, partition, divisions=8):
        """
        Computes average degree of middle node, normalized by divisions (total number
        of possible degrees).
        """
        count = 0
        supergraph = partition[self._templates[0]]:
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
        supergraph = partition[self._templates[0]]:
        for i in supergraph.nodes:
            nodes = supergraph.nodes[i].nodes
            middle_node = nodes[len(nodes)//2]

            # Ignores if the degree of middle node is only 2. Otherwise, the middle node could be
            # a room node in a thin change of room nodes. We hope middle node to be doorway node.
            if len(self._topo_map.neighbors(middle_node.id)) <= 2:
                continue
            
            single_node_graph = TopologicalMap({middle_node.id:middle_node}, {middle_node.id:set()})
            remainder = self._topo_map.subtract(single_node_graph)

            if len(remainder.connected_components) > 1:
                # separates, and non-trivial.
                count += 1

        return count / len(self._topo_map.nodes)
        
