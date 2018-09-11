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

        # The default values are set according to random search.
        self._similarity_coeff        = kwargs.get('similarity_coeff', -3.2892620362926266)
        self._complexity_coeff        = kwargs.get('complexity_coeff', 6.8987258916531236)
        self._straight_template_coeff = kwargs.get('straight_template_coeff', 7.997020601228423)
        self._dom_coeff               = kwargs.get('dom_coeff', 4.8590878043033126)
        self._separable_coeff         = kwargs.get('separable_coeff', 2.1314365714437775)


    def set_params(self, **kwargs):
        self._similarity_coeff        = kwargs.get('similarity_coeff', self._similarity_coeff)
        self._complexity_coeff        = kwargs.get('complexity_coeff', self._complexity_coeff)
        self._straight_template_coeff = kwargs.get('straight_template_coeff', self._straight_template_coeff)
        self._dom_coeff               = kwargs.get('dom_coeff', self._dom_coeff)
        self._separable_coeff         = kwargs.get('separable_coeff', self._separable_coeff)


    def sample_partition_sets(self, num_rounds, num_partitions, accept_threshold=float('inf'), pick_best=False):
        """
        Sample `num_rounds` sets of partitions, each with size `num_partitions`

        Returns partition_sets (list of partitions)
            and attributes (list of dictionaries with 'energies' and 'factors' as keys)

            If pick_best is True, the third element to be returned is the index for the best partition_set.
        """
        attributes = []
        partition_sets = []

        for i in range(num_rounds):
            p, a = self.sample_partitions(num_partitions, accept_threshold=accept_threshold)
            partition_sets.append(p)
            attributes.append(a)

            # print progress
            sys.stdout.write("round %d/%d\r" % (i+1, num_rounds))
            sys.stdout.flush()
        sys.stdout.write("\n")

        if pick_best:
            indices = sorted(range(len(attributes)), key=lambda i:np.median(attributes[i]['energies']))
            return partition_sets, attributes, indices[0]
        else:
            return partition_sets, attributes


    def sample_partitions(self, amount,
                          current_partitions=[],
                          accept_threshold=float('inf'),
                          debug=False):
        """Sample `amount` number of  partitions. Take into account `current_partitions`
        (if not None) when computing similarity. If accept_threshold is infinity,
        this function essentially randomly samples `amount` number of partitions"""
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
                * np.exp(-self._straight_template_coeff*factors['straight']) \
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

    def visualize_partition(self, partition, groundtruth_file):
        pass

    def _complexity(self, partition):
        pass

    def _similarity(self, partition, partitions):
        pass

    def _straight_template(self, partition, dist_func=util.abs_view_distance):
        pass

    def _degree_of_middle_node(self, partition, divisions=8):
        pass

    def _separable_by_middle_node(self, partition):
        pass

    

class NodeTemplatePartitionSampler(PartitionSampler):

    """
    For NodeTemplatePartitionSampler:
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
            

    def visualize_partition(self, partition, groundtruth_file, ax=None):
        """presult is the returned object from partition()"""

        if ax is None:
            ax = plt.gca()

        ctype = 2
        for template in self._templates:
            sgraph = partition[template]
            node_ids = []
            for snid in sgraph.nodes:
                node_ids.append(sgraph.nodes[snid].to_place_id_list())

            rcParams['figure.figsize'] = 22, 14
            self._topo_map.visualize_partition(ax, node_ids, groundtruth_file,  ctype=ctype)
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
        similarity equals to

           number of nodes in `partition` that is in the same main template as any partition in `partitions`
        =  --------------------------------------------------------------------------------------------
                   number of nodes in main template of all `partitions`
        """
        if len(partitions) == 0:
            return 0  # nothing to compare with
        count = 0
        for template in self._templates:
            supergraph = partition[template]
            for i in supergraph.nodes:
                for q in partitions:
                    for j in q[template].nodes:
                        if supergraph.nodes[i].to_place_id_list() == q[template].nodes[j].to_place_id_list() \
                           or list(reversed(supergraph.nodes[i].to_place_id_list())) == q[template].nodes[j].to_place_id_list():
                            count += template.size()
                            break
                            
        return count / (len(self._topo_map.nodes)*max(1, len(partitions)))

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
        

class EdgeRelationPartitionSampler(PartitionSampler):

    """
    For EdgeRelationPartitionSampler
      A partition is an ert_map (see topo_map:partition_by_edge_relatoins).
    This is a dictionary that maps from a tuple (num_nodes, num_edge_pairs) indicating
    the type of the template, to a list of such EdgeRelationTemplate instances.
    """

    def __init__(self, topo_map, **kwargs):
        super().__init__(topo_map, **kwargs)


    def random_partition(self):
        return self._topo_map.partition_by_edge_relations()

    def visualize_partition(self, partition, groundtruth_file, ax=None):
        if ax is None:
            ax = plt.gca()
        rcParams['figure.figsize'] = 22, 14
        self._topo_map.visualize_edge_relation_partition(ax, partition, groundtruth_file)
        
    def _complexity(self, partition):
        """
        should be equal to 1 for a graph completely covered by most complex templates
        """
        score = 0
        template_weight = 0.5
        for i, key in enumerate(partition):
            if key[0] != 0:
                score += template_weight * key[0] * len(partition[key])
                template_weight /= 2
        return score

    def _similarity(self, partition, partitions):
        """
        similarity equals to

           number of variables `partition` that is in the same main template as any partition in `partitions`
        =  --------------------------------------------------------------------------------------------
                   total number of variables in main template of all `partiions`

        As implied by the above, we only look at main template (3 nodes, 1 edge pair)
        """
        if len(partitions) == 0:
            return 0  # nothing to compare with
        count = 0
        total = 0
        for key in partition:
            if key == (3,1):
                for i in partition[key]:
                    for q in partitions:
                        for j in q[key]:
                            nids1 = [n.id for n in i.nodes]
                            nids2 = [n.id for n in j.nodes]
                            if nids1 == nids2 or reversed(nids1) == nids2:
                                count += 4
        for q in partitions:
            total += len(q[(3,1)]) * 4  # num_vars is 4
        return count / total
                            

    def _straight_template(self, partition, dist_func=util.abs_view_distance):
        """
        Percentage of nodes in straight templates
        """
        count = 0
        for i in partition[(3,1)]:
            _, vdist = i.to_sample()
            if vdist == 4:
                count += 4  # 4 variables per template
        return count / (len(partition[(3,1)]) * 4)


    def _degree_of_middle_node(self, partition, divisions=8):
        """
        Computes average degree of middle node, normalized by divisions (total number
        of possible degrees).
        """
        count = 0
        for i in partition[(3,1)]:
            middle_node = i.nodes[len(i.nodes)//2]
            count += len(self._topo_map.neighbors(middle_node.id))
        return count / len(partition[(3,1)]) / 8


    def _separable_by_middle_node(self, partition):
        count = 0
        for i in partition[(3,1)]:
            middle_node = i.nodes[len(i.nodes)//2]

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


### Useful for parameter selection ###
def factor_correlations(factor_names, num_rounds=20, num_partitions=5,
                        params=None, plot=False, dbs=['Stockholm', 'Freiburg', 'Saarbrucken'], num_seqs=10):
    """Sample a bunch of partitions. Then, for each parameter, plot each partition's value for that parameter
    versus the partition's energy. See if they correlate in the way we want.
    
    Returns a dictionary of the format:
    
        factor name -> {
            db_name -> {
                factor_name -> correlation_coefficient
            }
            _average_ -> average correlation coefficient
        }
    
    that maps from factor name to correlation coefficient of that factor to energy.
    """
    def corr_plot(ax, x, y, xlabel):
        ax.scatter(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("energy")
    
    result = {}
    dataset = TopoMapDataset(TOPO_MAP_DB_ROOT)
                             
    for db in dbs:
        print("On DB %s" % db)
        dataset.load(db, skip_unknown=True, skip_placeholders=True, single_component=True)
        topo_maps = dataset.get_topo_maps(db_name=db, amount=num_seqs)
        
        x = {}
        y = {}
    
        for seq_id in topo_maps:
            topo_map = topo_maps[seq_id]
            sampler = NodeTemplatePartitionSampler(topo_map=topo_map)
            if params is not None:
                sampler.set_params(**params)

            partition_sets, attributes = sampler.sample_partition_sets(num_rounds, num_partitions)

            for i in range(len(partition_sets)):
                for f in factor_names:
                    if f not in x:
                        x[f] = []
                        y[f] = []
                    for j, p in enumerate(partition_sets[i]):
                        attr = attributes[i]
                        if f in attr['factors'][j]:
                            x[f].append(attr['factors'][j][f])
                            y[f].append(attr['energies'][j])
                            
        if plot:
            fig, axes = plt.subplots(len(factor_names)//3+1, 3, sharey=True)
            fig.suptitle('%s' % db)
            for i, p in enumerate(factor_names):
                # Plot
                corr_plot(axes[i//3,i%3], x[p], y[p], p)
                
                
        # Compute correlation coefficent
        for f in factor_names:
            if f not in result:
                result[f] = {}
            result[f][db] = np.corrcoef(x[f], y[f])
            
    for f in factor_names:
        result[f]['_average_'] = np.mean([result[f][db] for db in dbs])
    return result


def random_search(params, ranges, desired, factor_names,
                  rounds=50, **fc_params):
    """Goal is to find a set of parameters that best separate the
    different partitions, and maximize desired correlation with
    the energy.
    
    The ith element in `desired` should be 1 if we hope the ith
    parameter to have a positive correlation with the energy. -1
    if otherwise.
    
    The ith element in prarams should be the coefficient for the
    ith element in factor_names.
    """
    best_setting = {}
    best_corrs = {}
    lowest_avg_gap = float('inf')
    for r in range(rounds):
        sys.stdout.write('*** round %d/%d ***\n' % (r+1, rounds)); sys.stdout.flush()
        setting = {}
        gaps = []
        for i in range(len(params)):
            setting[params[i]] = np.random.uniform(low=ranges[i][0], high=ranges[i][1])
        print(setting)
        
        corrs = factor_correlations(factor_names, **fc_params)
        for i in range(len(params)):
            f = factor_names[i]
            gaps.append(abs(desired[i] - corrs[f]['_average_']))
        
        avg_gap = np.mean(gaps)
        if avg_gap < lowest_avg_gap:
            lowest_averge_gap = avg_gap
            best_setting = setting
            best_corrs = corrs
            
    return best_setting, best_corrs, lowest_avg_gap
    
