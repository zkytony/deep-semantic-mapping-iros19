# Implementation of topological maps.
# Use python 3+
#
# author: Kaiyu Zheng

# Topological
import numpy as np
import random
import yaml
import os
import sys
import deepsm.util as util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
import copy
import math
import itertools
from collections import deque

from deepsm.graphspn.tbm.template import SingleEdgeTemplate, EdgeRelationTemplateInstance, ThreeNodeTemplate
from deepsm.util import CategoryManager


class TopologicalMap:

    """
    A TopologicalMap is a undirected graph. It can be constructed by a predefined set of nodes and a set of
    node connectivities. It supports partitioning by templates, and segmentation into less granularity of nodes.
    Functionality to modify the graph is not provided. Each node in a TopologicalMap is a PlaceNode, or a
    CompoundPlaceNode (see these classes below).
    """

    def __init__(self, nodes, conns):
        """
        Initializes a topological map from given nodes.

        @param nodes is a dictionary that maps from node id to actual node
        @param conns is a dictionary that maps from a node id to a set of tuples (neighbor_node_id, view_number)
        """
        self.nodes = nodes
        self.__conns = {} # conns is a map between node id and a dictionary of neighbor id -> edge object (different from the parameter!)
        self.edges = {} # edges is a map from edge id to edge. Used to conveniently partition graph by edges.
        for nid in conns:
            if nid not in self.__conns:
                self.__conns[nid] = {}
            for neighbor_nid, view_number in conns[nid]:
                # nid - neighbor_nid connection is new, but
                #  neighbor_nid - nid connection may have been observed.

                # Add an edge if needed
                edge = None
                
                if neighbor_nid in self.__conns and nid in self.__conns[neighbor_nid]:
                    edge = self.__conns[neighbor_nid][nid]
                else:
                    edge = Edge(len(self.edges), self.nodes[nid], self.nodes[neighbor_nid])
                    self.edges[edge.id] = edge
                    
                # Add a connectivity
                self.__conns[nid][neighbor_nid] = edge
                # TODO: We shouldn't need the following 3 lines. But it appears that in the data, there are nodes that don't have
                # mutual connections. We are assuming that if one node has an edge to another, then these two nodes
                # should be mutually connected. But this should be something handled on the dataset side.
                if neighbor_nid not in self.__conns:
                    self.__conns[neighbor_nid] = {}
                self.__conns[neighbor_nid][nid] = edge

        # Used when resetting the labels.
        self.__catg_backup = {nid:self.nodes[nid].label for nid in self.nodes}
    
        
    #--- Basic graph operations ---#
        
    def is_neighbor(self, node_id, test_id):
        return test_id in self.__conns[node_id]

    def neighbors(self, node_id):
        """
        Returns a set of neighbor node ids
        """
        return set(self.__conns[node_id].keys())

    def edge_between(self, node1_id, node2_id):
        if node2_id not in self.__conns[node1_id]:
            return None
        else:
            return self.__conns[node1_id][node2_id]

    def connections(self, nid):
        return self.__conns[nid]


    def hard_count(self):
        # Count the number of place nodes in this topological map
        c = 0
        for nid in self.nodes:
            c += self.nodes[nid].count()
        return c

    def copy(self):
        """
        Returns a new TopologicalMap which contains the same information as `self`. The new object
        is completely separate from `self`, meaning modifying any information in the copied topo-map
        does not affect `self`.
        """
        nodes_copy = copy.deepcopy(self.nodes)
        conns = {}   # a map from node id to a set of tuples (neighbor_node_id, view_number)
        for nid in nodes_copy:
            neighbors = set({})
            for neighbor_nid in self.neighbors(nid):
                neighbors.add((neighbor_nid, util.compute_view_number(nodes_copy[nid], nodes_copy[neighbor_nid])))
            conns[nid] = neighbors
        return TopologicalMap(nodes_copy, conns)

    def num_placeholders(self):
        """Returns the number of placeholders in this map"""
        return sum(1 for nid in self.nodes if self.nodes[nid].placeholder)


    def connected_edge_pairs(self):
        """
        Returns a dictionary from nid to set of all combinations of edge pairs.
        """
        node_edge_pairs = {}
        for nid in self.nodes: 
            neighbors = self.neighbors(nid)
            edges = set({})
            for nnid in neighbors:
                edges.add(self.edge_between(nid, nnid))
            pairs = set(itertools.combinations(edges, 2))  # order does not matter
            node_edge_pairs[nid] = pairs
        return node_edge_pairs


    #-- High level graph properties --#
    def connected_components(self):
        """
        Returns the connected components in this graph, each as a separate TopologicalMap instance.
        """
        # Uses BFS to find connected components
        copy_map = self.copy()
        to_cover = set(copy_map.nodes.keys())
        components = []
        while len(to_cover) > 0:
            start_nid = random.sample(to_cover, 1)[0]
            q = deque()
            q.append(start_nid)
            component_nodes = {start_nid:copy_map.nodes[start_nid]}
            component_conns = {start_nid:set()}
            visited = set()
            while len(q) > 0:
                nid = q.popleft()
                neighbors = copy_map.neighbors(nid)
                for neighbor_nid in neighbors:
                    if nid not in component_conns:
                        component_conns[nid] = set()
                    component_conns[nid].add((neighbor_nid,
                                              util.compute_view_number(copy_map.nodes[nid], copy_map.nodes[neighbor_nid])))
                    if neighbor_nid not in visited:
                        visited.add(neighbor_nid)
                        component_nodes[neighbor_nid] = copy_map.nodes[neighbor_nid]
                        q.append(neighbor_nid)
            component = TopologicalMap(component_nodes, component_conns)
            components.append(component)
            to_cover -= set(component.nodes.keys())
        return components
    
    
    #--- Partition ---#
    
    def partition(self, template, get_unused=False, relax=False):
        """
        Partitions this topological map by a given template (of nodes).

        Returns:
          a new topological map that is a 'supergraph' based on this one.
        If 'get_unused' is true, return a tuple where the first element
        is the 'supergraph' and the second element is a graph where nodes
        are not covered by this partition attempt.
        """
        def keep_unused_nn(nnid, unused_nodes):
            # Check if the given neighbor node id is in unused_nodes
            return nnid in unused_nodes

        random.seed()
        nodes_available = set(self.nodes.keys())
        nodes_used = set({})  # nodes already used to create new graph
        nodes_new = {}
        conns_new = {}
        nn_map = {} # map from topo map node's id to a node's id in the partitioned graph.
        
        while len(nodes_available) > 0:
            v_id = random.sample(nodes_available, 1)[0]
            nodes_match = template.match(self, self.nodes[v_id], nodes_used, relax=relax)
            if nodes_match is not None:
                nodes_available -= set(nodes_match)
                nodes_used |= set(nodes_match)
                
                u = CompoundPlaceNode(util.pick_id(nodes_new.keys(),
                                                   sum(n for n in nodes_match) % 211),
                                 list(self.nodes[n] for n in nodes_match))  # we need to keep the order using list.
                
                nodes_new[u.id] = u

                # Add a connection between u (the new node) and other new nodes mapped by neighbor
                # of matched nodes.
                for m_id in nodes_match:
                    for w_id in self.neighbors(m_id):
                        if nn_map.get(w_id) is not None:
                            # The neighbor is indeed already mapped to a new node.
                            new_w = nodes_new[nn_map.get(w_id)]   # new node in the partitioned graph
                            if new_w.id != u.id:  # Make sure we are not adding self edges in the new graph
                                util.sure_add(conns_new, u.id, (new_w.id, util.compute_view_number(u, new_w)))
                                util.sure_add(conns_new, new_w.id, (u.id, util.compute_view_number(new_w, u)))
                    nn_map[m_id] = u.id
            else:
                nodes_available.remove(v_id)

        # Return nodes that are not covered as another topological map object
        if get_unused:
            unused_nodes_keys = set(self.nodes.keys()) - nodes_used
            unused_nodes = {k:self.nodes[k] for k in unused_nodes_keys}
            unused_conns = {}
            for nid in unused_nodes:
                # We would only keep connections within unused nodes
                kept_conns = set((nbid, util.compute_view_number(self.nodes[nid], self.nodes[nbid])) \
                                 for nbid in set(filter(lambda x: keep_unused_nn(x, unused_nodes),
                                                        set(self.__conns[nid].keys()))))
                unused_conns[nid] = kept_conns
            return TopologicalMap(nodes_new, conns_new), TopologicalMap(unused_nodes, unused_conns)

        return TopologicalMap(nodes_new, conns_new)


    def partition_by_edge(self, template, relax=False):
        """
        Partitions this topological map by a given template (of edges).

        Returns:
           A tuple of two elements. The first element is a set of edges templates
        obtained in this partition attempt, and the second element is a set of single
        edge templates for uncovered edges. (Basically, two sets of EdgeTemplate objects)
           It does NOT return a new topological map object, since it is hard to define the
        nodes after partition.

        [Current code doesn't support hierarchy of edge templates. To some extent, it
         is unnecessary to support this; complex single templates may be equally good.]
        """
        edge_templates = set({})
        
        edges_available = set(self.edges.keys())  # set of available edge ids
        edges_used = set({})  # set of covered edge ids

        while len(edges_available) > 0:
            e_id = random.sample(edges_available, 1)[0]
            edges_match = template.match(self, self.edges[e_id], edges_used, relax=relax)
            if edges_match is not None:
                edges_available -= set(edges_match)
                edges_used |= set(edges_match)

                etmpl = template(*list(self.edges[eid] for eid in edges_match))
                edge_templates.add(etmpl)
            else:
                edges_available.remove(e_id)

        edges_uncovered = set(self.edges.keys()) - edges_used  # set of uncovered edge ids
        return edge_templates, set(SingleEdgeTemplate(self.edges[eid]) for eid in edges_uncovered)

        
    def partition_by_edge_relations(self):
        """
        Fully partition this graph into EdgeRelationTemplate objects. Returns a
        a dictionary that maps from a tuple (num_nodes, num_edge_pairs) indicating
        the type of the template, to a list of such EdgeRelationTemplate instances.
        """
        available_nodes = set(self.nodes.keys())
        node_edge_pairs = self.connected_edge_pairs()
        ert_map = {}  # returning dictionary
        
        # First, partition the graph using ThreeNodeTemplate. Then, create EdgeRelationTemplate instances with
        # single node and one edge pair, or no edge pair.
        # Then, we are left with edge paris. We remove edge_pairs and nodes as we go.

        # Step1
        supergraph, unused_graph = self.partition(ThreeNodeTemplate, relax=True, get_unused=True)
        # For every compound node in the super graph, we create an EdgeRelationTemplate with num_nodes=3, num_edge_paris=1
        ert_map[(3, 1)] = []
        for compound_nid in supergraph.nodes:
            nids = supergraph.nodes[compound_nid].to_place_id_list()
            center_nid = nids[1]
            edge_pair = (self.edge_between(center_nid, nids[0]), self.edge_between(center_nid, nids[2]))
            if edge_pair in node_edge_pairs[center_nid]:
                node_edge_pairs[center_nid].remove(edge_pair)
            elif tuple(reversed(edge_pair)) in node_edge_pairs[center_nid]:
                node_edge_pairs[center_nid].remove(tuple(reversed(edge_pair)))
            else:
                import pdb; pdb.set_trace()
                raise ValueError("Edge pair does not exist!")
                
            ert = EdgeRelationTemplateInstance(self.nodes[center_nid], nodes = [self.nodes[nid] for nid in nids],
                                       edge_pair = edge_pair)  # center node is meeting node
            ert_map[(3, 1)].append(ert)
            available_nodes -= set(nids)


        # Step2
        ert_map[(1, 1)] = []
        for nid in unused_graph.nodes:
            # Randomly pick an edge pair
            if len(node_edge_pairs[nid]) == 0:
                continue
            edge_pair = random.sample(node_edge_pairs[nid], 1)[0]
            node_edge_pairs[nid].remove(edge_pair)
            ert = EdgeRelationTemplateInstance(self.nodes[nid], nodes=[self.nodes[nid]], edge_pair=edge_pair)
            ert_map[(1, 1)].append(ert)
            available_nodes.remove(nid)

        # The remaining available nodes are ones with only one out-degree.
        ert_map[(1, 0)] = []
        while available_nodes:
            nid = available_nodes.pop()
            ert = EdgeRelationTemplateInstance(self.nodes[nid], nodes=[self.nodes[nid]], edge_pair=None)
            ert_map[(1, 0)].append(ert)
        assert len(available_nodes) == 0

        # Step3
        # We are left with edge pairs
        ert_map[(0, 1)] = []
        for nid in node_edge_pairs:
            while node_edge_pairs[nid]:
                edge_pair = node_edge_pairs[nid].pop()
                ert = EdgeRelationTemplateInstance(self.nodes[nid], nodes=None, edge_pair=edge_pair)
                ert_map[(0, 1)].append(ert)
            # We should have no edge pairs for this node remaining.
            assert len(node_edge_pairs[nid]) == 0

        return ert_map


    #--- Segmentation ---#
    def segment(self, remove_doorway=True):
        """
        Segments this topological map into a new TopologicalMap object where each node is a CompoundLabeledPlaceNode that
        includes nodes of the same label (supposedly) in the same room. This is done by sampling nodes and BFS from 
        sampled nodes to "flood" the graph.

        If `remove_doorway` is true, before segmenting the graph, all doorway nodes are replaced by a node which has class
        most common in the neighbors of the doorway node.

        ASSUME the nodes in `self` are all PlaceNode objects.
        """
        copy_map = self.copy()
        if remove_doorway:
            for nid in copy_map.nodes:
                if copy_map.nodes[nid].label == 'DW':
                    votes = [0]*util.CategoryManager.NUM_CATEGORIES
                    for nnid in copy_map.neighbors(nid):
                        votes[copy_map.nodes[nnid].label_num] += 1
                    copy_map.nodes[nid].label = util.CategoryManager.category_map(votes.index(max(votes)), rev=True)
        
        to_cover = set(copy_map.nodes.keys())
        nodes_new = {}
        conns_new = {}
        nn_map = {} # map from topo map node's id to a node's id in the segmented graph.
        while len(to_cover) > 0:
            start_nid = random.sample(to_cover, 1)[0]
            """BFS from start_nid"""
            q = deque()
            q.append(start_nid)
            same_label_nodes = [start_nid]
            visited = set({start_nid})
            while len(q) > 0:
                nid = q.popleft()
                neighbors = copy_map.neighbors(nid)
                for neighbor_nid in neighbors:
                    if neighbor_nid not in visited:
                        visited.add(neighbor_nid)
                        if copy_map.nodes[neighbor_nid].label_num == copy_map.nodes[start_nid].label_num:
                            same_label_nodes.append(neighbor_nid)
                            q.append(neighbor_nid)
            compound_node = CompoundLabeledPlaceNode(util.pick_id(nodes_new.keys(),
                                                                  sum(n for n in same_label_nodes) % 211),
                                                     list(copy_map.nodes[n] for n in same_label_nodes), copy_map.nodes[start_nid].label)
            nodes_new[compound_node.id] = compound_node
            """Remove covered nodes"""
            to_cover -= set(same_label_nodes)
            
            """Form connections"""
            for nid in same_label_nodes:
                for neighbor_nid in copy_map.neighbors(nid):
                    if nn_map.get(neighbor_nid) is not None:
                        # The neighbor is indeed already mapped to a new node in the segmented graph.
                        new_node_neighbor = nodes_new[nn_map.get(neighbor_nid)]   # new node in the partitioned graph
                        if new_node_neighbor.id != compound_node.id: # Make sure we are not adding self edges in the segmented graph
                            util.sure_add(conns_new, new_node_neighbor.id, (compound_node.id, util.compute_view_number(compound_node, new_node_neighbor)))
                            util.sure_add(conns_new, compound_node.id, (new_node_neighbor.id, util.compute_view_number(new_node_neighbor, compound_node)))
                nn_map[nid] = compound_node.id

        return TopologicalMap(nodes_new, conns_new)
    
            
    #--- Masking the graph ---#

    # Functions for running experiments
    def occlude_placeholders(self):
        """
        Sets the labels of nodes that are placeholders to be -1.
        """
        catg_map = {
            nid:util.CategoryManager.category_map(self.nodes[nid].label)
            if not self.nodes[nid].placeholder else util.CategoryManager.category_map('OC')
            for nid in self.nodes
        }
        self.assign_categories(catg_map)


    def swap_classes(self, swapped_classes):
        """
        swapped_classes (tuple): a tuple of two category names (str), which will be swapped.
        """
        catg_map = self.current_category_map()
        c1, c2 = util.CategoryManager.category_map(swapped_classes[0]), \
                 util.CategoryManager.category_map(swapped_classes[1])
        for nid in catg_map:
            if catg_map[nid] == c1:
                catg_map[nid] = c2
            elif catg_map[nid] == c2:
                catg_map[nid] = c1
        self.assign_categories(catg_map)
        return catg_map
        

    def current_category_map(self):
        """
        Returns a dictionary from node id to numerical value of the category.
        """
        catg_map = {
            nid:util.CategoryManager.category_map(self.nodes[nid].label)
            for nid in self.nodes
        }
        return catg_map


    def mask_by_policy(self, policy, **kwargs):
        """
        Sets node label to -1 according to `policy`, which is a function that determines if
        the node should be masked. It takes two parameters: topo_map and node. The policy
        is applied to nodes in random order.

        Note that the policy executed later will receive the information of formerly occluded graph nodes.

        kwargs: arguments for the policy.
        """
        nids = random.sample(list(self.nodes.keys()), len(self.nodes))
        for nid in nids:
            if policy(self, self.nodes[nid], **kwargs):
                self.nodes[nid].label = util.CategoryManager.category_map(-1, rev=True)

    
    def assign_categories_by_grid(self, cat_grid, placegrid):
        """
        Change the category of this topological map according to a category grid (cat_grid).
        And placegrid is the corresponding PlaceGrid object for this category grid.
        """
        h, w = cat_grid.shape
        for y in range(h):
            for x in range(w):
                mapped_places = placegrid.places_at(y, x)
                for p in mapped_places:
                    self.nodes[p.id].label = util.CatgoryManager.category_map(cat_grid[y, x], rev=True)


    def assign_categories(self, categories_map):
        """
        categories_map is a dictionary from node id to the numerical value of the category.
        """
        for nid in categories_map:
            if nid in self.nodes:
                self.nodes[nid].label = util.CategoryManager.category_map(categories_map[nid], rev=True)


    def reset_categories(self):
        for nid in self.__catg_backup:
            self.nodes[nid].label = self.__catg_backup[nid]


    #--- Visualizations ---#
    def visualize(self, ax, canonical_map_yaml_path=None, included_nodes=None, dotsize=13, img=None, consider_placeholders=False, show_nids=False):
        """Visualize the topological map `self`. Nodes are colored by labels, if possible."""
        # Open the yaml file
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
        plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")

        h, w = img.shape
        util.zoom_rect((w/2, h/2), img, ax, h_zoom_level=3.0, v_zoom_level=2.0)
        
        # Plot the nodes
        for nid in self.nodes:
            if included_nodes is not None and nid not in included_nodes:
                continue

            nid_text = str(nid) if show_nids else None
                
            place = self.nodes[nid]
            node_color = util.CategoryManager.category_color(place.label) if not (consider_placeholders and place.placeholder) else util.CategoryManager.PLACEHOLDER_COLOR
            pose_x, pose_y = place.pose  # gmapping coordinates
            plot_x, plot_y = util.plot_dot(ax, pose_x, pose_y, map_spec, img,
                                           dotsize=dotsize, color=node_color, zorder=2, linewidth=1.0, edgecolor='black', label_text=nid_text)

            # Plot the edges
            for neighbor_id in self.__conns[nid]:
                if included_nodes is not None and neighbor_id not in included_nodes:
                    continue

                util.plot_line(ax, place.pose, self.nodes[neighbor_id].pose, map_spec, img,
                               linewidth=1, color='black', zorder=1)


                
    def visualize_partition(self, ax, node_ids, canonical_map_yaml_path, ctype=1,
                            alpha=0.8, dotsize=6, img=None):
        """
        node_ids is the list of tuples where each represents node ids on a template.

        if `img` is not None, assume the visualization of the topological map is already plotted.
        """

        # First, visualize the graph in a whole.
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
            plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")
            self.visualize(ax, canonical_map_yaml_path=canonical_map_yaml_path, img=img, dotsize=13)
        
        colors = set({})
        for tmpl_node_ids in node_ids:
            color = util.random_unique_color(colors, ctype=ctype)
            colors.add(color)
            # Plot dots
            for nid in tmpl_node_ids:
                rx, ry = self.nodes[nid].pose
                px, py = util.transform_coordinates(rx, ry, map_spec, img)
                very_center = util.plot_dot(ax, rx, ry, map_spec, img, dotsize=dotsize,
                                            color=color, zorder=4, linewidth=1.0, edgecolor='black')

            # Plot edges
            for nid in tmpl_node_ids:
                for mid in tmpl_node_ids:
                    # Plot an edge
                    if nid != mid:
                        if self.edge_between(nid, mid):
                            util.plot_line(ax, self.nodes[nid].pose, self.nodes[mid].pose, map_spec, img,
                                           linewidth=3, color=color, zorder=3)

        return img


    def visualize_edge_relation_partition(self, ax, ert_map, canonical_map_yaml_path,
                                          dotsize=6, alpha=0.8, img=None):
        """
        Visualize the result of partitioning using edge relation templates. The result is specified
        by `ert_map`, a dictionary that maps from a tuple (num_nodes, num_edge_pairs) indicating
        the type of the template, to a list of such EdgeRelationTemplate instances.
        """
        # First, visualize the graph in a whole.
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
            plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")
            self.visualize(ax, canonical_map_yaml_path=canonical_map_yaml_path, img=img, dotsize=13)

        ctype = 2
        colors = set({})
        for t in sorted(ert_map):  # t is a (num_nodes, num_edge_pairs) tuple
            color = util.random_unique_color(colors, ctype=ctype)
            colors.add(color)
            
            for ert in ert_map[t]:
                if ert.nodes is not None:
                    # Plot dots
                    for node in ert.nodes:
                        rx, ry = node.pose
                        px, py = util.transform_coordinates(rx, ry, map_spec, img)
                        very_center = util.plot_dot(ax, rx, ry, map_spec, img, dotsize=dotsize,
                                                    color=color, zorder=4, linewidth=1.0, edgecolor='black')
                if ert.edge_pair is not None:
                    # plot edge pair
                    for edge in ert.edge_pair:
                        node1, node2 = edge.nodes
                        util.plot_line(ax, node1.pose, node2.pose, map_spec, img,
                                       linewidth=3, color=color, zorder=3)
            ctype += 1
        return img


########################################
#  Node
########################################
class Node:

    def __init__(self, id):
        """
        The id is expected to be unique in the graph.
        """
        self.id = id

    def __repr__(self):
        return "%s(%d)" % (type(self).__name__, self.id)


    def count(self):
        return 1


class PlaceNode(Node):

    def __init__(self, id, placeholder, pose, anchor_pose, label):
        """
        Args:
        
        id (int): id for this node
        placeholder (bool): True if this node is a placeholder
        pose (tuple): a tuple (x, y) of the node's pose in the topo map.
        anchor_pose (tuple): a tuple (x, y) of the node's anchored grid cell's pose
                             in the topo map.
        label (str): category of this place (e.g. DW).
        """
        self.id = id
        self.placeholder = placeholder
        self.pose = pose
        self.anchor_pose = anchor_pose
        self.label = label

    @property
    def label_num(self):
        return util.CategoryManager.category_map(self.label)

    def vscan(self, normalize=True):
        
        def normalize(polar_grid):
            vmap = {0: 0,
                    205: 1,
                    254: 2}
            for x in np.nditer(polar_grid, op_flags=['readwrite']):
                x[...] = vmap[int(x)]
            return polar_grid

        if self.polar_vscan is not None:
            return normalize(self.polar_vscan.grid.flatten())


class CompoundPlaceNode(Node):
    """
    Does not inherit PlaceNode because there is no well-defined attributes such as vscan for
    a CompoundPlaceNode.
    """

    def __init__(self, id, nodes):
        """
        nodes: a list of nodes included by this CompoundNode. Need to keep the order
        """
        self.id = id
        
        assert type(nodes) == type([])
        self.nodes = nodes
        self.node_ids = set({n.id for n in nodes})
        self.pose = (
            sum([n.pose[0] for n in self.nodes]) / len(self.nodes),
            sum([n.pose[1] for n in self.nodes]) / len(self.nodes)
        )
        self.anchor_pose = self.pose  # for the sake of having 'anchor_pose' field.
        self.label = 0

    def contains(self, node):
        return node.id in node_ids

    def __repr__(self):
        return "CompoundPlaceNode(%d){%s}" % (self.id, self.nodes)

    def to_catg_list(self):
        category_form = []
        for n in self.nodes:
            if isinstance(n, PlaceNode) or isinstance(n, CompoundLabeledPlaceNode):
                category_form.append(util.CategoryManager.category_map(n.label))
            elif isinstance(n, CompoundPlaceNode):
                category_form += n.to_catg_list()
        return category_form

    def to_vscans_list(self):
        vscan_form = []
        for n in self.nodes:
            if isinstance(n, PlaceNode):
                vscan_form.append(n.vscan())
            elif isinstance(n, CompoundPlaceNode):
                vscan_form += n.to_vscans_list()
        return vscan_form

    def to_place_id_list(self):
        pid_form = []
        for n in self.nodes:
            if isinstance(n, PlaceNode) or isinstance(n, CompoundLabeledPlaceNode):
                pid_form.append(n.id)
            elif isinstance(n, CompoundPlaceNode):
                pid_form += n.to_place_id_list()
        return pid_form
        

    def to_catg_ndarray(self):
        return np.array(self.to_list, dtype=int)


    def count(self):
        c = 0
        for n in self.nodes:
            c += n.count()
        return c

    
class CompoundLabeledPlaceNode(CompoundPlaceNode):
    """
    Same as CompoundPlaceNode except it has a `label` attribute. Used for graph segmentation.
    """
    def __init__(self, id, nodes, label):
        super().__init__(id, nodes)
        self.label = label
        


########################################
#  Edge
########################################
class Edge:
    """
    Undirected edge. Two edges are equal if node1 and node2 have equal ids respectively.
    """    
    
    def __init__(self, id, node1, node2, view_nums=None):
        """
        The id is expected to be unique in the graph edges.
        """
        self.id = id
        self.nodes = (node1, node2)
        
        # view numbers
        if view_nums is None:
            self.view_nums = (util.compute_view_number(node1, node2),
                              util.compute_view_number(node2, node1))
        else:
            assert len(view_nums) == 2
            self.view_nums = view_nums

            
    def __repr__(self):
        return "#%d[%d<%d>---%d<%d>]" % (self.id, self.nodes[0].id, self.view_nums[0],
                                         self.nodes[1].id, self.view_nums[1])


    def __eq__(self, other):
        """
        Two edges are equal if node1 and node2 have equal ids respectively (without order).
        """
        my_ids = set({self.nodes[0].id, self.nodes[1].id})
        other_ids = set({other.nodes[0].id, other.nodes[1].id})
        return my_ids == other_ids


    def __hash__(self):
        return hash((self.nodes[0].id, self.nodes[1].id))

    @classmethod
    def get_triplet_from_edge_pair(cls, topo_map, edge_pair, center_nid, catg=False):
        e1_nids = (edge_pair[0].nodes[0].id, edge_pair[0].nodes[1].id)
        e2_nids = (edge_pair[1].nodes[0].id, edge_pair[1].nodes[1].id)
        nid1 = e1_nids[1-e1_nids.index(center_nid)]
        nid2 = e2_nids[1-e2_nids.index(center_nid)]
        if catg:
            return (CategoryManager.category_map(topo_map.nodes[nid1].label),
                    CategoryManager.category_map(topo_map.nodes[center_nid].label),
                    CategoryManager.category_map(topo_map.nodes[nid2].label))
        else:
            return (nid1, center_nid, nid2)
