# Templates.
#
# author: Kaiyu Zheng
from abc import ABC, abstractmethod
import random
from collections import deque
from deepsm.util import abs_view_distance, compute_view_number, CategoryManager
import itertools

########################################
# Template
########################################
class Template(ABC):

    @classmethod
    @abstractmethod
    def size(cls):
        """
        A template has a defined size. (e.g. number of nodes/edges). 
        Useful for sorting template by complexity.
        """
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        """
        An integer that identifies the type of this template
        """
        pass

########################################
# NodeTemplate
########################################
class NodeTemplate(Template):

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass

        
    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, excluded_nodes, **kwargs):
        pass


    @classmethod
    def size(cls):
        return cls.num_nodes()

    @classmethod
    def code(cls):
        return 0


class SingletonTemplate(NodeTemplate):
    """
    Single node
    """
    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, P, excluded_nodes=set({}), **kwargs):
        """
        Returns a list of node ids that are matched in this template.
        """
        # Hardcode the match check
        return (P.id,)

class PairTemplate(NodeTemplate):
    """
    Simple pair
    """
    @classmethod
    def num_nodes(cls):
        return 2

    @classmethod
    def match(cls, graph, P, excluded_nodes=set({}), **kwargs):
        """
        Returns a list of node ids that are matched in this template. Order
        follows A-P or P-A.
        """
        # Hardcode the match check
        pivot_neighbors = graph.neighbors(P.id)

        for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
            if A_id not in excluded_nodes and A_id != P.id:
                return (A_id, P.id)
        return None


class StarTemplate(NodeTemplate):

    """
          A
          |
      B - P - C
          |
          D
    """
    @classmethod
    def num_nodes(cls):
        return 5
    
    @classmethod
    def match(cls, graph, X, excluded_nodes=set({}), relax=False):
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)

            #A
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    #B
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            #C
                            for C_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                if C_id not in excluded_nodes | set({A_id, B_id}):
                                    #D
                                    for D_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                        if D_id not in excluded_nodes | set({A_id, B_id, C_id}):
                                            return (A_id, B_id, P.id, C_id, D_id)
                                                                            
        nodes = match_by_pivot(X, excluded_nodes=excluded_nodes)
        if nodes is None:
            neighbors = graph.neighbors(X.id)
            # Didn't get luck when using X as the pivot. Use one of its neighbor.
            for N_id in random.sample(neighbors, len(neighbors)):
                if N_id not in excluded_nodes:
                    nodes = match_by_pivot(graph.nodes[N_id], excluded_nodes=excluded_nodes)
                    if nodes is not None:
                        return tuple(nodes)
            return None
        else:
            return nodes
    

    
class ThreeNodeTemplate(NodeTemplate):

    """
    Simple three node structure
    
    A--(P)--B

    P is the pivot. A and B are not connected
    """

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def match(cls, graph, X, excluded_nodes=set({}), relax=False):
        """
        Returns a list of node ids that are matched in this template. Order
        follows A-P-B, where P is the pivot node. X is the node where the
        matching starts, but not necessarily the pivot.
        """
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes:
                            if A_id == B_id:
                                continue
                            # if not relax and A_id in graph.neighbors(B_id):
                            #     continue
                            return (A_id, P.id, B_id)

        nodes = match_by_pivot(X, excluded_nodes=excluded_nodes)
        if nodes is None:
            neighbors = graph.neighbors(X.id)
            # Didn't get luck when using X as the pivot. Use one of its neighbor.
            for N_id in random.sample(neighbors, len(neighbors)):
                if N_id not in excluded_nodes:
                    nodes = match_by_pivot(graph.nodes[N_id], excluded_nodes=excluded_nodes)
                    if nodes is not None:
                        return tuple(nodes)
            return None
        else:
            return nodes

    
########################################
# EdgeTemplate
########################################
class EdgeTemplate(Template):

    @abstractmethod
    def __init__(self, *edges):
        pass

    @abstractmethod
    def to_sample(self):
        """
        Returns tuple of equivalent samples. Each sample is also
        a tuple, where the first half of elements are semantics, and
        the second half of elements are view numbers.
        """
        pass

    @classmethod
    @abstractmethod
    def num_edges(cls):
        pass

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass

    @classmethod
    @abstractmethod
    def num_vars(cls):
        # Total number of variables (including semantic classes and view numbers)
        pass

    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, excluded_nodes, **kwargs):
        pass

    @classmethod
    def size(cls):
        return cls.num_edges()
    
    @classmethod
    def code(cls):
        return 1


    @classmethod
    def semantic_sample_node_iterator(cls, sample):
        """
        sample (tuple): a single semantic sample. It has half of the size
                        of a sample returned by the to_sample() function.

        Because the actual graph nodes in an edge template may repeat,
        it would be useful if the edge template implements a function
        that yields the next **tuple** of indices in `sample` that belong
        to the same node
        """
        raise NotImplemented
    


class SingleEdgeTemplate(EdgeTemplate):

    def __init__(self, *edges):
        # (A) -- e -- (B)
        if len(edges) != 1:
            raise ValueError("%s only takes 1 edge. You provided %d." % (self.__class__.__name__,
                                                                         len(edges)))
        self.edges = edges

        
    def to_sample(self):
        """
        Returns a pair of equivalent samples
        """
        n1, n2 = self.edges[0].nodes
        v1, v2 = self.edges[0].view_nums
        return ((n1.label_num, n2.label_num, v1, v2),
                (n2.label_num, n1.label_num, v2, v1))


    @classmethod
    def num_edges(cls):
        return 1

    @classmethod
    def num_nodes(cls):
        return 2

    @classmethod
    def num_vars(cls):
        return 4


    @classmethod
    def match(cls, graph, e, excluded_nodes, **kwargs):
        """
        Returns a list of edge ids that are matched in this template.
        """
        # Hardcode the match check
        return (e.id,)

    
    @classmethod
    def semantic_sample_node_iterator(cls, sample):
        """
        sample: (n1_label, n2_label)
        """
        yield (0,)
        yield (1,)
            

    def __repr__(self):
        return "{%s}" % (self.edges[0])

    
class PairEdgeTemplate(EdgeTemplate):
    """
    This is the same as the ThreeNodeTemplate in terms of graph structure.
    """

    def __init__(self, *edges):
        # (A) -- e-- (B) -- f -- (C)
        e, f = edges
        self.edges = (e, f)

        if e.nodes[0].id == f.nodes[0].id:
            self.nends = [1, 0, 0, 1]   # indicates the order of edge end ids.
                                        # First two are on e, last two are on f.
                                        # The middle two are for B.
        elif e.nodes[0].id == f.nodes[1].id:
            self.nends = [1, 0, 1, 0]
        elif e.nodes[1].id == f.nodes[0].id:
            self.nends = [0, 1, 0, 1]
        elif e.nodes[1].id == f.nodes[1].id:
            self.nends = [0, 1, 1, 0]

    def to_sample(self):
        """
        Returns a pair of equivalent samples
        """
        # (A) -- e-- (B) -- f -- (C)
        a,be,bf,c = self.nends
        e, f = self.edges
        return ((e.nodes[a].label_num, e.nodes[be].label_num, f.nodes[bf].label_num, f.nodes[c].label_num, e.view_nums[a], e.view_nums[be], f.view_nums[bf], f.view_nums[c]),
                (f.nodes[c].label_num, f.nodes[bf].label_num, e.nodes[be].label_num, e.nodes[a].label_num, f.view_nums[c], f.view_nums[bf], e.view_nums[be], e.view_nums[a]))
    

    @classmethod
    def num_edges(cls):
        return 2

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def num_vars(cls):
        return 8

    @classmethod
    def match(cls, graph, e, excluded_edges, relax=False):
        """
        Returns a list of edge ids that are matched in this template. e-f, f-e
        """
        # Hardcode the match check
        # (A) -- e -- (B) -- f -- (C)  [note: there's no edge between A and C]
        end = random.randrange(2)
        neighbors = graph.neighbors(e.nodes[end].id)
        for Aid in random.sample(neighbors, len(neighbors)):
            if Aid != e.nodes[1-end].id:
                f = graph.edge_between(Aid, e.nodes[end].id)
                if f.id not in excluded_edges:
                    if not relax and graph.edge_between(Aid, e.nodes[1-end].id) is not None:
                        continue
                    return (e.id, f.id)
        return None


    @classmethod
    def semantic_sample_node_iterator(cls, sample):
        """
        sample: (A_label, B_label, B_label, C_label)
        """
        yield (0,)
        yield (1,2)
        yield (3,)


    def __repr__(self):
        return "{%s, %s}" % (self.edges[0], self.edges[1])



########################################
# EdgeRelationTemplate
########################################
class AbsEdgeRelationTemplate(Template):
        
    @classmethod
    @abstractmethod
    def size(cls):
        pass
            
    @classmethod
    def code(cls):
        return 2


class EdgeRelationTemplate(AbsEdgeRelationTemplate):

    """
    Unlike the NodeTemplate and EdgeTemplate, the EdgeRelationTemplate is not an
    abstract class. Instead, it is instantiated directly. This is because this
    template has more variety of structure, and partitioning a graph using this
    template actually relies on the matching methods of specific subclasses of
    NodeTemplate and EdgeTemplate.

    This template is specified by both semantic classes (nodes) and a edge relation
    (edge pair).
    """
    
    def __init__(self, meeting_node, nodes=None, edge_pair=None, distance_func=abs_view_distance):
        """
        Initialize an instance of EdgeRelationTemplate. 

        `meeting_node` (PlaceNode) is a node that defines this template.
        `edge_pair` (tuple) is a pair of edges that meets at the meeting node.
        `nodes` (list) is a list of nodes used to obtain category values. The order matters.
                       It might be None, in which case the `meeting_node` is not counted
                       as a piece of information stored in this template
        """
        if edge_pair is None and nodes is None:
            raise ValueError("Template cannot be empty. At least one edge pair or" \
                             "one node needs to be provided")
        self.meeting_node = meeting_node
        self.edge_pair = edge_pair
        self.nodes = nodes
        
        self.vdist = None

        if edge_pair is not None:

            # Get the other two nodes in the edge pair besides the meeting node
            other_nodes = []
            for edge in edge_pair:
                i = edge.nodes.index(meeting_node)
                other_nodes.append(edge.nodes[1-i])

            # Compute view numbers and relative distancex
            v1 = compute_view_number(meeting_node, other_nodes[0])
            v2 = compute_view_number(meeting_node, other_nodes[1])
            self.vdist = distance_func(v1, v2) # edge view relative distance


    @property
    def num_nodes(self):
        """
        Returns the nubmer of semantic class variables in this instance of the template.
        """
        return len(self.nodes) if self.nodes is not None else 0
    
    @property
    def num_edge_pair(self):
        """
        Returns the nubmer of edge pair relation variable in this template.
        
        The returned value is either 0 or 1.
        """
        return 1 if self.edge_pair is not None else 0
        

    @property
    def num_vars(self):
        """
        Returns the number of variables in this edge relation template. The variables
        are nodes and edge relations.
        """
        return self.num_nodes + self.num_edge_pair

        
    def __repr__(self):
        return "[#{%d}(%d, %d)]" % (self.meeting_node.id, self.num_nodes, self.num_edge_pair)


    def to_sample(self):
        """
        Returns a tuple. First element is a list of ordered node classes (numbers). Second element is
        the relative distance between the two edges in the pair. The first element is None if the meeting
        node is not used for information of this template. The second element is None if this
        template only contains one node class and no edge relations.
        """
        if self.nodes is not None:
            return [CategoryManager.category_map(n.label) for n in self.nodes], self.vdist
        else:
            return None, self.vdist

    @classmethod
    def get_class(cls, template_tuple):
        """
        template_tuple (tuple) representation of edge relation template (num_nodes, num_edge_pair)
        """
        class_map = {
            (3,1): ThreeRelTemplate,
            (1,1): SingleRelTemplate,
            (1,0): SingleTemplate,
            (0,1): RelTemplate,
        }
        try:
            return class_map[template_tuple]
        except KeyError as ex:
            print("No corresponding class.")
            raise ex

    @classmethod
    def size():
        """
        This template is to be instantiated.
        """
        raise NotImplementedError
            
# Some classes for typical relation templates
class ThreeRelTemplate(AbsEdgeRelationTemplate):
    @classmethod
    def to_tuple(cls):
        return(3, 1)

    @classmethod
    def size(cls):
        return 3 # 3 nodes

class SingleRelTemplate(AbsEdgeRelationTemplate):
    @classmethod
    def to_tuple(cls):
        return(1, 1)

    @classmethod
    def size(cls):
        return 2 # 1 nodes, 1 edge pair.


class SingleTemplate(AbsEdgeRelationTemplate):
    # note: different from SingletonTemplate.
    @classmethod
    def to_tuple(cls):
        return(1, 0)

    @classmethod
    def size(cls):
        return 1 # 1 node


class RelTemplate(AbsEdgeRelationTemplate):
    @classmethod
    def to_tuple(cls):
        return(0, 1)

    @classmethod
    def size(cls):
        return 0 # 0 nodes
