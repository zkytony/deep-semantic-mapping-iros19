# author: Kaiyu Zheng

import numpy as np
import deepsm.util as util
from deepsm.graphspn.tbm.topo_map import PlaceNode, TopologicalMap

def build_graph(graph_file_path):
    """
    Reads a file that is a specification of a graph, then construct
    a graph object using the TopologicalMap class. The TopologicalMap
    class is for undirected graphs, where each node on the graph contains
    a fixed label and edges do not have labels. It is possible for
    nodes to have uncertain labels.
    
    This script reads a file of format ".ug" that indicates "undirected graph".
    The format is:
    
    <class> <class_name>
    --
    <Node_Id> <pos_x> <pos_y> <class>
    --
    Log (or Linear)
    <Node_Id> <likelihood_class0> <likelihood_class1> ...
    --
    <Node_Id_1> <Node_Id_2>
    
    The first part specifies classes in the graph. <class_name> is a string,
    and <class> is a numerical value to represent the class string. The <class>
    must be starting from 0 and the class in a new line is incremented by one.

    The second part specifies the nodes. It is not required that Node_Id
    starts from 0. Also pos_x and pos_y should be in metric coordinates. <class>
    is the label for that node. <Node_Id> is of type int, <pos_x> and <pos_y> are
    of type float. <class> is the groundtruth class of the node, an integer.

    The third part specifies likelihoods of labels for each node (evidence).
    The first line of this part indicates whether the likelihood values are in
    the log scale or linear scale. This line can either be "Log" or "Linear".
    For subsequent lines, each starts with a Node_Id followed by a whitespace-separated list of
    likelihood values. The first likelihood value is for class 0, second for
    class 1, etc.  Eventually likelihoods will be converted into log likelihoods
    when this function returns.
    
    
    The fourth part specifies the edges. The edge is undirected, and
    the node ids should be defined in the first part of the file.

    There could be arbitrarily many empty lines, and can have comments by beginning the line with "#"
    
    This function can be used to parse a graph file and generate a TopologicalMap
    object, and then produce the likelihood file for a graph, which can
    be used in GraphSPN experiments.
    """
    with open(graph_file_path) as f:
        lines = f.readlines()

    classes = []  # List of classes where each element is a string of class name.
    likelihoods = {} # Map from node id to a map from class to likelihood
    nodes = {}  # Map from node id to an actual node object
    conns = {}  # Map from node id to set of tuples (neighbor node id, view number)
    use_log = None
        
    state = "classes"
    for i, line in enumerate(lines):
        # Handle transition, if encountered
        try:
            line = line.rstrip()
            if len(line) == 0:
                continue # blank line
            if line.startswith("#"):
                continue # comment

            if line == "--":
                state = _next_state(state)
                continue # read next line
            # This line belongs to a state
            if state == "classes":
                _parse_class(line, classes)
            elif state == "nodes":
                _parse_node(line, nodes, likelihoods, classes)
            elif state == "likelihoods":
                use_log = _parse_likelihood(line, likelihoods, classes, nodes, use_log)
            elif state == "edges":
                _parse_edge(line, conns, nodes)
            else:
                raise ValueError("Unexpected state %s" % state)
        except Exception as e:
            print("Line %d caused an Error:" % i)
            print(e)
            raise e
        
    return TopologicalMap(nodes, conns), likelihoods # We are done.            

#####################
# Utility functions #
#####################
def _next_state(state):
    if state == "classes":
        return "nodes"
    elif state == "nodes":
        return "likelihoods"
    elif state == "likelihoods":
        return "edges"
    elif state == "edges":
        return None
    else:
        raise ValueError("Unexpected state %s" % state)

def _parse_class(line, classes):
    """
    <class> <class_name>
    """
    tokens = line.split()  # split on whitespaces
    class_index, class_name = int(tokens[0]), tokens[1]
    if class_index != len(classes):
        raise ValueError("Class indices are not continuous and start from 0. Expect: %d; Got: %d" % (len(classes), class_index))
    classes.append(class_name)

def _parse_node(line, nodes, likelihoods, classes):
    """
    <Node_Id> <pos_x> <pos_y> <class>
    """
    tokens = line.split()  # split on whitespaces
    nid, x, y, class_index = int(tokens[0]), float(tokens[1]), float(tokens[2]), int(tokens[3])
    if nid in nodes:
        raise ValueError("Node %d is already defined" % (nid))
    if class_index < 0 or class_index >= len(classes):
        raise ValueError("Invalid class index %d" % class_index)
    nodes[nid] = PlaceNode(nid, False, (x,y), (x,y), classes[class_index])
    likelihoods[nid] = np.zeros(len(classes))

def _parse_likelihood(line, likelihoods, classes, nodes, use_log):
    """
    Log (or Linear)
    <Node_Id> <likelihood_class0> <likelihood_class1> ...

    Returns true if likelihood is in log space.
    """
    if use_log is None:
        if line == "Log":
            return True
        else:
            return False
        
    tokens = line.split()  # split on whitespaces
    if len(tokens)-1 != len(classes):
        raise ValueError("Expecting %d likelihood values but got %d" % (len(classes), len(tokens)-1))

    nid = int(tokens[0])
    if nid not in nodes:
        raise ValueError("Node %d is undefined" % (nid))
    node_likelihoods = list(map(float, tokens[1:]))
    if not use_log:
        node_likelihoods = np.log(node_likelihoods)
    likelihoods[nid] = node_likelihoods
    return use_log

def _parse_edge(line, conns, nodes):
    """
    <Node_Id_1> <Node_Id_2>

    Note: nodes should be a map from node id to a node object, instead of a tuple.
    """
    tokens = line.split()  # split on whitespaces
    nid1, nid2 = int(tokens[0]), int(tokens[1])
    if nid1 not in nodes:
        raise ValueError("Node %d is undefined" % nid1)
    if nid2 not in nodes:
        raise ValueError("Node %d is undefined" % nid2)
    if nid1 not in conns:
        conns[nid1] = set()
    if nid2 not in conns:
        conns[nid2] = set()
    conns[nid1].add((nid2, util.compute_view_number(nodes[nid1], nodes[nid2])))
    conns[nid2].add((nid1, util.compute_view_number(nodes[nid2], nodes[nid1])))
    
    
