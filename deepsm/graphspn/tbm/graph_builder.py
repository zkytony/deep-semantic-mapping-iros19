# Reads a file that is a specification of a graph, then construct
# a graph object using the TopologicalMap class. The TopologicalMap
# class is for undirected graphs, where each node on the graph contains
# a fixed label and edges do not have labels. It is possible for
# nodes to have uncertain labels.
#
# This script reads a file of format ".ug" that indicates "undirected graph".
# The format is:
#
# <Node_Id> <pos_x> <pos_y>
# --
# <Node_Id_1> <Node_Id_2>
# --
# <Node_Id> <class> <likelihood>
#
# The first part (before "--") specifies the nodes. It is required that Node_Id
# starts from 0. Also pos_x and pos_y should be in metric coordinates. <class>
# is the label for that node. <Node_Id> is of type int, <pos_x> and <pos_y> are
# of type float.
#
# The second part (after "--") specifies the edges. The edge is undirected, and
# the node ids should be defined in the first part of the file.
#
# The third part (after second "--") specifies graph labels for each node.
# <class> is of type int (Better to be within the range from 0 to
# util.CategoryManager.NUM_CATEGORIES-1. You need to specify
# a likelihood value for that class (if the value starts with 'l', then it is
# considered as a log-likelihood value). Therefore, we can have multiple lines
# about the same node but different classes with different likelihoods.
#
# This script can be used to parse a graph file and generate a TopologicalMap
# object, and then produce the likelihood file for a graph, which can
# be used in GraphSPN experiments.

