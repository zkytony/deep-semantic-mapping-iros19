# Full model
#
# Input: data
# Output: Likelihood
#
# data -> DGSM (-> likelihoods -> GraphSPN) -> likelihoods

import numpy as np
import libspn as spn
from dgsm.model import PlaceSubModel
from graphspn.tbm.spn_template import InstanceSpn
from graphspn.tbm.topo_map import TopologicalMap
import graphspn.util as util

def interpret(likelihoods):
    """Intepret likelihoods. Returns the vector of
    normalized marginal likelihoods"""
    marginals = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                       (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return marginals


class PlaceModel:
    """DGSM wrapper. Train and output classification likelihoods for given
    number of classes."""
    
    def __init__(self, data, **kwargs):
        """
        Build place model network structure. (For simplicity, we just retrain
        and don't save the trained model.)

        Args:
          data (dgsm.data.Data): Virtual scan dataset
          kwargs: Parameters for dgsm.model.PlaceSubModel
        """
        self._models = []
        for i in range(util.CategoryManager.NUM_CATEGORIES):
            self._models.append(PlaceSubModel(data=data, **kwargs))

    def train(self, **kwargs):
        """
        Train the model with data provided in initialization.
        """
        for model in self._models:
            model.train(**kwargs)
        

    def classify(self, vscan):
        """
        Outputs likelihoods for the given virtual scan.

        vscan (np.ndarray) Array representation of the polar virtual scan.
        """
        # TODO: Need to weight the SPN outputs
        likelihoods = []
        for model in self._models:
            likelihood_val = model._sess.run(model._train_likelihood,
                                             feed_dict={model._ivs: vscan})
            likelihoods.append(likelihood_val)
        return interpret(likelihoods)


class MapModel:
    """GraphSPN wrapper. Given likelihood values of places on a topological graph,
    output corresponding likelihood values after potential correction."""
    
    def __init__(self, topo_map, sess, tmpl_spns, **kwargs):
        """Initializes an instance GraphSPN from given topo_map.

        tmpl_spns (list): a list of tuples (TemplateSpn, Template). Note that assume
                          all templates are either node templates or edge templates.
        **kwargs (Same as graphspn.InstanceSpn):
           spn_paths (dict): a dictionary from Template to path. For loading the spn for Template at
                             path.
           num_partitions (int) number of child for the root sum node.
           seq_id (str): sequence id for the given topo map instance. Used as identified when
                         saving the instance spn. Default: "default_1"
           no_init (bool): True if not initializing structure; user might want to load structure
                           from a file.
           visualize_partitions_dirpath (str): Path to save the visualization of partitions on each child.
                                            Default is None, meaning no visualization is saved.
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
           db_name (str): Database name. Required if visualization_partitions_dirpath is not None.

           If template is EdgeTemplate, then:
             divisions (int) number of views per place
        """
        self._topo_map = topo_map
        self._model = InstanceSpn(topo_map, sess, *spn_tmpls, **kwargs)
    
    def marginal_inference(self, sess, query_lh):
        """Outputs likelihood values by feeding the given likelihoods into the network.

        Args:
           query_lh (dict) dictionary from node id to vector of likelihoods for that node"""
        query_nids = list(self._topo_map.nodes.keys())
        query = {k:-1 for k in query_nids}  # infer all nodes
        all_likelihoods = self._model.marginal_inference(sess, query_nids, query, query_lh=query_lh)
        all_marginals = {}
        for nid in query:
            all_marginals[nid] = interpret(all_likelihoods[nid])
        return all_marginals
