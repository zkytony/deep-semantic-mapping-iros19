# Test adding constant to weights.
import os
import sys
import tensorflow as tf

import numpy as np
import libspn as spn

from spn_topo.spn_model import SpnModel
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn, InstanceSpn
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, SingleEdgeTemplate, PairEdgeTemplate, SingletonTemplate, PairTemplate, ThreeNodeTemplate

if __name__ == "__main__":
    sess = tf.Session()
    spn1 = NodeTemplateSpn(SingletonTemplate)
    spn1.generate_random_weights()
    spn1.init_weights_ops()
    spn1.initialize_weights(sess)
    w = sess.run(spn1.root.weights.node.get_value())
    print(type(w))
    print(w)

    SpnModel.make_weights_same(sess, spn1.root, 100)
    w = sess.run(spn1.root.weights.node.get_value())
    print(type(w))
    print(w)

    sess.close()

