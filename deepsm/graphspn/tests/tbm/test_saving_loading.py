import os
import sys
import tensorflow as tf

import numpy as np
import libspn as spn

from spn_topo.spn_model import SpnModel
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn, InstanceSpn
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, SingleEdgeTemplate, PairEdgeTemplate, SingletonTemplate, PairTemplate, ThreeNodeTemplate


if __name__ == "__main__":
    print("...ThreeNode...")
    sess = tf.Session()

    path = "three_node_spn.spn"

    sample = np.array([3, 1, 3], dtype=int)

    spn1 = NodeTemplateSpn(ThreeNodeTemplate)
    spn1.generate_random_weights()
    spn1.init_weights_ops()
    spn1.init_learning_ops()
    spn1.initialize_weights(sess)
    val1 = spn1.evaluate(sess, sample)[0]
    print(val1)

    spn1.save(path, sess)

    # New session
    sess.close()
    sess = tf.Session()

    spn1_1 = NodeTemplateSpn(ThreeNodeTemplate)
    spn1_1.init_weights_ops()
    spn1_1.initialize_weights(sess)
    spn1_1.load(path, sess)
    spn1_1.init_learning_ops()
    val1_1 = spn1_1.evaluate(sess, sample)[0]
    print(val1_1)

    
    spn1_2 = NodeTemplateSpn(ThreeNodeTemplate)
    spn1_2.init_weights_ops()    
    spn1_2.initialize_weights(sess)
    spn1_2.load(path, sess)
    spn1_2.init_learning_ops()
    val1_2 = spn1_2.evaluate(sess, sample)[0]
    print(val1_2)

    sess.close()
