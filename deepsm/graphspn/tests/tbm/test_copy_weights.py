#!/usr/bin/env python

# Tests SpnMode.copyweights method

import tensorflow as tf
import libspn as spn

import numpy as np
import random

from spn_topo.spn_model import SpnModel
from spn_topo.tbm.spn_template import TemplateSpn, NodeTemplateSpn, EdgeTemplateSpn
from spn_topo.tbm.template import EdgeTemplate, NodeTemplate, SingleEdgeTemplate, PairEdgeTemplate, SingletonTemplate, PairTemplate, ThreeNodeTemplate

def case1():
    # NodeTemplateSpn copyweights.

    seed = 100
    
    dense_gen = spn.DenseSPNGenerator(num_decomps=1, num_subsets=2,
                                      num_mixtures=2, input_dist=spn.DenseSPNGenerator.InputDist.RAW,
                                      num_input_mixtures=2)
    weights_gen = spn.WeightsGenerator(init_value=spn.ValueType.RANDOM_UNIFORM(10,
                                                                               11),
                                       trainable=True)

    
    
    with tf.Session() as sess:    
        ivs1 = spn.IVs(num_vars=4, num_vals=5)
        ivs2 = spn.IVs(num_vars=4, num_vals=8)
        rnd = random.Random(seed)
        root = dense_gen.generate(ivs1, ivs2, rnd=rnd)
        weights_gen.generate(root)
        sess.run(spn.initialize_weights(root))

        ivs1_ = spn.IVs(num_vars=4, num_vals=5)
        ivs2_ = spn.IVs(num_vars=4, num_vals=8)
        rnd = random.Random(seed)
        root_ = dense_gen.generate(ivs1_, ivs2_, rnd=rnd)
        weights_gen.generate(root_)
        sess.run(spn.initialize_weights(root_))


        sample = np.array([3, 2, 2, 1, 4, 1, 2, 4], dtype=int)
        SpnModel.copyweights(sess, root, root_)

    # Single
#     with tf.Session() as sess:
#         sample = np.array([3, 2, 1], dtype=int)
        
#         spn1 = NodeTemplateSpn(ThreeNodeTemplate, seed=seed)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = NodeTemplateSpn(ThreeNodeTemplate, seed=seed)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
        
#         print("Before copying")
#         print(val1)
#         print(val2)

#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val1)
#         print(val2)


# def case2():
#     # EdgeTemplateSpn copyweights.

#     seed = 100

#     # Single
#     print("...Pair...")
#     with tf.Session() as sess:
#         sample = np.array([3, 2, 2, 1, 4, 1, 2, 4], dtype=int)
        
#         spn1 = EdgeTemplateSpn(PairEdgeTemplate)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = EdgeTemplateSpn(PairEdgeTemplate)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("Before copying")
#         print(val1)
#         print(val2)
#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2


        

if __name__ == "__main__":

    ## THE TESTS SHOW THAT copy_weights() DOES NOT WORK AS EXPECTED.
    #case1()
    case1()


#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2



#     # Pair
#     print("...Pair...")
#     with tf.Session() as sess:
#         sample = np.array([3, 2], dtype=int)
        
#         spn1 = NodeTemplateSpn(PairTemplate, seed=seed)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = NodeTemplateSpn(PairTemplate, seed=seed)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("Before copying")
#         print(val1)
#         print(val2)
#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2


#     # Three
#     print("...ThreeNode...")
#     with tf.Session() as sess:
#         sample = np.array([3, 1, 2], dtype=int)
        
#         spn1 = NodeTemplateSpn(ThreeNodeTemplate, seed=seed)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = NodeTemplateSpn(ThreeNodeTemplate, seed=seed)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("Before copying")
#         print(val1)
#         print(val2)
#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2


# def case2():
#     # EdgeTemplateSpn copyweights.

#     seed = 100

#     # Single
#     print("...Pair...")
#     with tf.Session() as sess:
#         sample = np.array([3, 2, 2, 1, 4, 1, 2, 4], dtype=int)
        
#         spn1 = EdgeTemplateSpn(PairEdgeTemplate, seed=seed)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = EdgeTemplateSpn(PairEdgeTemplate, seed=seed)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("Before copying")
#         print(val1)
#         print(val2)
#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2



#     # Pair
#     print("...Single...")
#     with tf.Session() as sess:
#         sample = np.array([3, 1, 4, 1], dtype=int)
        
#         spn1 = EdgeTemplateSpn(SingleEdgeTemplate, seed=seed)
#         sess.run(spn1._initialize_weights)
#         val1 = spn1.evaluate(sess, sample)[0]

#         spn2 = EdgeTemplateSpn(SingleEdgeTemplate, seed=seed)
#         sess.run(spn2._initialize_weights)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("Before copying")
#         print(val1)
#         print(val2)
#         assert val1 != val2
        
#         SpnModel.copyweights(sess, spn1.root, spn2.root)
#         val2 = spn2.evaluate(sess, sample)[0]
#         print("After copying")
#         print(val2)
#         assert val1 == val2
