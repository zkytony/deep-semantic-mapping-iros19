#!/bin/bash

full_train_star="./ispn_cfg/full_train_star.yaml"
full_test_star="./ispn_cfg/full_test_star.yaml"
four_node_train_pair="./ispn_cfg/4_node_train_pair.yaml"
four_node_test_pair="./ispn_cfg/4_node_test_pair.yaml"
full_train_edgerel="./ispn_cfg/full_train_edgerel.yaml"
full_test_edgerel="./ispn_cfg/full_test_edgerel.yaml"

echo -e "==== -_-||| ===="

# correct: (0.0004, 0.00065)  incorrect: (0.17, 0.22)  -- 0.99 
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif -t fullEdgeRelP5Noise0Stockholm 
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif -t fullEdgeRelP5Noise0Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif -t fullEdgeRelP5Noise0Saarbrucken

# correct: (0.002, 0.008)   incorrect: (0.22, 0.40)  -- 0.91
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t fullEdgeRelP5Noise1Stockholm
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t fullEdgeRelP5Noise1Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t fullEdgeRelP5Noise1Saarbrucken

# correct: (0.014, 0.029)  incorrect: (0.22, 0.35)   -- 0.72
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)" -t fullEdgeRelP5Noise2Stockholm
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)" -t fullEdgeRelP5Noise2Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)" -t fullEdgeRelP5Noise2Saarbrucken

# correct: (0.04, 0.09001)  incorrect: (0.22, 0.315)   -- 0.43
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)" -t fullEdgeRelP5Noise2Stockholm
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)" -t fullEdgeRelP5Noise2Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)" -t fullEdgeRelP5Noise2Saarbrucken

# correct: (0.07, 0.13)  incorrect: (0.17, 0.22)       -- 0.31
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullEdgeRelP5Noise3Stockholm
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullEdgeRelP5Noise3Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullEdgeRelP5Noise3Saarbrucken

# correct: (0.17, 0.234)  incorrect: (0.13, 0.155)     -- 0.15
cuda [0] ./train_test_graphspn.py synthetic st "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullEdgeRelP5Noise4Stockholm
cuda [0] ./train_test_graphspn.py synthetic fr "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullEdgeRelP5Noise4Freiburg
cuda [0] ./train_test_graphspn.py synthetic sa "$full_train_edgerel" "$full_test_edgerel" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullEdgeRelP5Noise4Saarbrucken

echo -e "==== ^o^ ===="
