#!/bin/bash

full_train_star="./ispn_cfg/full_train_star.yaml"
full_test_star="./ispn_cfg/full_test_star.yaml"
four_node_train_pair="./ispn_cfg/4_node_train_pair.yaml"
four_node_test_pair="./ispn_cfg/4_node_test_pair.yaml"

echo -e "==== -_-||| ===="

# --------- 4 node graphs ---------- #
# correct: (0.0004, 0.00065)  incorrect: (0.17, 0.22)  -- 0.99
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif -t 4nodePairP5Noise0Stockholm

# # correct: (0.002, 0.008)   incorrect: (0.22, 0.40)    -- 0.91
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t 4nodePairP5Noise1Stockholm

# # correct: (0.014, 0.029)  incorrect: (0.22, 0.35)     -- 0.72
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)" -t 4nodePairP5Noise2Stockholm

# # correct: (0.04, 0.09001)  incorrect: (0.22, 0.315)   -- 0.43
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)" -t 4nodePairP5Noise3Stockholm

# # correct: (0.07, 0.13)  incorrect: (0.17, 0.22)       -- 0.31
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t 4nodePairP5Noise4Stockholm

# # correct: (0.17, 0.234)  incorrect: (0.13, 0.155)     -- 0.15
# ./test_commander.py st "$four_node_train_pair" "$four_node_test_pair" -e NoisificationExperiment -N 60 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t 4nodePairP5Noise5Stockholm

# --------- full graph ---------- #

# # 0.0005, 0.0006, uniform
# (Finished) ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarP5Noise0Stockholm 
# (Finished) ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarP5Noise0Freiburg
# (Finished)  ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarP5Noise0Saarbrucken

# # 0.0005, 0.001
# (Finished) ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarP5Noise1Stockholm
# (Finished) ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarP5Noise1Freiburg
# (Finished) ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarP5Noise1Saarbrucken

# # 0.0005, 0.04
# (Finished) ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarP5Noise2Stockholm
# (Finished) ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarP5Noise2Freiburg
# (Finished) ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarP5Noise2Saarbrucken
# # 0.0005, 0.12
# (Finished) ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarP5Noise3Stockholm
# (Finished) ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarP5Noise3Freiburg
# (Finished) ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarP5Noise3Saarbrucken

# # correct: (0.07, 0.13)  incorrect: (0.17, 0.22) 
# ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarP5Noise4Stockholm
# ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarP5Noise4Freiburg
# ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarP5Noise4Saarbrucken

# # correct: (0.17, 0.234)  incorrect: (0.13, 0.155)
# ./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarP5Noise5Stockholm
# ./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarP5Noise5Freiburg
# ./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarP5Noise5Saarbrucken

# --------- segmented full graph ---------- #
# 0.0005, 0.0006, uniform
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarSegP5Noise0Stockholm 
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarSegP5Noise0Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.0006)" -unif -t fullStarSegP5Noise0Saarbrucken

# 0.0005, 0.001
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarSegP5Noise1Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarSegP5Noise1Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.001)" -t fullStarSegP5Noise1Saarbrucken

# 0.0005, 0.04
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarSegP5Noise2Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarSegP5Noise2Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.04)" -t fullStarSegP5Noise2Saarbrucken
# 0.0005, 0.12
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarSegP5Noise3Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarSegP5Noise3Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi "(0.5, 0.7)" -lo "(0.0005, 0.12)" -t fullStarSegP5Noise3Saarbrucken

# correct: (0.07, 0.13)  incorrect: (0.17, 0.22)
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarSegP5Noise4Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarSegP5Noise4Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t fullStarSegP5Noise4Saarbrucken

# correct: (0.17, 0.234)  incorrect: (0.13, 0.155)
./test_commander.py st "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarSegP5Noise5Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarSegP5Noise5Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e NoisificationExperiment -N 5 -n 3 -seg -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)" -t fullStarSegP5Noise5Saarbrucken


echo -e "==== ^o^ ===="
