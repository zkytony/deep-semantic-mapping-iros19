#!/bin/bash
# Infer placeholders tests

full_train_star="./ispn_cfg/full_train_star.yaml"
full_test_star="./ispn_cfg/full_test_star_infph.yaml"

echo -e "==== -_-||| ===="

# --------- full graph ---------- #

# correct: (0.002, 0.008)  incorrect: (0.22, 0.40)
./test_commander.py st "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t agfullStarInfPhP5Noise1Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t agfullStarInfPhP5Noise1Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)" -t agfullStarInfPhP5Noise1Saarbrucken

# correct: (0.07, 0.13)  incorrect: (0.17, 0.22)
./test_commander.py st "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t agfullStarInfPhP5Noise4Stockholm
./test_commander.py fr "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t agfullStarInfPhP5Noise4Freiburg
./test_commander.py sa "$full_train_star" "$full_test_star" -e InferPlaceholdersExperiment -N 5 -n 3 -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -t agfullStarInfPhP5Noise4Saarbrucken

echo -e "==== ^o^ ===="
