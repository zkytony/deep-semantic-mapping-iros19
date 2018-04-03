#!/bin/bash

# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif
# read -p "Press a key to run Noise1"
# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)"
# read -p "Press a key to run Noise2"
# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)"
# read -p "Press a key to run Noise3"
# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)"
# read -p "Press a key to run Noise4"
# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)"
# read -p "Press a key to run Noise5"
# ./run_factor_graph_tests.py -N 60 -n 3 -l -t 4 -tr -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)"

# read -p "Press to run on full graph"
# ./run_factor_graph_tests.py -N 5 -n 3 -l  -hi-c "(0.5, 0.7)" -lo-c "(0.0004, 0.00065)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)" -unif
# read -p "Press a key to run Noise1"
# ./run_factor_graph_tests.py -N 5 -n 3 -l   -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)"
# read -p "Press a key to run Noise2"
# ./run_factor_graph_tests.py -N 5 -n 3 -l   -hi-c "(0.5, 0.7)" -lo-c "(0.014, 0.029)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.35)"
# read -p "Press a key to run Noise3"
# ./run_factor_graph_tests.py -N 5 -n 3 -l   -hi-c "(0.5, 0.7)" -lo-c "(0.04, 0.09001)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.315)"
# read -p "Press a key to run Noise4"
# ./run_factor_graph_tests.py -N 5 -n 3 -l   -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)"
# read -p "Press a key to run Noise5"
# ./run_factor_graph_tests.py -N 5 -n 3 -l    -hi-c "(0.5, 0.7)" -lo-c "(0.17, 0.234)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.13, 0.155)"


# read -p "[inferph] Press a key to run Noise1"
# ./run_factor_graph_tests.py -N 5 -n 3 --inferplaceholder-exp   -hi-c "(0.5, 0.7)" -lo-c "(0.002, 0.008)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.22, 0.40)"
# read -p "Press a key to run Noise4"
# ./run_factor_graph_tests.py -N 5 -n 3 --inferplaceholder-exp   -hi-c "(0.5, 0.7)" -lo-c "(0.07, 0.13)" -hi-ic "(0.5, 0.7)" -lo-ic "(0.17, 0.22)"


# Novelty
# Stockholm
# echo "---Stockholm---\n\n" >> MRF2_train.log
# ./run_one_seq.py st floor4_night_c -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py st floor7_night_c -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py st floor6_night_a1 -nov -n 1 >> MRF2_train.log; echo "."
# # Freiburg
# echo "---Freiburg---\n\n" >> MRF2_train.log
# ./run_one_seq.py fr seq2_night1 -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py fr seq1_cloudy3 -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py fr seq1_sunny2 -nov -n 1 >> MRF2_train.log; echo "."
# # Saarbrucken
# echo "---Saarbrucken---\n\n" >> MRF2_train.log
# ./run_one_seq.py sa seq4_night3 -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py sa seq3_night2 -nov -n 1 >> MRF2_train.log; echo "."
# ./run_one_seq.py sa seq1_night2 -nov -n 1 >> MRF2_train.log; echo "."

# echo "---Stockholm---\n\n" >> MRF2_test.log
# ./run_one_seq.py st floor4_night_c -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py st floor7_night_c -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py st floor6_night_a1 -nov -n 1 >> MRF2_test.log; echo "."
# # Freiburg
# echo "---Freiburg---\n\n" >> MRF2_test.log
# ./run_one_seq.py fr seq2_night1 -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py fr seq1_cloudy3 -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py fr seq1_sunny2 -nov -n 1 >> MRF2_test.log; echo "."
# # Saarbrucken
# echo "---Saarbrucken---\n\n" >> MRF2_test.log
# ./run_one_seq.py sa seq4_night3 -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py sa seq3_night2 -nov -n 1 >> MRF2_test.log; echo "."
# ./run_one_seq.py sa seq1_night2 -nov -n 1 >> MRF2_test.log; echo "."

# #MRF3
# echo "---Stockholm---\n\n" >> MRF3_train.log
# ./run_one_seq.py st floor4_night_c  -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py st floor7_night_c  -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py st floor6_night_a1 -tr -nov -n 1 >> MRF3_train.log; echo "."
# # Freiburg
# echo "---Freiburg---\n\n" >> MRF3_train.log
# ./run_one_seq.py fr seq2_night1  -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py fr seq1_cloudy3 -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py fr seq1_sunny2  -tr -nov -n 1 >> MRF3_train.log; echo "."
# # Saarbrucken
# echo "---Saarbrucken---\n\n" >> MRF3_train.log
# ./run_one_seq.py sa seq4_night3 -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py sa seq3_night2 -tr -nov -n 1 >> MRF3_train.log; echo "."
# ./run_one_seq.py sa seq1_night2 -tr -nov -n 1 >> MRF3_train.log; echo "."


echo "---Stockholm---\n\n" >> MRF3_test.log
./run_one_seq.py st floor4_night_c  -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py st floor7_night_c  -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py st floor6_night_a1 -tr -nov -n 1 >> MRF3_test.log; echo "."
# Freiburg
echo "---Freiburg---\n\n" >> MRF3_test.log
./run_one_seq.py fr seq2_night1  -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py fr seq1_cloudy3 -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py fr seq1_sunny2  -tr -nov -n 1 >> MRF3_test.log; echo "."
# Saarbrucken
echo "---Saarbrucken---\n\n" >> MRF3_test.log
./run_one_seq.py sa seq4_night3 -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py sa seq3_night2 -tr -nov -n 1 >> MRF3_test.log; echo "."
./run_one_seq.py sa seq1_night2 -tr -nov -n 1 >> MRF3_test.log; echo "."
