#!/bin/bash

# Different combos - but run on particular sequences

# Stockholm sequences
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d fr-st -t fullStarP5NoveltyStockholm -e NoveltyDetectionExperiments -i floor4_night_c
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d sa-st -t fullStarP5NoveltyStockholm -e NoveltyDetectionExperiments -i floor7_night_c 
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d fr-st -t fullStarP5NoveltyStockholm -e NoveltyDetectionExperiments -i floor6_night_a1

# Freiburg sequences
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d sa-fr -t fullStarP5NoveltyFreiburg -e NoveltyDetectionExperiments -i seq2_night1
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d st-fr -t fullStarP5NoveltyFreiburg -e NoveltyDetectionExperiments -i seq1_cloudy3
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d sa-fr -t fullStarP5NoveltyFreiburg -e NoveltyDetectionExperiments -i seq1_sunny2

# Saarbrucken sequences
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d st-sa -t fullStarP5NoveltySaarbrucken -e NoveltyDetectionExperiments -i seq4_night3
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d fr-sa -t fullStarP5NoveltySaarbrucken -e NoveltyDetectionExperiments -i seq3_night2
./test_instance_spn.py ispn_cfg/full_train_star_novelty.yaml ispn_cfg/full_test_star_novelty.yaml -N 1 -d st-sa -t fullStarP5NoveltySaarbrucken -e NoveltyDetectionExperiments -i seq1_night2

