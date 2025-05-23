#!/usr/bin/env bash
# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz
gunzip ./*.gz