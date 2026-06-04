#!/usr/bin/env python3
# coding: utf-8

# This script will unshuffle the NAMSS dataset by removing the number preffix in each file name
import os

DATASET = 'NAMSS'
HR_dirs = ['NAMSS_train_HR', 'NAMSS_test_HR', 'NAMSS_valid_HR']
HR_dirs = [os.path.join(DATASET, d) for d in HR_dirs]

for HR_dir in HR_dirs:
    HR_files = sorted(os.listdir(HR_dir))
    
    if HR_files[0][:4] != '0001':
        import sys
        sys.exit(f"{HR_dir} is not shuffled, aborting.")

    for shuffled_file in HR_files:
        original_file = '.'.join(shuffled_file.split('.')[1:])
        shuffled_file = os.path.join(HR_dir, shuffled_file)
        original_file = os.path.join(HR_dir, original_file)
        os.rename(shuffled_file, original_file)

print("If necessary, re-generate the Low Resolution images so that the file names are matching")
