#!/usr/bin/env python3
# coding: utf-8

# This script will shuffle the NAMSS dataset by renaming files in in random order.
# A seed is set for repeatability
import os
import random

DATASET = 'NAMSS'
SEED = 3309

random.seed(SEED)

HR_dirs = ['NAMSS_train_HR', 'NAMSS_test_HR', 'NAMSS_valid_HR']
HR_dirs = [os.path.join(DATASET, d) for d in HR_dirs]
for HR_dir in HR_dirs:
    HR_files = sorted(os.listdir(HR_dir))

    if HR_files[0][:4] == '0001':
        import sys
        sys.exit(f"{HR_dir} is already shuffled, aborting.")

    random.shuffle(HR_files)
    for i, old_file in enumerate(HR_files, start=1):
        new_file = f'{i:04d}.{old_file}'
        old_file = os.path.join(HR_dir, old_file)
        new_file = os.path.join(HR_dir, new_file)
        os.rename(old_file, new_file)

print("\nIf necessary, re-generate the Low Resolution images so that the file names are matching.\n")
