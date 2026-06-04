#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys

from imageio import imsave, imread
from multiprocessing import Pool
from skimage.transform import rescale
from time import time, sleep
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Converts HR to LR images in different scales.')
    parser.add_argument('dataset', type=str, 
            help='Dataset dir from which to read HR and save LR images.')
    parser.add_argument('--scales', type=str, default='2,4',
            help="Downsampling scales to generate, separated by comma. Ex: --scales=2,4")
    parser.add_argument('--n_proc', type=int, default=None,
            help='Number of processes to parallelize if not specified, uses CPU count.')

    args = parser.parse_args()
    args.scales = [int(s.strip()) for s in args.scales.split(',')]

    args.n_proc = args.n_proc if args.n_proc is not None else os.cpu_count()

    return args


args = get_args()
print("Executing in {} parallel processes.\n".format(args.n_proc))

print("Downsampling factor of " + ', '.join([f'{s}x' for s in args.scales]) + ".\n")
sleep(2)

# Read HR dirs
HR_dirs = [d for d in os.listdir(args.dataset) if 'HR' in d]
HR_dirs = [os.path.join(args.dataset, d) for d in HR_dirs]

start_time = time()
for HR_dir in HR_dirs:
    HR_files = os.listdir(HR_dir)
    print("Converting", len(HR_files), "files from", HR_dir)

    # Create LR dirs
    LR_base = HR_dir.replace('HR', 'LR_bicubic')
    LR_dirs = {} 
    for scale in args.scales:
        LR_dir = os.path.join(LR_base, 'X{}'.format(scale))
        os.makedirs(LR_dir, exist_ok=True)
        LR_dirs[scale] = LR_dir

    def downsampler(HR_file):
        base, ext = os.path.splitext(HR_file)
        base = base + 'x{}'
        # Lê imagem de alta
        HR_file = os.path.join(HR_dir, HR_file)
        img_high = imread(HR_file)
        for scale in args.scales:
            LR_file = base.format(scale) + ext
            LR_file = os.path.join(LR_dirs[scale], LR_file)
            img_low = rescale(img_high, 1/scale, anti_aliasing=True, channel_axis=len(img_high.shape)-1)
            imsave(LR_file, img_low)

    with Pool(args.n_proc) as p:
        # List is necessary to make tqdm iterate
        list(tqdm(p.imap(downsampler, HR_files), total=len(HR_files)))

print("\nTotal time: {:.3f}s\n".format(time()-start_time))
