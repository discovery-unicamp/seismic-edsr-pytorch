#!/usr/bin/env python3
# coding: utf-8

import argparse
import glob
import os
import sys

from imageio import imsave, imread
from multiprocessing import Pool
from skimage.transform import rescale
from time import time, sleep
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Bicubic upsampling of LR images to their original dimensions.')
    parser.add_argument('dataset', type=str, 
            help='Dataset dir.')
    parser.add_argument('--split', type=str, default='valid', 
            help='Dataset split. Default: valid')
    parser.add_argument('--save_dir', type=str, default='../experiment-01',
            help='Dir to save the result. Default:../edsr-pytorch/experiment-01/BICUBIC-X<scale>-Seed0000/results-<dataset>')
    parser.add_argument('--scale', type=int, default=2,
            help="Upsampling scale.")

    args = parser.parse_args()

    return args


# Read HR dirs
args = get_args()
if args.split not in ['train', 'valid', 'test']:
    import sys
    sys.exit(f"--split option should be one of 'train', 'valid' or 'test' but it was {args.split}. Aborting.")

search_pattern = f"{args.dataset}/*{args.split}*LR_bicubic*/X{args.scale}"
LR_dirs = glob.glob(search_pattern)
if not LR_dirs:
    import sys
    sys.exit(f"No match for {search_pattern}. Aborting.")

start_time = time()
benchmark = '-benchmark' if args.split == 'test' else ''
task = 'DIV2K_TIFF' if args.dataset not in ['DIV2K_TIFF', 'NAMSS'] else args.dataset
for LR_dir in LR_dirs:
    LR_files = os.listdir(LR_dir)

    UP_dir = os.path.join(
            args.save_dir,
            f"BICUBIC-X{args.scale}-{task}-Seed0000",
            f"results-{args.dataset}{benchmark}"
            )
    os.makedirs(UP_dir, exist_ok=True)

    def upsampler(LR_file):
        base, ext = os.path.splitext(LR_file)
        base = base.replace(f"x{args.scale}", '')
        UP_file = f"{base}_x{args.scale}_SR{ext}"
        UP_file = os.path.join(UP_dir, UP_file)
        LR_file = os.path.join(LR_dir, LR_file)
        img_LR = imread(LR_file)
        img_UP = rescale(img_LR, args.scale, anti_aliasing=True)
        imsave(UP_file, img_UP)

    n_proc = min(os.cpu_count(), len(LR_files))
    print("Upsampling", len(LR_files), "files from", LR_dir, "in", n_proc, "parallel processes.\n")
    with Pool(n_proc) as p:
        # List is necessary to make tqdm iterate
        list(tqdm(p.imap(upsampler, LR_files), total=len(LR_files)))

print("\nFiles saved in", UP_dir)
print("\nTotal time: {:.3f}s\n".format(time()-start_time))
