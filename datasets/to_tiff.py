#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import imghdr
import numpy as np
import os
import sys

from imageio import imread, imwrite
from multiprocessing import Pool
from time import time
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()

    # Mandatory argument
    parser.add_argument("input_folder", help=("Top level folder "
                        "containing the images to be converted."))

    # Optional arguments
    parser.add_argument("--output_folder", default='./DIV2K_TIFF',
                        help=("Folder where to write "
                        "the converted images. Default: DIV2K_TIFF."))

    parser.add_argument("--exclude", default=None,
                        help=("Excludes directories containing patterns. "
                        "Separate multiple patterns by single comma. "
                        "Ex: --exclude=unknown,X3"
                        "Default: None."))

    parser.add_argument("--ext", default='tiff',
                        help=("Extension to save the converted image file. "
                        "Default: tiff."))

    parser.add_argument("--dry", action='store_true',
                        help=("Don't do anything, just print what would happen."))

    parser.add_argument('--n_proc', type=int, default=None,
            help='Number of processes to parallelize if not specified, uses CPU count.')

    parser.add_argument('--clip', type=float, default=1,
            help='Value between 0 and 1 to clip the image.')

    args =  parser.parse_args()
    args.n_proc = args.n_proc if args.n_proc is not None else os.cpu_count()
    if args.clip <= 0 or args.clip > 1:
        import sys
        sys.exit("--clip should be in the interval (0,1]")

    return args


def rgb_to_gray(rgb_img):
    gray_coefs = np.array([0.2125, 0.7154, 0.0721])

    gray_img = rgb_img @ gray_coefs.T
    gray_img = (gray_img / 127.5) - 1.

    return gray_img.squeeze()


if __name__ == '__main__':
    args = get_arguments()
    print(f"Executing in {args.n_proc} parallel processes.\n")

    exclude_patterns = []
    if args.exclude is not None:
        exclude_patterns = args.exclude.split(',')

    start_time = time()
    for dirpath, subdirs, files in os.walk(args.input_folder):
        for exclude in exclude_patterns:
            for d in subdirs:
                if exclude in d:
                    subdirs.remove(d)

        # Get path without the top (input folder)
        subpath = os.path.normpath(dirpath).split(os.sep)[1:]
        output_path = os.path.join(args.output_folder, *subpath)
        if not args.dry:
            os.makedirs(output_path, exist_ok=True)

        def convert_file(fname):
            read_fname = os.path.join(dirpath, fname)
            if imghdr.what(read_fname) is None:
                return
            # Remove previous and add new extension
            fname = os.path.splitext(fname)[0]
            fname = fname + '.' + args.ext
            write_fname = os.path.join(output_path, fname)
            if not args.dry:
                img = imread(read_fname)
                if img.ndim == 3:
                    img = rgb_to_gray(img)
                if args.clip < 1.0:
                    vmax = np.abs([img.min(), img.max()]).max() * args.clip
                    img = np.clip(img, -vmax, vmax)
                img = img.astype(np.float32)
                imwrite(write_fname, img)

        if files:
            print("Converting", len(files), "files from", dirpath, f"to .{args.ext}")
            with Pool(args.n_proc) as p:
                # List is necessary to make tqdm iterate
                list(tqdm(p.imap(convert_file, files), total=len(files)))
            print()

    print("Total time: {:.3f}s\n".format(time()-start_time))
