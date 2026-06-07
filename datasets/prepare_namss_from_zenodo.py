#!/usr/bin/env python3
# coding: utf-8

"""
Prepares the Unicamp-NAMSS dataset from Zenodo for Super-Resolution experiments.
Reads extracted train/validation/test directories, trims images so dimensions are
divisible by 12, splits wide images (>2000 px), and saves to NAMSS/NAMSS_<split>_HR/.
"""

import argparse
import os

from imageio import imread, imsave
from multiprocessing import Pool
from tqdm import tqdm

DIVISIBLE_BY = 12
SMALLEST_IMG_SIZE = 192
MAX_IMG_WIDTH = 2000

SPLITS = {
    'train': 'NAMSS_train_HR',
    'validation': 'NAMSS_valid_HR',
    'test': 'NAMSS_test_HR',
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Prepare Unicamp-NAMSS from Zenodo for SR experiments.')
    parser.add_argument('--source_dir', type=str, default='.',
        help='Directory containing extracted train/, validation/, test/ folders. Default: .')
    parser.add_argument('--target_dir', type=str, default='NAMSS',
        help='Output directory. Default: NAMSS')
    parser.add_argument('--n_proc', type=int, default=None,
        help='Number of parallel processes. Default: CPU count.')
    parser.add_argument('--delete_source', action='store_true',
        help='Delete each source file after processing to avoid duplicating ~30GB of data.')
    return parser.parse_args()


def process_file(args):
    fpath, save_dir, delete_source = args
    fname = os.path.basename(fpath)

    img = imread(fpath)
    if hasattr(img, 'meta'):
        img.meta.clear()

    row_begin = (img.shape[0] % DIVISIBLE_BY) // 2
    row_end = row_begin + (img.shape[0] // DIVISIBLE_BY) * DIVISIBLE_BY
    col_begin = (img.shape[1] % DIVISIBLE_BY) // 2
    col_end = col_begin + (img.shape[1] // DIVISIBLE_BY) * DIVISIBLE_BY
    new_height = row_end - row_begin
    new_width = col_end - col_begin

    if new_height < SMALLEST_IMG_SIZE or new_width < SMALLEST_IMG_SIZE:
        if delete_source:
            os.remove(fpath)
        return

    img = img[row_begin:row_end, col_begin:col_end]
    base, ext = os.path.splitext(fname)

    if new_width <= MAX_IMG_WIDTH:
        imsave(os.path.join(save_dir, fname), img)
    else:
        num_sub_imgs = new_width // MAX_IMG_WIDTH + 1
        sub_width = new_width // num_sub_imgs
        sub_width -= (sub_width % DIVISIBLE_BY)
        for i in range(num_sub_imgs - 1):
            begin = i * sub_width
            end = (i + 1) * sub_width
            sub_img = img[:, begin:end]
            imsave(os.path.join(save_dir, f"{base}.part{i+1:02d}{ext}"), sub_img)
        begin = (num_sub_imgs - 1) * sub_width
        sub_img = img[:, begin:]
        imsave(os.path.join(save_dir, f"{base}.part{num_sub_imgs:02d}{ext}"), sub_img)

    if delete_source:
        os.remove(fpath)


if __name__ == '__main__':
    args = get_args()
    n_proc = args.n_proc if args.n_proc else os.cpu_count()

    for split_name, target_name in SPLITS.items():
        source = os.path.join(args.source_dir, split_name)
        if not os.path.isdir(source):
            print(f"Skipping {split_name}/ (not found)")
            continue

        save_dir = os.path.join(args.target_dir, target_name)
        os.makedirs(save_dir, exist_ok=True)

        files = sorted(os.listdir(source))
        tasks = [(os.path.join(source, f), save_dir, args.delete_source) for f in files]

        print(f"Processing {len(files)} files from {source} -> {save_dir}")
        with Pool(n_proc) as p:
            list(tqdm(p.imap(process_file, tasks), total=len(tasks)))

    print("\nDone.")
