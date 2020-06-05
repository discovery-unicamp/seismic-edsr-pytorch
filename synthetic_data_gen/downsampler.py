#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

from imageio import imsave, imread
from multiprocessing import Pool
from skimage.transform import rescale
from time import time


# Lê argumentos de entrada
parser = argparse.ArgumentParser(description='Converts HR to LR images in different scales.')
parser.add_argument('dataset', type=str, 
        help='Dataset dir from which to read HR and savel LR images.')
parser.add_argument('--scales', type=str, default='2,4',
        help="Downsampling scales to generate, separated by comma. Default: '2,4'")
parser.add_argument('--n_proc', type=int, default=None,
        help='Number of processes to parallelize if not specified, uses CPU count.')

args = parser.parse_args()
args.scales = [int(s.strip()) for s in args.scales.split(',')]

# Lê diretório HR
HR_dir = None
for d in os.listdir(args.dataset):
    if 'HR' in d:
        if HR_dir is None:
            HR_dir = d
        else:
            sys.exit("Erro: mais de um diretório de HR")
HR_dir = os.path.join(args.dataset, HR_dir)
LR_base = HR_dir.replace('HR', 'LR_bicubic')
#print("Diretório de alta resolução:", HR_dir)

# Cria diretórios de baixa resolução
LR_dirs = {} 
for scale in args.scales:
    LR_dir = os.path.join(LR_base, 'X{}'.format(scale))
    os.makedirs(LR_dir, exist_ok=True)
    LR_dirs[scale] = LR_dir

# Fazer downsampling das imagens
def downsampler(HR_file):
    base, ext = os.path.splitext(HR_file)
    base = base + 'x{}'
    # Lê imagem de alta
    HR_file = os.path.join(HR_dir, HR_file)
    img_high = imread(HR_file)
    for scale in args.scales:
        LR_file = base.format(scale) + ext
        LR_file = os.path.join(LR_dirs[scale], LR_file)
        img_low = rescale(img_high, 1/scale, anti_aliasing=True)
        imsave(LR_file, img_low)
        print("Convertido:", LR_file)

start_time = time()
n_proc = args.n_proc if args.n_proc is not None else os.cpu_count()
print("\nParalelizando em {} processos\n.".format(n_proc))
with Pool(n_proc) as p:
    p.map(downsampler, os.listdir(HR_dir))
print("\nTempo para geração de dados: {:.3f}s\n".format(time()-start_time))
