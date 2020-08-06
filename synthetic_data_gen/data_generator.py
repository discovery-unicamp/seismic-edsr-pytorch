#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import matplotlib.pyplot as plt
import os

from imageio import imsave
from multiprocessing import Pool
from time import time

from syntheticdatagen import *

def generate_data(params):
    seismic_data = makeReflectivityPanel(params=params)
    seismic_data = applyFolding(seismic_data, params=params)
    seismic_data = applyFault(seismic_data, params=params)
    seismic_data = removePadding(seismic_data, params=params)
    seismic_data = applyConvolution(seismic_data, params=params)
    seismic_data = normalize(seismic_data)

    seismic_data = np.squeeze(seismic_data)

    return seismic_data

def save_data(data, data_name, dtype=np.float32):
    data_path = os.path.join(args.save_dir, data_name)
    data = dtype(data)
    imsave(data_path, data)
    print(data_path, "gerado e salvo.")

def generate_and_save_data(params):
    data = generate_data(params)
    if np.equal(data, 0.0).min() == True:
        print("AVISO: dado {} só contém zeros".format(params['name']))
    save_data(data, params['name'])


# Lê argumentos de entrada
parser = argparse.ArgumentParser(description='Reads parameters from JSON file, gerates data and saves them.')
parser.add_argument('--json_file', type=str, default='parameters.json',
        help='File from which to read the parameters. Default: \'parameters.json\'')
parser.add_argument('--save_dir', type=str, default='../../Data/Synthetic/Synthetic_train_HR',
        help="Directory in which to save generated data. Default: '../../Data/Synthetic/Synthetic_train_HR'.")
parser.add_argument('--view_only', action='store_true',
        help="Instead of saving all data, randomly select a parameter to generate and plot it.")
parser.add_argument('--n_proc', type=int, default=None,
        help='Number of processes to parallelize if not specified, uses CPU count.')

args = parser.parse_args()

# Cria diretório, se necessário
if not args.view_only:
    os.makedirs(args.save_dir, exist_ok=True)

# Lê parâmetros
with open(args.json_file) as f:
    parameter_list = json.load(f)

# Se for só para visualizar, sorteia e plota dado e termina
if args.view_only:
    import sys

    rand_idx = np.random.randint(len(parameter_list))
    params = parameter_list[rand_idx]

    data = generate_data(params)

    print("Plotando dado", rand_idx+1, "de", len(parameter_list))
    data_viewer(data)

    sys.exit("Encerrando.")

#for p in parameter_list:
    #generate_and_save_data(p)

start_time = time()

print("\n{} dados para serem gerados.".format(len(parameter_list)))
n_proc = args.n_proc if args.n_proc is not None else os.cpu_count()
print("Paralelizando em {} processos\n.".format(n_proc))
with Pool(n_proc) as p:
    p.map(generate_and_save_data, parameter_list)
print("\nTempo para geração de dados: {:.3f}s\n".format(time()-start_time))
