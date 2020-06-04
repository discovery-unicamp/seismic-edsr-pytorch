#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import argparse
from syntheticdatagen import generate_parameters

np.random.seed(42)

# Lê argumentos de entrada
parser = argparse.ArgumentParser(description='Generate randomized parameters for the Data Generator and saves it as JSON')
parser.add_argument('--n_samples', type=int, default=5,
                    help='Number of data parameters to generate.')
parser.add_argument('--json_file', type=str, default='parameters.json',
                    help='File name to save the generated parameters.')
parser.add_argument('--file_mode', type=str, default='w',
                    help="Default 'w'. For appending instead of overwritting, use 'a'.")

args = parser.parse_args()

# Gera amostras aleatórias
parameter_list = []
for i in range(args.n_samples):
    parameter_list.append(generate_parameters(i))

# Salva dicionário como arquivo JSON
with open(args.json_file, args.file_mode) as f:
    json.dump(parameter_list, f)

print(args.n_samples, "parâmetros salvos em", args.json_file)

