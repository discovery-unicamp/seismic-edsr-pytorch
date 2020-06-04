#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np

from imageio import imread
from syntheticdatagen import data_viewer


# Lê argumentos de entrada
parser = argparse.ArgumentParser(description='Reads parameters from JSON file, gerates data and saves them.')
parser.add_argument('data_path', type=str, 
        help='Path to the data to be viewed.')
args = parser.parse_args()

data = imread(args.data_path)
data_viewer(data)
