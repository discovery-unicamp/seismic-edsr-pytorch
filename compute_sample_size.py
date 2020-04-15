import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train set size from parameters')

parser.add_argument('images_dim_file', 
                    help='File name with list of image dimensions')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')

args = parser.parse_args()

train_range, test_range = [[int(t) for t in g.split('-')] for g in args.data_range.split('/')]
train_range[0] -= 1
test_range[0] -= 1
train_range = np.arange(*train_range)
test_range = np.arange(*test_range)

# Load image dimensions
img_dims = np.loadtxt(args.images_dim_file, delimiter=',')

# Assume side-by-side windows per image (grid)
train_samples = np.floor(img_dims[train_range] / args.patch_size)
train_samples = train_samples.prod(axis=1).sum()
test_samples = np.floor(img_dims[test_range] / args.patch_size)
test_samples = test_samples.prod(axis=1).sum()

print("Total train samples:", train_samples)
print("Total test samples:", test_samples)
print("Samples per epoch:", args.batch_size * args.test_every)
