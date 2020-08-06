import argparse
import os
import numpy as np
from imageio import get_reader

parser = argparse.ArgumentParser(description='Train set size from parameters')

parser.add_argument('HR_images_dir', 
                    help='Directory path to HR images')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--scale', type=int, default=2,
                    help='Super-Resolution scale')
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
img_dims = []
d = args.HR_images_dir
img_files = os.listdir(d)
img_files.sort()

for img_file in img_files:
    img_file = os.path.join(d, img_file)
    img_reader = get_reader(img_file)
    metadata = img_reader.get_meta_data()
    s = metadata['description']
    dim = [int(d) for d in s.split('[')[1].split(']')[0].split(',')]
    img_dims.append(dim[::-1])
    img_reader.close()

img_dims = np.array(img_dims)

# Assume side-by-side windows per image (grid)
train_samples = np.floor(img_dims[train_range] / args.patch_size)
train_samples = train_samples.prod(axis=1).sum()
test_samples = np.floor(img_dims[test_range] / args.patch_size)
test_samples = test_samples.prod(axis=1).sum()
samples_per_epoch = args.batch_size * args.test_every

print("Output patch:", args.patch_size)
print("Input patch:", args.patch_size // args.scale)
print("Total train samples:", train_samples)
print("Total test samples:", test_samples)
print("Samples per epoch:", samples_per_epoch)
print("Train set percentage per epoch: {:.2f}".format((samples_per_epoch/train_samples)*100))
print("Test to train ratio: {:.2f}".format((test_samples/train_samples)*100))
print("Test to epoch ratio: {:.2f}".format((test_samples/samples_per_epoch)*100))
