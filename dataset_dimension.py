from imageio import get_reader
import numpy as np
import os

d = '../Data/DIV2K_TIFF/DIV2K_train_HR'

img_dims = []

for img_file in os.listdir(d):
    img_file = os.path.join(d, img_file)
    img_reader = get_reader(img_file)
    metadata = img_reader.get_meta_data()
    s = metadata['description']
    dim = [int(d) for d in s.split('[')[1].split(']')[0].split(',')]
    img_dims.append(dim)
    img_reader.close()

img_dims = np.array(img_dims)

print(type(img_dims))

