import os
from data import srdata
import numpy as np
import imageio

def _get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip],
        *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret

class Synthetic(srdata.SRData):
    def __init__(self, args, name='Synthetic', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Synthetic, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = np.expand_dims(imageio.imread(f_hr), axis=2)
        lr = np.expand_dims(imageio.imread(f_lr), axis=2)

        return lr, hr, filename

    def _scan(self):
        names_hr, names_lr = super(Synthetic, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Synthetic, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Synthetic_train_HR')
        self.dir_lr = os.path.join(self.apath, 'Synthetic_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.tiff', '.tiff')
