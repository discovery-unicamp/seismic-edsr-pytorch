import numpy as np
import os

from data import srdata
from data import common


class DIV2K_TIFF(srdata.SRData):
    def __init__(self, args, name='DIV2K_TIFF', train=True, benchmark=False):
        self.cache_data = args.cache_data
        self.no_augment = args.no_augment
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        args.no_augment = False
        super(DIV2K_TIFF, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def get_patch(self, lr, hr):
        lr, hr = super().get_patch(lr, hr)
        if not self.no_augment: lr, hr = common.augment(lr, hr, hflip=True, rot=False)
        return lr, hr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        if self.cache_data:
            lr, hr, filename = common.load_and_cache_file(f_hr, f_lr)
        else:
            lr, hr, filename = common.load_file(f_hr, f_lr)

        return lr, hr, filename

    def _scan(self):
        names_hr, names_lr = super(DIV2K_TIFF, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K_TIFF, self)._set_filesystem(dir_data)
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        else:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.tiff', '.tiff')
