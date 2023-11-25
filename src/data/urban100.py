import os
from data import div2k_tiff
from data import common

class Urban100(div2k_tiff.DIV2K_TIFF):
    def __init__(self, args, name='Urban100', train=False, benchmark=True):
        self.no_augment = True
        super(Urban100, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(div2k_tiff.DIV2K_TIFF, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Urban100_test_HR')
        self.dir_lr = os.path.join(self.apath, 'Urban100_test_LR_bicubic')
        self.ext = ('.tiff', '.tiff')
