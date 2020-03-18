import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

from inspector import *
import sys

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def inspect(obj, obj_type='class', exit=True):
    if obj_type == 'class':
        inspect_class(obj)
    else:
        inspect_module(obj)
    if exit:
        sys.exit(0)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)

            # from importlib import import_module
            # for m in ['ddbpn', 'edsr', 'mdsr', 'rcan', 'rdn', 'vdsr']:
            #     module = import_module('model.' + m.lower())
            #     print("Analizing model", m)
            #     inspect(module, obj_type='module', exit=False)
            #     print()
            # sys.exit(0)

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # inspect(trainer, obj_type='module')

            _model = model.Model(args, checkpoint)
            t = Trainer(args, loader, _model, _loss, checkpoint)
            inspect(t, obj_type='class')
            # while not t.terminate():
            #     t.train()
            #     t.test()
            #
            # checkpoint.done()

if __name__ == '__main__':
    main()
