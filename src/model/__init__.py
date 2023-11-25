import gc
import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model {}...'.format(args.model))

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        # Chop params
        self.chop = args.chop
        self.max_img_area = 10**9
        self.shave = 20
        self.max_slices = args.max_chop_slices

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def _get_chops(self, x, num_slices):
        shave = self.shave
        height, width = x.size()[-2:]
        h_chop, w_chop = height//num_slices, width//num_slices
        x_chop = list()
        for i in range(num_slices):
            row_begin = i*h_chop - shave if i > 0 else 0
            row_end = (i+1)*h_chop + shave if i < (num_slices - 1) else height
            for j in range(num_slices):
                col_begin = j*w_chop - shave if j > 0 else 0
                col_end = (j+1)*w_chop + shave if j < (num_slices - 1) else width
                x_chop.append(x[...,row_begin:row_end,col_begin:col_end])
        return x_chop

    def _assemble_chops(self, y_chops, x_height, x_width):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        shave = self.shave * scale
        num_slices = int(len(y_chops)**0.5)
        y_height, y_width = x_height * scale, x_width * scale
        dims = y_chops[0].size()[:-2]
        y = y_chops[0].new(*dims, y_height, y_width)
        row_begin = 0
        for i in range(num_slices):
            col_begin = 0
            for j in range(num_slices):
                shave_top = shave if i > 0 else 0
                shave_bottom = shave if i < (num_slices - 1) else 0
                shave_left = shave if j > 0 else 0
                shave_right = shave if j < (num_slices - 1) else 0
                y_chop = y_chops[i*num_slices + j]
                h_chop, w_chop = y_chop.size()[-2:]
                h_chop -= (shave_top + shave_bottom)
                w_chop -= (shave_left + shave_right)
                row_end = row_begin + h_chop
                col_end = col_begin + w_chop
                y[...,row_begin:row_end,col_begin:col_end] = y_chop[...,shave_top:shave_top+h_chop,shave_left:shave_left+w_chop]
                col_begin = col_end
            row_begin = row_end
        return y

    def _clean_memory(self):
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad
        gc.collect()
        torch.cuda.empty_cache()
        return

    def forward_chop(self, x, num_slices=2):
        h, w = x.size()[-2:]
        img_area = h * w

        #print(torch.cuda.memory_summary())
        # Tries forward without chop first
        if img_area < self.max_img_area:
            oom = False
            try:
                y = self.model(x)
            except Exception as e:
                if not e.__class__ == torch.cuda.OutOfMemoryError:
                    raise e
                else:
                    oom = True
            if oom:
                self.max_img_area = img_area
                self._clean_memory()
                y = self.forward_chop(x, num_slices)
        else:
            oom = False
            try:
                x_chops = self._get_chops(x, num_slices)
                y_chops = [self.model(_x) for _x in x_chops]
                y = self._assemble_chops(y_chops, h, w)
            except Exception as e:
                if not e.__class__ == torch.cuda.OutOfMemoryError:
                    raise e
                if num_slices >= self.max_slices:
                    print(f"Forward chop reached maximum number of slices {self.max_slices} and image is still too big.")
                    raise e
                else:
                    oom = True
            if oom:
                self._clean_memory()
                y = self.forward_chop(x, num_slices+1)
        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
