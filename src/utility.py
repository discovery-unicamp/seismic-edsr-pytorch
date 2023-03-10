import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from data import common

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)

        if args.reset and os.path.exists(self.dir):
            if input("--reset option will erase {}. Continue? (y/N): ".
                    format(self.dir)).lower() == 'y':
                os.system('rm -rf ' + self.dir)
                args.load = ''
            else:
                import sys
                print("Terminating.")
                sys.exit()

        if os.path.exists(self.dir) and args.load:
            self.log = torch.load(self.get_path('psnr_log.pt'))
            print('Continue from epoch {}...'.format(len(self.log)+1))
        else:
            args.resume = 0
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

        # TensorBoard
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            import tensorboard

            # Check tensorboard version
            tb_version = tensorboard.__version__
            if int(tb_version.split('.')[0]) < 2:
                print("Requires TensorBoard version >= 2.X but has version",
                        tb_version)
                print("You can unset the --tensorboard option. Exiting.")
                sys.exit()

            # Launch TensorBoard and create writer
            tb_path = self.get_path(args.tensorboard_dir)
            tb = tensorboard.program.TensorBoard()
            tb.configure(argv=[None, '--logdir', tb_path, 
                '--host', args.tensorboard_host,
                '--port', args.tensorboard_port])
            url = tb.launch()
            print("TensorBoard launched at", url)
                      
            # If resuming, throw out intermediate steps between epochs
            if args.resume:
                purge_step = len(self.log) * args.test_every
            else:
                purge_step = 0
            self.tensorboard_writer = SummaryWriter(tb_path, purge_step=purge_step)

    def tensorboard_log(self, tag, scalar_dic, global_step=None):
        if self.args.tensorboard:
            self.tensorboard_writer.add_scalars(tag, scalar_dic, global_step)

    def tensorboard_images(self, tag, img_list, global_step=None):
        # img_list must be a list of pairs [SR, LR]
        if self.args.tensorboard:
            nrow = len(img_list)
            ncol = len(img_list[0])

            fig, axes = plt.subplots(nrow, ncol, squeeze=False)

            # Set fig dimensions
            fig_width = max([pair[0].shape[3] for pair in img_list])
            fig_width = (ncol*fig_width) / fig.dpi
            fig_height = sum([pair[0].shape[2] for pair in img_list])
            fig_height /= fig.dpi
            fig.set_size_inches(fig_width, fig_height)

            for i in range(nrow):
                for j in range(ncol):
                    img = img_list[i][j].squeeze()
                    # If it's color image, make it channel last and in [0, 1] range
                    if img.dim() == 3:
                        img = img.permute(1, 2, 0)
                        img, = common.linear_shift(img,
                                in_range=self.args.tensor_range,
                                out_range=[0., 1.])
                    img = img.cpu()
                    ax = axes[i][j]
                    ax.imshow(img, vmin=self.args.tensor_range[0], 
                            vmax=self.args.tensor_range[1],
                            cmap=self.args.tensorboard_color,
                            interpolation='lanczos')
                    ax.set_axis_off()


            self.tensorboard_writer.add_figure(tag, fig, global_step)

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()
        if self.args.tensorboard and self.args.tensorboard_keep_alive:
            import signal
            import sys
            original_sigint = signal.getsignal(signal.SIGINT)

            def exit_gracefully(signum, frame):
                signal.signal(signal.SIGINT, original_sigint)

                try:
                    if input("\nDo you want to quit? (y/n)> ").lower().startswith('y'):
                        print("Bye bye!")
                        sys.exit(1)
                    else: print("Continue waiting. Press CTRL+C to quit.")
                except KeyboardInterrupt:
                    print("\nBye bye!")
                    sys.exit(1)

                # restore the exit gracefully handler here    
                signal.signal(signal.SIGINT, exit_gracefully)


            signal.signal(signal.SIGINT, exit_gracefully)
            print("TensorBoard is active. Press CTRL+C to end program.")
            while True: time.sleep(1)

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            ext = dataset.dataset.ext[1]
            for v, p in zip(save_list, postfix):
                if ext.find('.tif') >= 0:
                    normalized = to_output(v[0], 
                            input_range=self.args.input_range,
                            tensor_range=self.args.tensor_range)[0]
                    tensor_cpu = normalized.squeeze().cpu()
                else:
                    normalized = to_output(v[0], 
                            input_range=self.args.input_range,
                            tensor_range=self.args.tensor_range,
                            quantize=True)[0]
                    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}{}'.format(filename, p, ext), tensor_cpu))

def clip(img, clip_range):
    return img.clamp(clip_range[0], clip_range[1])

def to_output(*tensors, input_range, tensor_range, quantize=False):
    normalized = common.linear_shift(*tensors, 
            in_range=tensor_range,
            out_range=input_range)
    if quantize:
        normalized = [t.round() for t in normalized]
    return normalized

def calc_psnr(sr, hr, scale, dynamic_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / dynamic_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            # Coeficients to compute luminance from RGB
            gray_coeffs = [0.2125, 0.7154, 0.0721]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) 
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()


    return -10 * math.log10(mse)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    kwargs_scheduler = {'mode': 'max', 'factor': args.gamma,
            'patience': args.lr_patience, 'verbose': True, 'threshold': 0.0001,
            'threshold_mode': 'rel', 'cooldown': 0,
            'min_lr': args.lr * (args.gamma ** args.lr_max_updates),
            'eps': 1e-08}
    scheduler_class = lrs.ReduceLROnPlateau

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)
            self.scheduler.last_epoch = 0

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir, 'optimizer.pt'))
            torch.save(self.scheduler.state_dict(),
                    self.get_dir(save_dir, 'lr_scheduler.pt'))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir, 'optimizer.pt')))
            self.scheduler.load_state_dict(torch.load(
                self.get_dir(load_dir, 'lr_scheduler.pt')))

        def get_dir(self, dir_path, file_name):
            return os.path.join(dir_path, file_name)

        def schedule(self, metric):
            self.scheduler.step(metric)

        def get_lr(self):
            return self.param_groups[0]['lr']

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
