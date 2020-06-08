import os
import math
from decimal import Decimal
from numpy import random

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.step = self.optimizer.get_last_epoch() * args.test_every
        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            self.step += 1
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

                # TensorBoard Log
                running_avg = {'train': self.loss.log[-1, -1]/(batch+1)}
                self.ckp.tensorboard_log('Loss/running_avg', running_avg, self.step)

                snapshot = {'train': loss.item()}
                self.ckp.tensorboard_log('Loss/snapshot', snapshot, self.step)

                lr_dic = {'lr': self.optimizer.get_lr()}
                self.ckp.tensorboard_log('Learning_Rate', lr_dic, self.step)

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)

                # TensorBoard vars
                best_psnr = 0 
                worst_psnr = 1e9
                avg_psnr = 0 

                max_loss = 0 
                min_loss = 1e9
                avg_loss = 0 

                # Draw image index to plot in TensorBoard
                idx_size = min(self.args.tensorboard_nimgs, len(d))
                tb_img_idxs = random.choice(len(d), size=idx_size, replace=False)
                tb_img_idxs.sort()
                tb_imgs = []

                for i, (lr, hr, filename) in enumerate(tqdm(d, ncols=80)):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.clip(sr, self.args.tensor_range)

                    save_list = [sr]
                    dynamic_range = (self.args.tensor_range[1] - 
                                    self.args.tensor_range[0])
                    psnr = utility.calc_psnr(
                        sr, hr, scale, dynamic_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += psnr

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    # TensorBoard update vars
                    if not self.args.test_only:
                        avg_psnr += psnr 
                        best_psnr = psnr if psnr > best_psnr else best_psnr 
                        worst_psnr = psnr if psnr < worst_psnr else worst_psnr 

                        loss = self.loss(sr, hr).item()
                        avg_loss += loss 
                        max_loss = loss if loss > max_loss else max_loss 
                        min_loss = loss if loss < min_loss else min_loss 
                    if i in tb_img_idxs:
                        tb_imgs.append([sr, hr])

                # EDSR log
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

                # TensorBoard log
                # Só salva imagem se é melhor época até agora
                epoch = self.optimizer.get_last_epoch()
                if best[1][0, 0] == epoch:
                    self.ckp.tensorboard_images('Test', tb_imgs, self.step)
                if not self.args.test_only:
                    loss_dic = {'avg': avg_loss/len(d),
                                'min': min_loss,
                                'max': max_loss} 
                    self.ckp.tensorboard_log('Loss/test', loss_dic, self.step)

                    psnr_dic = {'avg': avg_psnr/len(d),
                                'best': best_psnr,
                                'worst': worst_psnr} 
                    self.ckp.tensorboard_log('PSNR', psnr_dic, self.step)

        if not self.args.test_only:
            self.optimizer.schedule(avg_psnr/len(d))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            epoch = self.optimizer.get_last_epoch()
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
