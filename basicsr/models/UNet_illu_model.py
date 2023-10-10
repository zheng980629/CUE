import torch
from torch import nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class UNetIlluBilateralFilterModel(BaseModel):
    """Base SR model for single image super-resolution."""
    def __init__(self, opt):
        super(UNetIlluBilateralFilterModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define loss
        if train_opt.get('forward'):
            self.cri_forward = build_loss(train_opt['forward']).to(self.device)
        else:
            self.cri_forward = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq1 = data['lq1'].to(self.device)
        self.lq2 = data['lq2'].to(self.device)
        self.mask_disruption = data['mask_disruption'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt1 = data['gt1'].to(self.device)
            self.gt2 = data['gt2'].to(self.device)
        # if 'mask' in data:
        #     self.mask = data['mask'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output = self.net_g(x=self.lq)
        self.output1 = self.net_g(x=self.lq1)
        self.output2 = self.net_g(x=self.lq2)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_forward:
            l_forw_pix = self.cri_forward(self.gt, self.output, self.mask_disruption)
            l_forw_pix1 = self.cri_forward(self.gt1, self.output1, self.mask_disruption)
            l_forw_pix2 = self.cri_forward(self.gt2, self.output2, self.mask_disruption)
            l_total += l_forw_pix + l_forw_pix1 + l_forw_pix2
            loss_dict['l_forw_pix'] = l_forw_pix
            loss_dict['l_forw_pix1'] = l_forw_pix1
            loss_dict['l_forw_pix2'] = l_forw_pix2

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.fake_H = self.net_g_ema(x=self.lq)
                self.fake_H1 = self.net_g_ema(x=self.lq1)
                self.fake_H2 = self.net_g_ema(x=self.lq2)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.fake_H = self.net_g(x=self.lq)
                self.fake_H1 = self.net_g(x=self.lq1)
                self.fake_H2 = self.net_g(x=self.lq2)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
            metric_data_rev = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # [0-255]
            low_img = tensor2img([visuals['low']])
            enhanced_img = tensor2img(visuals['enhanced'])
            normal_img = tensor2img(visuals['gt'])
            masked_gt_img = tensor2img(visuals['masked_gt'])

            enhanced_img1 = tensor2img(visuals['enhanced1'])
            normal_img1 = tensor2img(visuals['gt1'])

            enhanced_img2 = tensor2img(visuals['enhanced2'])
            normal_img2 = tensor2img(visuals['gt2'])

            metric_data['img'] = enhanced_img
            if 'gt' in visuals:
                metric_data['img2'] = normal_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.fake_H
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_low.png')
                    save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced.png')
                    save_img_path_normal = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                    save_img_path_enhanced1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced1.png')
                    save_img_path_normal1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt1.png')
                    save_img_path_enhanced2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced2.png')
                    save_img_path_normal2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt2.png')
                    save_img_path_masked_gt = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_masked_gt.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_low.png')
                        save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhanced.png')
                        save_img_path_normal = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                        save_img_path_enhanced1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced1.png')
                        save_img_path_normal1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt1.png')
                        save_img_path_enhanced2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced2.png')
                        save_img_path_normal2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt2.png')
                        save_img_path_masked_gt = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_masked_gt.png')
                    else:
                        save_img_path_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_low.png')
                        save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhanced.png')
                        save_img_path_normal = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                        save_img_path_enhanced1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced1.png')
                        save_img_path_normal1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt1.png')
                        save_img_path_enhanced2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced2.png')
                        save_img_path_normal2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt2.png')
                        save_img_path_masked_gt = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_masked_gt.png')

                imwrite(low_img, save_img_path_low)
                imwrite(enhanced_img, save_img_path_enhanced)
                imwrite(normal_img, save_img_path_normal)
                imwrite(enhanced_img1, save_img_path_enhanced1)
                imwrite(normal_img1, save_img_path_normal1)
                imwrite(enhanced_img2, save_img_path_enhanced2)
                imwrite(normal_img2, save_img_path_normal2)
                # imwrite(masked_gt_img, save_img_path_masked_gt)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    # def window_partition(self, x, win_size):
    #     x = x.permute(0, 2, 3, 1)
    #     B, H, W, C = x.shape
    #     x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    #     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    #     return windows

    # def window_reverse(self, windows, win_size, H, W):
    #     # B' ,Wh ,Ww ,C
    #     B = int(windows.shape[0] / (H * W / win_size / win_size))
    #     x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    #     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    #     return x

    def window_partition(self, x, win_size=4):
        x = x.permute(1, 2, 0)
        H, W, C = x.shape
        x = x.view(H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C  B'= H * W / win_size / win_size
        return windows


    def window_reverse(self, windows, win_size, H, W):
        # B' ,Wh ,Ww ,C
        x = windows.view(H // win_size, W // win_size, win_size, win_size, -1)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(H, W, -1)
        x = x.permute(2, 0, 1)
        return x

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low'] = self.lq.detach()[0].float().cpu()
        out_dict['enhanced'] = self.fake_H.detach().cpu()
        out_dict['gt'] = self.gt.detach()[0].float().cpu()
        out_dict['enhanced1'] = self.fake_H1.detach().cpu()
        out_dict['gt1'] = self.gt1.detach()[0].float().cpu()
        out_dict['enhanced2'] = self.fake_H2.detach().cpu()
        out_dict['gt2'] = self.gt2.detach()[0].float().cpu()

        H, W = self.gt.shape[-2:]
        masked_gt = self.gt.detach()[0]
        masked_gt = self.window_partition(masked_gt, self.opt['datasets']['val']['win_size_test'])
        masked_gt[self.mask_disruption, :, :, :] = 0
        masked_gt = self.window_reverse(masked_gt, self.opt['datasets']['val']['win_size_test'], H, W)
        out_dict['masked_gt'] = masked_gt.detach().float().cpu()

        masked_gt1 = self.gt1.detach()[0]
        masked_gt1 = self.window_partition(masked_gt1, self.opt['datasets']['val']['win_size_test'])
        masked_gt1[self.mask_disruption, :, :, :] = 0
        masked_gt1 = self.window_reverse(masked_gt1, self.opt['datasets']['val']['win_size_test'], H, W)
        out_dict['masked_gt1'] = masked_gt1.detach().float().cpu()

        masked_gt2 = self.gt2.detach()[0]
        masked_gt2 = self.window_partition(masked_gt2, self.opt['datasets']['val']['win_size_test'])
        masked_gt2[self.mask_disruption, :, :, :] = 0
        masked_gt2 = self.window_reverse(masked_gt2, self.opt['datasets']['val']['win_size_test'], H, W)
        out_dict['masked_gt2'] = masked_gt2.detach().float().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

