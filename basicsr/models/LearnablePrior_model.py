from cgi import test
from basicsr.losses.losses import L_color
import torch
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
class LearnablePirorModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(LearnablePirorModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.net_noisePrior = build_network(opt['network_noisePrior'])
        self.net_noisePrior = self.model_to_device(self.net_noisePrior)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        
        load_path_noisePrior = self.opt['path'].get('pretrain_network_noisePrior', None)
        if load_path_noisePrior is not None:
            param_key = self.opt['path'].get('param_key_decom', 'params')
            self.load_network(self.net_noisePrior, load_path_noisePrior, self.opt['path'].get('strict_load_noisePrior', True), param_key)

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

        if train_opt.get('gtRecon_opt'):
            self.cri_gtRecon = build_loss(train_opt['gtRecon_opt']).to(self.device)
        else:
            self.cri_gtRecon = None

        if train_opt.get('lowRecon_opt'):
            self.cri_lowRecon = build_loss(train_opt['lowRecon_opt']).to(self.device)
        else:
            self.cri_lowRecon = None

        if train_opt.get('refl_opt'):
            self.cri_refl = build_loss(train_opt['refl_opt']).to(self.device)
        else:
            self.cri_refl = None

        if train_opt.get('illuMutualInput_opt'):
            self.cri_illuMutualInput = build_loss(train_opt['illuMutualInput_opt']).to(self.device)
        else:
            self.cri_illuMutualInput = None

        if train_opt.get('illuMutual_opt'):
            self.cri_illuMutual = build_loss(train_opt['illuMutual_opt']).to(self.device)
        else:
            self.cri_illuMutual = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('enhancedIllu_opt'):
            self.cri_enhancedIllu = build_loss(train_opt['enhancedIllu_opt']).to(self.device)
        else:
            self.cri_enhancedIllu = None

        if train_opt.get('enhancedIlluTV_opt'):
            self.cri_enhancedIlluTV = build_loss(train_opt['enhancedIlluTV_opt']).to(self.device)
        else:
            self.cri_enhancedIlluTV = None

        if train_opt.get('reflRestore_opt'):
            self.cri_reflRestore = build_loss(train_opt['reflRestore_opt']).to(self.device)
        else:
            self.cri_reflRestore = None

        if train_opt.get('noisePrior_opt'):
            self.cri_noisePrior = build_loss(train_opt['noisePrior_opt']).to(self.device)
        else:
            self.cri_noisePrior = None

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
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, self.enhanced_L, self.L, self.restored_R, self.R, self.noise, self.L_hat = self.net_g(self.lq)

        _, _, self.gt_L, _, self.gt_R, self.gt_noise, _ = self.net_g(self.gt)

        _, _, self.output_noisePrior = self.net_noisePrior(self.output)
        _, _, self.gt_noisePrior = self.net_noisePrior(self.gt)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_lowRecon:
            l_lowRecon = self.cri_lowRecon(self.L * self.R + self.noise, self.lq)
            l_total += l_lowRecon
            loss_dict['l_lowRecon'] = l_lowRecon

        if self.cri_gtRecon:
            l_gtRecon = self.cri_gtRecon(self.gt_L * self.gt_R, self.gt)
            l_total += l_gtRecon
            loss_dict['l_gtRecon'] = l_gtRecon

        if self.cri_refl:
            l_refl = self.cri_refl(self.R, self.gt_R)
            l_total += l_refl
            loss_dict['l_refl'] = l_refl

        if self.cri_illuMutualInput:
            l_illuMutualInputLQ = self.cri_illuMutualInput(self.L, self.lq)
            l_total += l_illuMutualInputLQ
            loss_dict['l_illuMutualInputLQ'] = l_illuMutualInputLQ

            l_illuMutualInputGT = self.cri_illuMutualInput(self.gt_L, self.gt)
            l_total += l_illuMutualInputGT
            loss_dict['l_illuMutualInputGT'] = l_illuMutualInputGT

        if self.cri_illuMutual:
            l_illuMutual = self.cri_illuMutual(self.L, self.gt_L)
            l_total += l_illuMutual
            loss_dict['l_illuMutual'] = l_illuMutual

            l_illuMutualEnhanced = self.cri_illuMutual(self.enhanced_L, self.gt_L)
            l_total += l_illuMutualEnhanced
            loss_dict['l_illuMutualEnhanced'] = l_illuMutualEnhanced

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_enhancedIllu:
            l_EnhancedIllu = self.cri_enhancedIllu(torch.mean(self.enhanced_L, 1).unsqueeze(1), self.gt_L)
            l_total += l_EnhancedIllu
            loss_dict['l_enhancedIllu'] = l_EnhancedIllu

        if self.cri_enhancedIlluTV:
            l_EnhancedIlluTV = self.cri_enhancedIlluTV(torch.mean(self.enhanced_L, 1).unsqueeze(1), self.gt_L)
            l_total += l_EnhancedIlluTV
            loss_dict['l_EnhancedIlluTV'] = l_EnhancedIlluTV

        if self.cri_reflRestore:
            l_reflRestore = self.cri_reflRestore(self.restored_R, self.gt_R)
            l_total += l_reflRestore
            loss_dict['l_reflRestore'] = l_reflRestore

        if self.cri_noisePrior:
            l_noisePrior = self.cri_noisePrior(self.output_noisePrior, self.gt_noisePrior)
            l_total += l_noisePrior
            loss_dict['l_noisePrior'] = l_noisePrior

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output_test, self.enhanced_L_test, self.L_test, self.restored_R_test, self.R_test, self.noise_test, self.L_prior_cond_test = self.net_g_ema(self.lq)
                _, _, self.gt_L_test, _, self.gt_R_test, self.gt_noise_test, _ = self.net_g_ema(self.gt)
                # self.gt_R_test = self.gt / (torch.max(self.gt, 1)[0].unsqueeze(1) + 1e-8)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output_test, self.enhanced_L_test, self.L_test, self.restored_R_test, self.R_test, self.noise_test, self.L_prior_cond_test = self.net_g(self.lq)
                _, _, self.gt_L_test, _, self.gt_R_test, self.gt_noise_test, _ = self.net_g(self.gt)
                # self.gt_R_test = self.gt / (torch.max(self.gt, 1)[0].unsqueeze(1) + 1e-8)
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
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # [0-255]

            enhanced_img = tensor2img([visuals['enhanced']])
            metric_data['img'] = enhanced_img
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhanced.png')
                    else:
                        save_img_path_enhanced = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhanced.png')
                imwrite(enhanced_img, save_img_path_enhanced)

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

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['enhanced'] = self.output_test.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
