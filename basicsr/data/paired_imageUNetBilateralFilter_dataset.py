from torch.utils import data as data
from torchvision.transforms.functional import normalize
from skimage import feature
import random
import cv2

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, pairedDehaze_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import torch
import numpy as np

@DATASET_REGISTRY.register()
class PairedImageUNetBilateralFilterDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageUNetBilateralFilterDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)


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
        return x


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        gamma1 = random.uniform(0.2, 0.5)
        gamma2 = random.uniform(0.5, 1.0)
        img_lq1 = img_lq ** (1 / gamma1)
        img_lq2 = img_lq ** (1 / gamma2)

        img_lq = img_lq.max(2)
        img_lq1 = img_lq1.max(2)
        img_lq2 = img_lq2.max(2)

        img_lq = np.expand_dims(img_lq, axis=2)
        img_lq1 = np.expand_dims(img_lq1, axis=2)
        img_lq2 = np.expand_dims(img_lq2, axis=2)

        img_gt = cv2.bilateralFilter(img_lq, 5, 0.2, 50)
        img_gt1 = cv2.bilateralFilter(img_lq1, 5, 0.2, 50)
        img_gt2 = cv2.bilateralFilter(img_lq2, 5, 0.2, 50)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_gt1 = np.expand_dims(img_gt1, axis=2)
        img_gt2 = np.expand_dims(img_gt2, axis=2)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_gt1, img_gt2, img_lq, img_lq1, img_lq2 = img2tensor([img_gt, img_gt1, img_gt2, img_lq, img_lq1, img_lq2], bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            win_size = self.opt['win_size']

            img_lq = self.window_partition(img_lq, win_size=win_size)
            img_lq1 = self.window_partition(img_lq1, win_size=win_size)
            img_lq2 = self.window_partition(img_lq2, win_size=win_size)
            # img_gt = self.window_partition(img_gt, win_size=win_size)

            disruption = torch.randperm(img_lq.size(0))
            index = [i for i in range(img_lq.shape[0])]
            mask_disruption = disruption[:int(self.opt['percent'] * img_lq.shape[0])]
            for i in range(len(mask_disruption)):
                img_lq[mask_disruption[i], :, :, :] = 0
                img_lq1[mask_disruption[i], :, :, :] = 0
                img_lq2[mask_disruption[i], :, :, :] = 0
            # for i in range(int(mask_percent * img_lq.shape[0])):
            #     img_lq[i, :, :, :] = img_lq[disruption[i], :, :, :]

            img_lq = self.window_reverse(img_lq, win_size=win_size, H=gt_size, W=gt_size).permute(2, 0, 1)
            img_lq1 = self.window_reverse(img_lq1, win_size=win_size, H=gt_size, W=gt_size).permute(2, 0, 1)
            img_lq2 = self.window_reverse(img_lq2, win_size=win_size, H=gt_size, W=gt_size).permute(2, 0, 1)
        else:
            H, W = img_lq.shape[1], img_lq.shape[2]
            img_lq = self.window_partition(img_lq, win_size=self.opt['win_size_test'])
            img_lq1 = self.window_partition(img_lq1, win_size=self.opt['win_size_test'])
            img_lq2 = self.window_partition(img_lq2, win_size=self.opt['win_size_test'])

            disruption = torch.randperm(img_lq.size(0))
            index = [i for i in range(img_lq.shape[0])]
            mask_disruption = disruption[:int(self.opt['percent'] * img_lq.shape[0])]
            for i in range(len(mask_disruption)):
                img_lq[mask_disruption[i], :, :, :] = 0
                img_lq1[mask_disruption[i], :, :, :] = 0
                img_lq2[mask_disruption[i], :, :, :] = 0

            img_lq = self.window_reverse(img_lq, win_size=self.opt['win_size_test'], H=H, W=W).permute(2, 0, 1)
            img_lq1 = self.window_reverse(img_lq1, win_size=self.opt['win_size_test'], H=H, W=W).permute(2, 0, 1)
            img_lq2 = self.window_reverse(img_lq2, win_size=self.opt['win_size_test'], H=H, W=W).permute(2, 0, 1)
        return {'lq': img_lq, 'lq1': img_lq1, 'lq2': img_lq2, 'gt': img_gt, 'gt1': img_gt1, 'gt2': img_gt2, 'mask_disruption': mask_disruption, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
