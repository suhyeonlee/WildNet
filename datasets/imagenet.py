"""
ImageNet Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image
from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.cityscapes_labels as cityscapes_labels
import scipy.misc as m

from config import cfg

import pdb
import random
from itertools import cycle, islice

root = cfg.DATASET.IMAGENET_DIR
img_postfix = '.JPEG'


def make_cv_splits():
    """
    Create splits of train/valid data.
    A split is a lists of cities.
    split0 is aligned with the default Cityscapes train/valid.
    """
    trn_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')

    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_cities = sorted(trn_cities)

    all_cities = val_cities + trn_cities
    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_cities // cfg.DATASET.CV_SPLITS
        for j in range(num_cities):
            if j >= offset and j < (offset + num_val_cities):
                split['val'].append(all_cities[j])
            else:
                split['train'].append(all_cities[j])
        cv_splits.append(split)

    return cv_splits


def add_items(items, aug_items, cities, img_path, mode, maxSkip):
    """

    Add More items ot the list from the augmented dataset
    """

    for c in cities:
        c_items = [name.split(img_postfix)[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + img_postfix))
            items.append(item)


def make_dataset(mode, maxSkip=0, cv_split=0):
    """
    Assemble list of images + mask files

    fine -   modes: train/valid/test/trainval    cv:0,1,2
    coarse - modes: train/valid                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    aug_items = []

    assert mode in ['train', 'val', 'trainval']
    cv_splits = make_cv_splits()
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        logging.info('{} imagenet: '.format(mode) + str(cv_splits[cv_split][mode]))
        add_items(items, aug_items, cv_splits[cv_split][mode], root, mode, maxSkip)

    logging.info('ImageNet-{}: {} images'.format(mode, len(items) + len(aug_items)))

    return items, aug_items


class ImageNet(data.Dataset):

    def __init__(self, mode, maxSkip=0, joint_transform=None, sliding_crop=None,
                 transform=None, dump_images=False,
                 cv_split=None, eval_mode=False,
                 eval_scales=None, eval_flip=False, image_in=False,
                 extract_feature=False):
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        self.image_in = image_in
        self.extract_feature = extract_feature


        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0
        self.imgs, _ = make_dataset(mode, self.maxSkip, cv_split=self.cv_split)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(final_tensor)
            return_imgs.append(imgs)
        return return_imgs

    def __getitem__(self, index):

        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.eval_mode:
            return [transforms.ToTensor()(img)], self._eval_get_item(img, 
                                                                     self.eval_scales,
                                                                     self.eval_flip), img_name
        # Image Transformations
        if self.extract_feature is not True:
            if self.joint_transform is not None:
                img = self.joint_transform(img)

        if self.transform is not None:
            img = self.transform(img)
        
        rgb_mean_std_gt = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_gt = transforms.Normalize(*rgb_mean_std_gt)(img)

        rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.image_in:
            eps = 1e-5
            rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                    [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
        img = transforms.Normalize(*rgb_mean_std)(img)

        # Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            img.save(out_img_fn)

        return img, img_name

    def __len__(self):
        return len(self.imgs)
