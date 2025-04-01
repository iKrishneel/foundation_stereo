#!/usr/bin/env python

import os.path as osp
import torch
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import click

from igniter.logger import logger
from igniter.main import get_full_config
from igniter.builder import build_engine
import foundation_stereo  # NOQA

@click.command()
@click.argument('cfg')
@click.option('--images', type=str)
@click.option('--weights', type=str)
@click.option('--scale', type=float, default=1)
def main(cfg, images, weights, scale, valid_iters=32,):
    cfg = get_full_config(cfg)

    if weights is not None:
        logger.info(f'Using weights: {weights}')
        cfg.build[cfg.build.model].weights = weights 

    engine = build_engine(cfg)

    im_left = cv.imread(osp.join(images, 'left.png'))
    im_right = cv.imread(osp.join(images, 'right.png'))

    disparity = engine(np.asarray(im_left), np.asarray(im_right))

    disp = disparity.cpu().numpy()
    plt.imshow(disp); plt.show()


if __name__ == '__main__':
    main()

