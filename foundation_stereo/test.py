#!/usr/bin/env python

import os.path as osp
import torch
import cv2 as cv
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import click

from foundation_stereo.models.libs.utils import InputPadder
from foundation_stereo.models.foundation_stereo import FoundationStereo

@click.command()
@click.argument('cfg')
@click.option('--images', type=str)
@click.option('--weights', type=str)
@click.option('--scale', type=float, default=1)
def main(cfg, images, weights, scale, valid_iters=32,):
    cfg = OmegaConf.load(cfg)
    model = FoundationStereo(cfg)

    wpth = torch.load(weights, weights_only=False)
    model.load_state_dict(wpth['model'])
    model.cuda().eval()

    im_left = cv.imread(osp.join(images, 'left.png'))
    im_right = cv.imread(osp.join(images, 'right.png'))

    img0 = cv.resize(im_left, fx=scale, fy=scale, dsize=None)
    img1 = cv.resize(im_right, fx=scale, fy=scale, dsize=None)

    img0 = cv.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv.resize(img1, fx=scale, fy=scale, dsize=None)

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(True):
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.inference_mode():
            if not cfg.hiera:
                disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(im_left.shape[:2])

    breakpoint()
    plt.imshow(disp); plt.show()
    

if __name__ == '__main__':
    main()

