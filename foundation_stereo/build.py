#!/usr/bin/env python

from typing import Dict, Any

import torch
import numpy as np
from omegaconf import DictConfig
from igniter.registry import model_registry, engine_registry
from igniter.engine import InferenceEngine as _InferenceEngine

from .models import FoundationStereo
from .models.libs.utils import InputPadder


@model_registry('foundation_stereo')
def foundation_stereo_model(**kwargs: Dict[str, Any]) -> FoundationStereo:
    args = DictConfig(kwargs)
    model = FoundationStereo(args)
    return model
    

@engine_registry('stereo_inference_engine')
class InferenceEngine(_InferenceEngine):
    def __init__(self, *args, **kwargs):
        super(InferenceEngine, self).__init__(*args, **kwargs)

    @torch.inference_mode()
    def __call__(self, img_left: np.ndarray, img_right: np.ndarray):
        assert img_left.shape == img_right.shape, f'Both images must be same size but got {img_left.shape} and  {img_right.shape}'

        input_shape = img_left.shape[:2]
        device = self.model.classifier[2].weight.device

        if self.transforms is not None:
            img_left, img_right = [self.transforms(img) for img in [img_left, img_right]]

        img_left, img_right = [img[None].to(device) for img in [img_left, img_right] if len(img.shape) != 4]
        with torch.cuda.amp.autocast(True):
            disparity = self.model.forward(img_left, img_right, iters=32, test_mode=True)

        disparity = torch.nn.functional.interpolate(disparity, input_shape, mode='bilinear', align_corners=True)
        return disparity[0, 0]
