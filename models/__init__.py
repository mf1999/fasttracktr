# Copyright (c) RuopengGao. All Rights Reserved.
import torch

from utils.utils import distributed_rank
from .fasttracktr import build as build_rtmot_cross


def build_model(config: dict):
    # motip/rtmot (the MOTIP baseline used only for the paper's comparison table) pull in
    # models/deformable_detr, which requires the compiled MultiScaleDeformableAttention CUDA
    # extension (models/ops). Import lazily so the real FastTrackTr path (build_rt_model below,
    # which uses the pure-PyTorch rt_detr_cross attention) doesn't need that extension built.
    from .rtmot import build as build_rtmot
    # model = build_motl(config=config)
    model = build_rtmot(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model

def build_rt_model(config: dict):
    model = build_rtmot_cross(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model