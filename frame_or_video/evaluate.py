import json
import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

from argparser import parse_args
from utils import fix_seed, get_criterion, model_type, get_weights, get_crop
from frame_or_video.utils import eval_net

from frame_or_video.dataset import FramesDataset, VideoDataset
from frame_or_video.model import MedicalParametersModel


def run(args=None):
    args = parse_args(args)
    fix_seed(args.seed)
    is_regression=model_type(args.target)

    model = MedicalParametersModel(
        args.two_layers, args.hidden_dim, is_regression, 
        args.model_name, args.target, args.unfreeze, 
        args.from_scratch
    )

    model.load_state_dict(torch.load(args.path_to_ckpt))

    preprocess = get_weights(args.model_name).transforms()
    
    if args.model_class == 'by_video':
        crop = get_crop(args.model_name, args.crop)
        test_dataset = VideoDataset(
            args.dataframe, args.target, 
            preprocess,
            is_regression=is_regression, 
            scale=args.scale,
            mode='test',
            crop=crop,
            frequency = args.frequency,
            avg_over_frames = args.avg_over_frames
        )
        
    else:
        test_dataset = FramesDataset(
            args.dataframe, args.target,
            preprocess,
            is_regression=is_regression, 
            scale=args.scale,
            mode='test'
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions, metrics = eval_net(
        model, test_loader,
        is_regression=is_regression, 
        device=args.device,
        scale=args.scale,
        all_targets=(args.target == 'all')
    )
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    predictions.to_csv(os.path.join(args.exp_name, 'predictions.csv'), index=False)

    with open(os.path.join(args.exp_name, 'metrics.json'), "w") as file:
        json.dump(metrics, file)

    
if __name__ == '__main__':
    run()