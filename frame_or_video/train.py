import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

from argparser import parse_args
from utils import fix_seed, get_criterion, model_type, get_weights, get_crop
from frame_or_video.utils import train_net

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

    preprocess = get_weights(args.model_name).transforms()
    
    if args.model_class == 'by_video':
        crop = get_crop(args.model_name, args.crop)
    
        train_dataset = VideoDataset(
            args.dataframe, args.target, 
            preprocess,
            is_regression=is_regression, 
            scale=args.scale,
            mode='train',
            crop=crop,
            frequency = args.frequency,
            avg_over_frames = args.avg_over_frames
        )
        
    else:
        train_dataset = FramesDataset(
            args.dataframe, args.target,
            preprocess,
            is_regression=is_regression, 
            scale=args.scale,
            mode='train'
        )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    criterion = get_criterion(is_regression, args.criterion_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = train_net(
        model, train_loader, optimizer, 
        criterion, args.num_epochs, 
        is_regression=is_regression, 
        device=args.device,
        scale=args.scale,
        all_targets=(args.target == 'all')
    )
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    for metric, value in results.items():
        metric_path = os.path.join(args.exp_name, metric)
        np.save(metric_path, np.array(value))

    torch.save(
        model.to('cpu').state_dict(), 
        os.path.join(args.exp_name, 'state_dict.pt')
    )
    
if __name__ == '__main__':
    run()