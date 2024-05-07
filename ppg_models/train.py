import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

from argparser import parse_args
from utils import fix_seed, get_criterion, model_type, get_weights, get_crop
from ppg_models.utils import train_net

from ppg_models.dataset import TSCANDataset
from ppg_models.TSCAN import TSCAN


def run(args=None):
    args = parse_args(args)
    fix_seed(args.seed)

    config ={
        'DO_CROP_FACE': True,
        'BACKEND': 'HC',
        'USE_LARGE_FACE_BOX': True,
        'LARGE_BOX_COEF': 1.5,
        'DO_DYNAMIC_DETECTION': False,
        'DYNAMIC_DETECTION_FREQUENCY': 30,
        'USE_MEDIAN_FACE_BOX': False,
        'W': 72,
        'H': 72,
        'DATA_TYPE': ['DiffNormalized', 'Standardized'],
        'LABEL_TYPE': 'DiffNormalized',
        'EPOCHS': args.num_epochs,
        'BATCH_SIZE': args.batch_size,
        'LR': args.lr
    }
    
    train_dataset = TSCANDataset(args.dataframe, config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=True
    )
    
    model = TSCAN(img_size=config['H'])
    results = train_net(model, train_loader, config, args.device)

    
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