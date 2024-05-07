import json
import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

from argparser import parse_args
from utils import fix_seed, get_criterion, model_type, get_weights, get_crop
from ppg_models.utils import eval_net

from ppg_models.dataset import TSCANDataset
from ppg_models.TSCAN import TSCAN
import ppg_models.POS_WANG
import ppg_models.LGI


def run(args=None):
    args_parsed = parse_args(args)

    if 'POS_WANG' in args_parsed.model_name:
        ppg_models.POS_WANG.run(args)
        return 
        
    if 'LGI' in args_parsed.model_name:
        ppg_models.LGI.run(args)
        return

    args = args_parsed
    fix_seed(args.seed)
    
    assert args.model_name == 'TSCAN', "Only POS_WANG, LGI and TSCAN are supported"
    
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
    
    test_dataset = TSCANDataset(args.dataframe, config, mode='test')
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False
    )
    
    model = TSCAN(img_size=config['H'])
    model.load_state_dict(torch.load(args.path_to_ckpt))

    predictions, metrics = eval_net(model, test_loader, config, args.device)
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    predictions.to_csv(os.path.join(args.exp_name, 'predictions.csv'), index=False)

    with open(os.path.join(args.exp_name, 'metrics.json'), "w") as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    run()