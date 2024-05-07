import argparse
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_class',
        type=str,
        required=True,
        choices=['by_frame', 'by_video', 'ppg']
    )

    parser.add_argument(
        '--dataframe',
        type=str,
        required=True,
        help="Data that will be used for training"
    )

    # args possible in all settings
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Batch size"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="Learning rate for optimization"
    )

    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default="test_exp",
        help="Directory where results will be saved"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=43,
        help='Random seed'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        choices = [
            'r3d', 'mc3', 'r2plus1d', 'mvit', 
            'resnet50', 'vit', 'swin', 
            'rexnet150_pretrained', 'enet2_pretrained',
            'TSCAN', 'POS_WANG', 'LGI',
            'POS_WANG_mask', 'LGI_mask'
        ],
        help="Model name"
    )

    parser.add_argument(
        '--path_to_ckpt',
        type=str,
        help="Path checkpoint for evaluation"
    )

    # args for frame or video setting
    parser.add_argument(
        '--target', 
        type=str,
        choices = [
            'sex', 'age', 'bmi', 'lower_ap', 'upper_ap', 'saturation', 
            'temperature', 'stress', 'hemoglobin', 'glycated_hemoglobin',
            'cholesterol', 'respiratory', 'rigidity', 'pulse', 'all'
        ],
        help="Target parameter"
    )

    parser.add_argument(
        '--criterion_name',
        type=str,
        default='mae',
        choices = ['mae', 'mse'],
        help="Criterion for regression task"
    )
    
    parser.add_argument(
        '--two_layers',
        type=bool,
        default=True,
        help="Whether there should be 1 or 2 layers"
    )

    parser.add_argument(
        '--scale',
        type=int,
        default=1,
        help='By which number target should be divided'
    )
    
    parser.add_argument(
        '--unfreeze',
        action='store_true'
    )

    parser.add_argument(
        '--from_scratch', 
        action='store_true'
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=300,
        help="Hidden dimension if two layers"
    )

    # only for video
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='To which size the video is cropped'
    )

    parser.add_argument(
        '--frequency',
        type=int,
        default=1,
        help='Each n-th grame is processed'
    )
    
    parser.add_argument(
        '--avg_over_frames',
        action='store_true'
    )

    return parser.parse_args(args)
    