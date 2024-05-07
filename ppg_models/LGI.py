"""LGI
Local group invariance for heart rate estimation from face videos.
Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
(2018).
"""
import json
import math
import os
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import signal
from tqdm.auto import tqdm

from argparser import parse_args
from ppg_models.hr_utils import calculate_metric_per_video
from ppg_models.dataset import calculate_hr_fft


def _process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)

def _process_video_masked(frames, masks):
    """Calculates the average value of each frame."""
    RGB = []
    for i, frame in enumerate(frames):
        mask = masks[i]
        frame *= mask[..., None]
        summation_frame = np.sum(np.sum(frame, axis=0), axis=0)
        summation_mask = np.sum(np.sum(mask, axis=0), axis=0)
        RGB.append(summation_frame / summation_mask)

    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)


def LGI(frames, masked=False, masks=None):
                                      
    if masked:
        precessed_data = _process_video_masked(frames, masks)
    else:
        precessed_data = _process_video(frames)
        
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)

    return bvp


def run(args=None):
    args = parse_args(args)
    df = pd.read_csv(args.dataframe)
    hrs_pred = []
    
    for _, row in tqdm(df.iterrows()):
        frames = np.load(row['video'])

        if args.model_name == 'LGI':
            BVP = LGI(frames)
        else:
            masks = (np.load(row['mask']) == 3)
            BVP = LGI(frames, True, masks)
            
        hrs_pred.append(calculate_metric_per_video(BVP))

    hrs_true = calculate_hr_fft(df)

    predictions = {
        'video': df['video'],
        'hrs_true': hrs_true,
        'hrs_pred': hrs_pred
    }
    
    predictions = pd.DataFrame(predictions)
    metrics = {
        'hr_mae': str(np.abs(np.array(hrs_true) - np.array(hrs_pred)).mean())
    }
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    predictions.to_csv(os.path.join(args.exp_name, 'predictions.csv'), index=False)

    with open(os.path.join(args.exp_name, 'metrics.json'), "w") as file:
        json.dump(metrics, file)
    
if __name__ == '__main__':
    run()