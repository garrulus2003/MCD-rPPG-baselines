"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""
import json
import math
import os
import numpy as np
import pandas as pd
from scipy import signal
from ppg_models.hr_utils import detrend
from argparser import parse_args
from tqdm.auto import tqdm

from ppg_models.hr_utils import calculate_metric_per_video
from ppg_models.dataset import calculate_hr_fft

def _process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
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
    return np.asarray(RGB)


def POS_WANG(frames, fs, masked=False, masks=None):
    WinSec = 1.6
    
    if masked:
        RGB = _process_video_masked(frames, masks)
    else:
        RGB = _process_video(frames)
        
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP

def run(args=None):
    args = parse_args(args)
    df = pd.read_csv(args.dataframe)
    hrs_pred = []
    
    for _, row in tqdm(df.iterrows()):
        frames = np.load(row['video'])

        if args.model_name == 'POS_WANG':
            BVP = POS_WANG(frames, 30)
        else:
            masks = (np.load(row['mask']) == 3)
            BVP = POS_WANG(frames, 30, True, masks)
            
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