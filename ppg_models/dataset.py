import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import ppg_from_txt

from ppg_models.hr_utils import calculate_metric_per_video
from ppg_models.data_utils import preprocess

def calculate_hr_fft(df):
    hr = []
    for _, row in df.iterrows():
        ppg = ppg_from_txt(row['ppg_sync'])
        ppg = ppg[row['begin']:row['end']]
        hr.append(calculate_metric_per_video(ppg))
    
    return hr

class TSCANDataset(Dataset):
    def __init__(self, df, config, mode='train'):
        self.df = pd.read_csv(df)
        self.hr = calculate_hr_fft(self.df)
        self.config = config
        self.mode = mode
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        video = np.load(self.df.loc[index]['video'])
        ppg = ppg_from_txt(self.df.loc[index]['ppg_sync'])
        ppg = ppg[self.df.loc[index]['begin']:self.df.loc[index]['end']]
        
        frames, ppg = preprocess(video, ppg, self.config)

        if self.mode == 'train':
            return frames[0], ppg[0], self.hr[index]
        return frames[0], ppg[0], self.hr[index], self.df.loc[index]['video']