import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils import TARGETS, SCALINGS, POS_CLASS

class BaseDataset(Dataset):
    def __init__(self, df, target, preprocess, is_regression=True, scale=1, mode='train'):
        
        df = pd.read_csv(df)
        
        self.mode = mode
        self.preprocess = preprocess

        df_aggregated = df.groupby(
            by=['patient_id', 'camera', 'step']
        ).aggregate(lambda x: x.tolist()).reset_index()

        df_aggregated = df_aggregated[['patient_id', 'camera', 'step', 'video']]
        df_aggregated.rename(columns = {'video': 'video_list'}, inplace=True)

        self.df = df.merge(df_aggregated, on=['patient_id', 'camera', 'step'])

        if mode == 'train':
            self._items = list(self.df['video_list'])
        else:
            self._items = [np.random.choice(paths, 1)[0] for paths in list(self.df['video_list'])]

        if target == 'all':
            targets_dict = {}
            for target in TARGETS:
                targets_dict[target] = np.array(self.df[target]).astype(np.float32)
                targets_dict[target] = targets_dict[target] / SCALINGS[target]
                
            self._target = np.array(
                [[targets_dict[target][i] for target in TARGETS] for i in range(len(self.df))]
            )
        else:
            if is_regression:
                self._target = np.array(self.df[target]).astype(np.float32)
                self._target = self._target / scale
            else:
                self._target = np.array(self.df[target] == POS_CLASS[target]).astype(int)
                assert scale == 1, 'Do not scale classes'
                
    def __len__(self):
        return len(self._items)
        
    def __getitem__(self, index):
        raise ValueError("Only in descendant classes")


class FramesDataset(BaseDataset):
    def __init__(self, df, target, preprocess, is_regression=True, scale=1, mode='train'):
        super().__init__(df, target, preprocess, is_regression, scale, mode)

    def __getitem__(self, index):
        if self.mode == 'train':
            path = np.random.choice(self._items[index], 1)[0]
        else:
            path = self._items[index]
            
        label = self._target[index]

        image = np.array(np.load(path))
        image = np.transpose(image, (2, 0, 1))

        if self.mode == 'train':
            return self.preprocess(torch.tensor(image)), torch.tensor(label)
            
        return self.preprocess(torch.tensor(image)), torch.tensor(label), path

class VideoDataset(BaseDataset):
    def __init__(self, df, target, preprocess,
                 is_regression=True, mode='train', scale=1, crop=None,
                 frequency=1, avg_over_frames=False):
        
        super().__init__(df, target, preprocess, is_regression, scale, mode)
        
        self.crop = crop
        self.frequency = frequency
        self.avg_over_frames = avg_over_frames
        
    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if self.mode == 'train':
            path = np.random.choice(self._items[index], 1)[0]
        else:
            path = self._items[index]
            
        label = self._target[index]

        video = np.load(path)
        video = torch.tensor(np.transpose(video, (0, 3, 1, 2)))

        crop = self.crop if self.crop is not None else video.shape[1]

        assert crop is not None
        assert crop * self.frequency <= video.shape[0]
        
        video_cropped = self.preprocess(video)[:, :crop * self.frequency, :, :]

        if not self.avg_over_frames:
            if self.mode == 'train':
                return video_cropped[:, ::self.frequency], torch.tensor(label)
            return video_cropped[:, ::self.frequency], torch.tensor(label), path

        shape = video_cropped.shape


        video_reshaped = video_cropped.reshape(shape[0], crop, self.frequency, shape[-2], shape[-1]).mean(axis=2)
        if self.mode == 'train':
            return video_reshaped, torch.tensor(label)

        return video_reshaped, torch.tensor(label), path
