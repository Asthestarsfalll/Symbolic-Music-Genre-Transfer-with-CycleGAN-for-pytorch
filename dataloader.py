import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from utils import load_npy


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        baridx, bar_a, bar_b, bar_mixed = sample['baridx'], sample['bar_a'], sample['bar_b'], sample['bar_mixed']
        bar_a = bar_a.transpose((2, 0, 1))
        bar_b = bar_b.transpose((2, 0, 1))
        bar_mixed = bar_mixed.transpose((2, 0, 1))

        return {'baridx': torch.tensor(baridx), 'bar_a': torch.tensor(bar_a, requires_grad=False),
                'bar_b': torch.tensor(bar_b, requires_grad=False),
                'bar_mixed': torch.tensor(bar_mixed, requires_grad=False)}


class MusicDataset(Dataset):
    def __init__(self, a_dir_list, b_dir_list, mode='full', transform=None):
        super(MusicDataset, self).__init__()
        self.A_dir = a_dir_list
        self.B_dir = b_dir_list
        self.transform = transform
        assert len(a_dir_list) == len(b_dir_list), 'the lengths of a and b are different'
        assert mode in ['partial', 'full'], 'wrong mode was given'
        if mode == 'partial':
            self.mixed_dir = a_dir_list + b_dir_list
        else:
            self.mixed_dir = glob.glob('./traindata/JCP_mixed/*.*')

    def __len__(self):
        len_a = len(self.A_dir)
        len_b = len(self.B_dir)
        return len_a if len_a < len_b else len_b

    def __getitem__(self, idx):
        bar_a = load_npy(self.A_dir[idx])
        bar_b = load_npy(self.B_dir[idx])
        bar_mixed = load_npy(self.mixed_dir[idx])
        baridx = np.array([idx])
        if len(bar_a.shape) != 3:
            bar_a = np.expand_dims(bar_a, axis=2)
        if len(bar_b.shape) != 3:
            bar_b = np.expand_dims(bar_b, axis=2)

        sample = {'baridx': baridx, 'bar_a': bar_a, 'bar_b': bar_b, 'bar_mixed': bar_mixed}

        if self.transform:
            sample = self.transform(sample)

        return sample
