import os
import copy
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
    def __init__(self, data_dir, train_mode='CP' ,data_mode='full', is_train: str ='train'):
        super(MusicDataset, self).__init__()
        assert train_mode in ['CP', 'JP', 'JC'], 'wrong train mode was given'
        assert data_mode in ['partial', 'full'], 'wrong data mode was given'

        a_data_dir = train_mode + '_' + train_mode[0]
        b_data_dir = train_mode + '_' + train_mode[1]
        print('data_dir',data_dir, a_data_dir, b_data_dir)
        a_dir = os.path.join(data_dir, a_data_dir, is_train + os.sep)
        b_dir = os.path.join(data_dir, b_data_dir, is_train + os.sep)
        print('read data from ', a_dir, b_dir)
        self.A_list = glob.glob(a_dir + '*' + '.npy')
        self.B_list = glob.glob(b_dir + '*' + '.npy')
        self.transform = ToTensor()

        assert len(self.A_list) == len(
            self.B_list), 'the lengths of a and b are different'

        if is_train == 'test':
            self.mixed_dir = self.A_list
        elif data_mode == 'partial':
            self.mixed_dir = self.A_list + self.B_list
        elif data_mode == 'full':
            self.mixed_dir = glob.glob(data_dir + '/JCP_mixed/*.*')

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        bar_a = load_npy(self.A_list[idx])
        bar_b = load_npy(self.B_list[idx])
        bar_mixed = load_npy(self.mixed_dir[idx])
        baridx = np.array([idx])
        if len(bar_a.shape) != 3:
            bar_a = np.expand_dims(bar_a, axis=2)
        if len(bar_b.shape) != 3:
            bar_b = np.expand_dims(bar_b, axis=2)

        sample = {'baridx': baridx, 'bar_a': bar_a,
                  'bar_b': bar_b, 'bar_mixed': bar_mixed}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_name(self, idx):
        data_dir = self.A_list[idx]
        name = data_dir.split(os.sep)[-1]
        name = name.split('.')[0]
        return name

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=ToTensor()):
        assert os.path.exists(data_dir)

        self.data_dir_list = glob.glob(data_dir + "*" + '.npy')
        self.transform = transform
    def __getitem__(self, idx):
        bar_a = load_npy(self.data_dir_list[idx])
        bar_b = copy.deepcopy(bar_a)
        bar_mixed = copy.deepcopy(bar_a)
        baridx = np.array([idx])
        if len(bar_a.shape) != 3:
            bar_a = np.expand_dims(bar_a, axis=2)
        if len(bar_b.shape) != 3:
            bar_b = np.expand_dims(bar_b, axis=2)

        sample = {'baridx': baridx, 'bar_a': bar_a,
                  'bar_b': bar_b, 'bar_mixed': bar_mixed}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_dir_list)

    def _get_name(self, idx):
        data_dir = self.data_dir_list[idx]
        name = data_dir.split(os.sep)[-1]
        name = name.split('.')[0]
        return name