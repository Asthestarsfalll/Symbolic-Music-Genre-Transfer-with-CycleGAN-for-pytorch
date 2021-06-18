import numpy as np
from torch.utils import data


def load_npy(npy_dir):
    npy = np.load(npy_dir).astype(np.float32)
    return npy


data_dir = '/run/media/czh/soft/Music/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch/traindata/CP_P/train/pop_piano_train_14514.npy'
n = load_npy(data_dir)
print(n.shape)
