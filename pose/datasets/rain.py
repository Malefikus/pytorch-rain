from __future__ import print_function, absolute_import

import os
import numpy as np
import scipy.io as io

import torch
import torch.utils.data as data

from pose.utils.osutils import *

class Rain(data.Dataset):
    def __init__(self, train=True):
        self.is_train = train           # training set or test set

        # create train/val split
        rain = io.loadmat('data/rain/years66.mat')
        self.rain_data = rain['data']
        self.rain_train = self.rain_data[0:61]
        self.rain_valid = self.rain_data[56:66]
        self.consec_idx = _make_idx(train)

    def _make_idx(self, isTrain):
        consec_idx = []
        if isTrain:
            for i in range(len(self.rain_train)-6):
                sample_idx = np.linspace(i, i+6, 7)
                consec_idx.append(sample_idx)
        else:
            for i in range(len(self.rain_valid)-6):
                sample_idx = np.linspace(i, i+6, 7)
                consec_idx.append(sample_idx)
        return consec_idx

    def __getitem__(self, index):
        indices = self.consec_idx[index]

        if self.is_train:
            inp = self.rain_train[indices[0]:indices[6]]
            target = self.rain_train[indices[6]]
        else:
            inp = self.rain_valid[indices[0]:indices[6]]
            target = self.rain_valid[indices[6]]

        return inp, target

    def __len__(self):
        len(self.consec_idx)
