from __future__ import absolute_import

import math
import numpy as np
from random import randint

from .misc import *

__all__ = ['accuracy', 'AverageMeter']

def accuracy(output, target, thr=100):
    acc = 0
    output = np.array(output)
    target = np.array(target)
    for batch in range(output.shape[0]):
        for month in range(output.shape[1]):
            err = np.sqrt(np.square(output[batch][month] - target[batch][month]))
            if err < thr:
                acc += 1
    acc = acc / (output.shape[0] * output.shape[1])

    return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
