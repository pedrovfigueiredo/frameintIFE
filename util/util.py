import glob
import numpy as np
import os

from datetime import datetime
from tensorboardX import SummaryWriter

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.reset()
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

    def __str__(self):
        return "{} - {}".format(self.total, self.steps)

    def reset(self):
        self.steps = 0
        self.total = 0

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def current_timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def centerCrop(img, width, height):
	assert img.shape[1] >= height
	assert img.shape[2] >= width
	_,h,w,_ = img.shape
	diff_w = (w-width)//2
	diff_h = (h-height)//2
	return img[:, diff_h:diff_h+height, diff_w:diff_w+width]