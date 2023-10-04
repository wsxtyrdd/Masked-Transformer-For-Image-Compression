import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


class Distortion(torch.nn.Module):
    def __init__(self, distortion_type):
        super(Distortion, self).__init__()
        if distortion_type == 'MSE':
            self.metric = nn.MSELoss()
        elif distortion_type == 'MS-SSIM':
            self.metric = ms_ssim
        else:
            print("Unknown distortion type!")
            raise ValueError

    def forward(self, X, Y):
        if self.metric == ms_ssim:
            return 1 - self.metric(X, Y, data_range=1)
        else:
            return self.metric(X, Y)