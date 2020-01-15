import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, stride=2, padding=1, kernel_size=3),
            nn.tanh()

        )

            # nn.Upsample(align_corners=True, scale_factor=2, mode='bilinear'),
            # nn.Conv2d(128, 64, stride=1, padding=1, kernel_size=3),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Upsample(align_corners=True, scale_factor=2, mode='bilinear'),
            # nn.Conv2d(64, 32, stride=1, padding=1, kernel_size=3),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Upsample(align_corners=True, scale_factor=2, mode='bilinear'),
            # nn.Conv2d(32, 16, stride=1, padding=1, kernel_size=3),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Upsample(align_corners=True, scale_factor=2, mode='bilinear'),
            # nn.Conv2d(16, 1, stride=1, padding=1, kernel_size=3),
            # nn.ReLU(True),)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x