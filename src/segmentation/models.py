import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ConvBlock(nn.Module):

    DROP_PROBABILITY = 0.2
    FILTER_SIZE = (5, 5)
    PADDING = (2, 2)

    def __init__(self, in_channels: int, out_channels: int, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.conv_1 = nn.Conv2d(in_channels, out_channels, self.FILTER_SIZE, padding=self.PADDING, device=device)
        self.elu_1 = nn.ELU()
        self.batch_norm_1 = nn.BatchNorm2d(out_channels, device=device)

        self.dropout = nn.Dropout2d(self.DROP_PROBABILITY)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, self.FILTER_SIZE, padding=self.PADDING, device=device)
        self.elu_2 = nn.ELU()
        self.batch_norm_2 = nn.BatchNorm2d(out_channels, device=device)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.elu_1(x)
        x = self.batch_norm_1(x)

        x = self.dropout(x)

        x = self.conv_2(x)
        x = self.elu_2(x)
        x = self.batch_norm_2(x)

        return x


class Unet(nn.Module):

    NUM_CHANNELS_1 = 32
    NUM_CHANNELS_2 = 64
    NUM_CHANNELS_3 = 128
    NUM_CHANNELS_4 = 256
    NUM_CHANNELS_5 = 512

    T_CONV_STRIDE = (2, 2)
    T_CONV_KERNEL = (2, 2)

    def __init__(self, in_channels: int, out_channels: int, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.contract_1 = ConvBlock(in_channels, self.NUM_CHANNELS_1, device=device)
        self.pool_1 = nn.MaxPool2d(2)

        self.contract_2 = ConvBlock(self.NUM_CHANNELS_1, self.NUM_CHANNELS_2, device=device)
        self.pool_2 = nn.MaxPool2d(2)

        self.contract_3 = ConvBlock(self.NUM_CHANNELS_2, self.NUM_CHANNELS_3, device=device)
        self.pool_3 = nn.MaxPool2d(2)

        self.contract_4 = ConvBlock(self.NUM_CHANNELS_3, self.NUM_CHANNELS_4, device=device)
        self.pool_4 = nn.MaxPool2d(2)

        self.bottom = ConvBlock(self.NUM_CHANNELS_4, self.NUM_CHANNELS_5, device=device)

        self.t_conv_4 = nn.ConvTranspose2d(self.NUM_CHANNELS_5, self.NUM_CHANNELS_4, self.T_CONV_KERNEL, self.T_CONV_STRIDE, device=device)
        self.expand_4 = ConvBlock(self.NUM_CHANNELS_5, self.NUM_CHANNELS_4)

        self.t_conv_3 = nn.ConvTranspose2d(self.NUM_CHANNELS_4, self.NUM_CHANNELS_3, self.T_CONV_KERNEL, self.T_CONV_STRIDE, device=device)
        self.expand_3 = ConvBlock(self.NUM_CHANNELS_4, self.NUM_CHANNELS_3)

        self.t_conv_2 = nn.ConvTranspose2d(self.NUM_CHANNELS_3, self.NUM_CHANNELS_2, self.T_CONV_KERNEL, self.T_CONV_STRIDE, device=device)
        self.expand_2 = ConvBlock(self.NUM_CHANNELS_3, self.NUM_CHANNELS_2)

        self.t_conv_1 = nn.ConvTranspose2d(self.NUM_CHANNELS_2, self.NUM_CHANNELS_1, self.T_CONV_KERNEL, self.T_CONV_STRIDE, device=device)
        self.expand_1 = ConvBlock(self.NUM_CHANNELS_2, self.NUM_CHANNELS_1)

        self.segment = nn.Sequential(nn.Conv2d(self.NUM_CHANNELS_1, out_channels, (1, 1), device=device), nn.Sigmoid())

    def forward(self, x):
        cat_dim = 1 if len(x.shape) == 4 else 0

        down_1 = self.contract_1(x)
        down_2 = self.contract_2(self.pool_1(down_1))
        down_3 = self.contract_3(self.pool_2(down_2))
        down_4 = self.contract_4(self.pool_3(down_3))

        deep = self.bottom(self.pool_4(down_4))

        up_4 = self.expand_4(torch.cat((down_4, self.t_conv_4(deep)), cat_dim))
        up_3 = self.expand_3(torch.cat((down_3, self.t_conv_3(up_4)), cat_dim))
        up_2 = self.expand_2(torch.cat((down_2, self.t_conv_2(up_3)), cat_dim))
        up_1 = self.expand_1(torch.cat((down_1, self.t_conv_1(up_2)), cat_dim))
        return self.segment(up_1)

def brainsec_resnet18():
    model = resnet18(ResNet18_Weights)
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, 1)
    return model


        






