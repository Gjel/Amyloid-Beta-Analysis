import torch.nn as nn

class ConvBlock(nn.Module):

    DROP_PROBABILITY = 0.2
    FILTER_SIZE = (5,)

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv2d(in_channels, out_channels, self.FILTER_SIZE)
        self.elu_1 = nn.ELU()
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(self.DROP_PROBABILITY)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, self.FILTER_SIZE)
        self.elu_2 = nn.ELU()
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

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

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.contract_1 = ConvBlock(in_channels, self.NUM_CHANNELS_1)
        self.pool_1 = nn.MaxPool2d(2)

        self.contract_2 = ConvBlock(self.NUM_CHANNELS_1, self.NUM_CHANNELS_2)
        self.pool_2 = nn.MaxPool2d(2)

        self.contract_3 = ConvBlock(self.NUM_CHANNELS_2, self.NUM_CHANNELS_3)
        self.pool_3 = nn.MaxPool2d(2)

        self.contract_4 = ConvBlock(self.NUM_CHANNELS_3, self.NUM_CHANNELS_4)
        self.pool_4 = nn.MaxPool2d(2)

        self.bottom = ConvBlock(self.NUM_CHANNELS_4, self.NUM_CHANNELS_5)

        self.t_conv_4 = nn.ConvTranspose2d(self.NUM_CHANNELS_5, self.NUM_CHANNELS_4, self.T_CONV_KERNEL, self.T_CONV_STRIDE)
        self.expand_4 = ConvBlock(self.NUM_CHANNELS_5, self.NUM_CHANNELS_4)

        self.t_conv_3 = nn.ConvTranspose2d(self.NUM_CHANNELS_4, self.NUM_CHANNELS_3, self.T_CONV_KERNEL, self.T_CONV_STRIDE)
        self.expand_3 = ConvBlock(self.NUM_CHANNELS_4, self.NUM_CHANNELS_3)

        self.t_conv_2 = nn.ConvTranspose2d(self.NUM_CHANNELS_3, self.NUM_CHANNELS_2, self.T_CONV_KERNEL, self.T_CONV_STRIDE)
        self.expand_2 = ConvBlock(self.NUM_CHANNELS_3, self.NUM_CHANNELS_2)

        self.t_conv_1 = nn.ConvTranspose2d(self.NUM_CHANNELS_2, self.NUM_CHANNELS_1, self.T_CONV_KERNEL, self.T_CONV_STRIDE)
        self.expand_1 = ConvBlock(self.NUM_CHANNELS_2, self.NUM_CHANNELS_1)

    def forward(self, x):
        layer_1 =








