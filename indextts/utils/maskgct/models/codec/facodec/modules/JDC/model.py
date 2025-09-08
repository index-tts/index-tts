# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is borrowed from https://github.com/yl4579/PitchExtractor/blob/main/model.py

"""
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import torch
from torch import nn


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """

    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False
            ),  # out: (b, 64, 31, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 31, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(
            in_channels=64, out_channels=128
        )  # (b, 128, 31, 128)
        self.res_block2 = ResBlock(
            in_channels=128, out_channels=192
        )  # (b, 192, 31, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 31, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.2),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        # in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        # in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        # in = (b, 640, 31, 2), out = (b, 256, 31, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.2),
        )

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256, batch_first=True, bidirectional=True
        )  # (b, 31, 512)

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256, batch_first=True, bidirectional=True
        )  # (b, 31, 512)

        # input: (b * 31, 512)
        self.classifier = nn.Linear(
            in_features=512, out_features=self.num_class
        )  # (b * 31, num_class)

        # input: (b * 31, 512)
        self.detector = nn.Linear(
            in_features=512, out_features=2
        )  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def get_feature_GAN(self, x):
        seq_len = x.shape[-2]
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)

        return poolblock_out.transpose(-1, -2)

    def get_feature(self, x):
        seq_len = x.shape[-2]
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)

        return self.pool_block[2](poolblock_out)

    def forward(self, x):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 31, 722), (b, 31, 2)
        """
        ###############################
        # forward pass for classifier #
        ###############################
        seq_len = x.shape[-1]
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        GAN_feature = poolblock_out.transpose(-1, -2)
        poolblock_out = self.pool_block[2](poolblock_out)

        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        classifier_out = (
            poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        classifier_out, _ = self.bilstm_classifier(
            classifier_out
        )  # ignore the hidden states

        classifier_out = classifier_out.contiguous().view((-1, 512))  # (b * 31, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view(
            (-1, seq_len, self.num_class)
        )  # (b, 31, num_class)

        # sizes: (b, 31, 722), (b, 31, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return torch.abs(classifier_out.squeeze(-1)), GAN_feature, poolblock_out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x
