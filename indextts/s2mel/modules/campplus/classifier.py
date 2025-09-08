# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.campplus.layers import DenseLayer


class CosineClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, config_str='batchnorm')
            )
            input_dim = inter_dim

        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x

class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        self.nonlinear = nn.ReLU(inplace=True)
        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, bias=True)
            )
            input_dim = inter_dim

        self.linear = nn.Linear(input_dim, out_neurons, bias=True)

    def forward(self, x):
        # x: [B, dim]
        x = self.nonlinear(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.linear(x)
        return x