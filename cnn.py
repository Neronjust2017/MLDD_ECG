import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import linear, conv1d


class MLP(nn.Module):
    def __init__(self, num_classes=1000):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # when you add the convolution and batch norm, below will be useful
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):

        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(x, inplace=True)

        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        end_points = {'Predictions': F.softmax(input=x, dim=-1)}

        return x, end_points


def MLPNet(**kwargs):
    model = MLP(**kwargs)
    return model


class CNN(nn.Module):
    def __init__(self, num_classes=2, channels=[32, 64, 128], kernel_size=[11, 11, 11], pool='max', pool_kernel_size=2, n_hid=[100, 80], drop_rate=0.2, soft_max=True):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        # Convolution
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=11)

        # Pooling
        self.pooling = nn.MaxPool1d(kernel_size=2)

        # Fully-connected Layer
        self.fc1 = nn.Linear(in_features=128, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=num_classes)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

        # Activation
        self.act = nn.ReLU()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):

        x = conv1d(inputs=x,
                   weight=self.conv1.weight,
                   bias=self.conv1.bias,
                   meta_step_size=meta_step_size,
                   meta_loss=meta_loss,
                   stop_gradient=stop_gradient)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = conv1d(inputs=x,
                   weight=self.conv2.weight,
                   bias=self.conv2.bias,
                   meta_step_size=meta_step_size,
                   meta_loss=meta_loss,
                   stop_gradient=stop_gradient)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = conv1d(inputs=x,
                   weight=self.conv3.weight,
                   bias=self.conv3.bias,
                   meta_step_size=meta_step_size,
                   meta_loss=meta_loss,
                   stop_gradient=stop_gradient)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = torch.mean(x, dim=2)

        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = self.act(x)
        x = self.dropout(x)

        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = self.act(x)
        x = self.dropout(x)

        x = linear(inputs=x,
                   weight=self.fc3.weight,
                   bias=self.fc3.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        end_points = {'Predictions': F.softmax(input=x, dim=-1)}

        return x, end_points


def CNNNet(**kwargs):
    model = CNN(**kwargs)
    return model