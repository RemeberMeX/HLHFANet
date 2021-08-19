import torch
import torch.nn as nn
import torch.nn.functional as F


class Densenly(nn.Module):
    def __init__(self, nin, ksize=3, pad=1, dila=1):
        super(Densenly, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nin, ksize, 1, pad, dila), nn.PReLU())

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.conv(input + x1)
        x3 = self.conv(input + x2 + x1)
        output = input + x3
        return output


class Attention(nn.Module):
    def __init__(self, nin):
        super(Attention, self).__init__()
        ksize1 = 3
        ksize2 = 5
        ksize3 = 7

        pad1 = int(2*(ksize1-1)/2)
        pad2 = int(5*(ksize1-1)/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(nin, nin, ksize1, 1, padding=1, dilation=1), nn.PReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(nin, nin, ksize1, 1, pad1, dilation=2), nn.PReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(nin, nin, ksize1, 1, pad2, dilation=5), nn.PReLU())
        self.dense1 = Densenly(nin, ksize1, 1, 1)
        self.dense2 = Densenly(nin, ksize1, pad1, 2)
        self.dense3 = Densenly(nin, ksize1, pad2, 5)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, input):
        # top
        x1 = self.conv1(input)
        x1 = self.dense1(x1)
        x1 = self.conv1(x1)
        # mid
        x2 = self.conv2(input)
        x2 = self.conv2(x2+x1)
        x2 = self.dense2(x2)
        x2 = self.conv2(x2)
        # bottom
        x3 = self.conv2(input)
        x3 = self.conv2(x3+x2)
        x3 = self.dense3(x3)
        x3 = self.conv2(x3)

        x4 = self.avgpool(x3)
        x4 = x4*x1
        x5 = x4*x3
        out = x5+input
        return out


class Network(nn.Module):
    def __init__(self, nin=3, nout=64, use_GPU=True):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nin, nout, 3, 1, 1), nn.PReLU())
        self.attention = Attention(nout)
        self.dense = Densenly(nout)
        self.conv2 = nn.Sequential(nn.Conv2d(nout, nin, 3, 1, 1), nn.PReLU())

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = x1
        for i in range(3):
            x1_1 = self.attention(x1)
            x1 = x1_1 + x1
        x1 = self.attention(x1)
        #x1 = x2 + x1
        x1 = self.dense(x1)
        x1 = self.conv2(x1)
        output = input + x1
        return output


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


model = Network(3, 64)
print_network(model)
