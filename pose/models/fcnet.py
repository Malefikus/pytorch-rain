import torch.nn as nn
import torch.nn.functional as F

__all__ = ['fc']

class FCNet(nn.Module):
    '''Simple conv and fully connected network'''
    def __init__(self):
        super(FCNet, self).__init__()
        self.inplanes = 1
        self.planes = 3

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(4*2, 12)

    def forward(self, x):

        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x += residual

        x = self.avg_pool(x)
        out = x.view(-1)
        out = self.fc(out)

        return out


def fc():
    model = FCNet()
    return model
