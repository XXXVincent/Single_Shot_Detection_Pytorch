import torch.nn as nn
import torch
import torchvision.models as models


def vgg_backbone():
    vgg_net = models.vgg16().features
    vgg_net[30] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
    vgg_net[28] = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    add_module = nn.Sequential(
        nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
    )
    vgg_net.add_module('31', add_module)
    return vgg_net


if __name__ == '__main__':
    vgg16 = vgg_backbone()
    print(vgg16)
    x = torch.randn(size=(1, 3, 320, 192))
    print(vgg16(x).shape)
