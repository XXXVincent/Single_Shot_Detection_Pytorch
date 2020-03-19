import torch.nn as nn
import torch
import torchvision.models as models





config = [4, 6, 6, 6, 4, 4]
vgg_ = [21, 24, 32, 34, 36, 38]
num_classes = 7


def vgg_backbone():
    vgg_net = models.vgg16().features
    vgg_net[16] = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False)
    vgg_net[-1] = nn.MaxPool2d(3, 1, 1, 1, ceil_mode=False)

    add_modules = [
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
    ]
    for i in range(len(add_modules)):
        vgg_net.add_module(str(len(vgg_net)), add_modules[i])
    return vgg_net


def multi_class_layer(num_classes, config, vgg_net):
    loc_layers = {}
    conf_layers = {}
    for layer_no, box_num in config.items():
        layer_num_int = int(layer_no.split('_')[-1])
        layer_out_channel = vgg_net[layer_num_int].out_channels
        min_kernel_size = min(vgg_net[layer_num_int].kernel_size)
        loc_layer = nn.Conv2d(layer_out_channel, box_num * 4, kernel_size=min_kernel_size,
                              padding=1 if min_kernel_size == 3 else 0)
        conf_layer = nn.Conv2d(layer_out_channel, box_num * num_classes, kernel_size=min_kernel_size,
                               padding=1 if min_kernel_size == 3 else 0)
        loc_layers[str(layer_num_int)] = loc_layer
        conf_layers[str(layer_num_int)] = conf_layer
    return loc_layers, conf_layers


class SSD_net(nn.Module):
    def __init__(self, num_classes, box_config):
        super(SSD_net, self).__init__()
        self.backbone = vgg_backbone()
        self.num_classes = num_classes
        self.multi_class_layer_no = list(map(lambda x: int(x.split('_')[-1]), box_config.keys()))
        self.multi_class_layer_no = sorted(self.multi_class_layer_no)
        self.loc_layers, self.conf_layers = multi_class_layer(num_classes, box_config, self.backbone)

    def forward(self, input):
        loc_output_dict = {}
        conf_output_dict = {}
        current_multi_class_layer_idx = 0
        for i in range(len(self.backbone)):
            if i == 0:
                x = self.backbone[i](input)
            else:
                x = self.backbone[i](x)

            if i == self.multi_class_layer_no[current_multi_class_layer_idx]:
                loc = self.loc_layers[str(i)](x)
                conf = self.conf_layers[str(i)](x)
                loc_output_dict[str(i)] = loc
                conf_output_dict[str(i)] = conf
                current_multi_class_layer_idx += 1
        loc_output = torch.cat([loc_pred.view(loc_pred.size(0), -1, 4) for loc_pred in loc_output_dict.values()], 1)
        conf_output = torch.cat([conf_pred.view(conf_pred.size(0), -1, self.num_classes,)
                                 for conf_pred in conf_output_dict.values()], 1)
        del loc_output_dict, conf_output_dict
        return {'loc': loc_output, 'conf': conf_output}


    # # @property
    # def multi_class_layer(self, num_classes, config, vgg_net):
    #     loc_layers = {}
    #     conf_layers = {}
    #     for layer_no, box_num in config.items():
    #         layer_num_int = int(layer_no.split('_')[-1])
    #         layer_out_channel = vgg_net[layer_num_int].out_channels
    #         min_kernel_size = min(vgg_net[layer_num_int].kernel_size)
    #         loc_layer = nn.Conv2d(layer_out_channel, box_num * 4, kernel_size=min_kernel_size,
    #                               padding=1 if min_kernel_size == 3 else 0)
    #         conf_layer = nn.Conv2d(layer_out_channel, box_num * num_classes, kernel_size=min_kernel_size,
    #                                padding=1 if min_kernel_size == 3 else 0)
    #         loc_layers[str(layer_num_int)] = loc_layer
    #         conf_layers[str(layer_num_int)] = conf_layer
    #     return loc_layers, conf_layers
    #
    # # @property
    # def vgg_backbone(self):
    #     vgg_net = models.vgg16().features
    #     vgg_net[16] = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False)
    #     vgg_net[-1] = nn.MaxPool2d(3, 1, 1, 1, ceil_mode=False)
    #
    #     add_modules = [
    #         nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
    #         nn.Conv2d(1024, 1024, kernel_size=1),
    #         nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
    #         nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #         nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
    #         nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #         nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
    #         nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
    #         nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
    #         nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
    #     ]
    #     for i in range(len(add_modules)):
    #         vgg_net.add_module(str(len(vgg_net)), add_modules[i])
    #     return vgg_net


if __name__ == '__main__':
    config = {
        'layer_21': 4,
        'layer_28': 6,
        'layer_34': 6,
        'layer_36': 6,
        'layer_38': 6,
        'layer_40': 4
    }
    ssd = SSD_net(num_classes=7, box_config=config)
    x = torch.randn(size=(1, 3, 320, 192))
    ssd(x)
