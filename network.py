import torch.nn as nn
import torch
import torchvision.models as models
import math

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
    def __init__(self, num_classes, box_config, prior_setting):
        super(SSD_net, self).__init__()
        self.backbone = vgg_backbone()
        self.num_classes = num_classes
        self.multi_class_layer_no = list(map(lambda x: int(x.split('_')[-1]), box_config.keys()))
        self.multi_class_layer_no = sorted(self.multi_class_layer_no)
        self.loc_layers, self.conf_layers = multi_class_layer(num_classes, box_config, self.backbone)
        prior = PriorBox(prior_setting)
        self.priorbox = prior.forward()


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
        return {'loc': loc_output, 'conf': conf_output, 'priorbox': self.priorbox}


class PriorBox(nn.Module):
    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.config = config
        self.img_size = config['img_size']
        self.layer_shape = config['layer_shape']
        self.layer_priorbox = config['layer_priorbox']
        self.layer_aspect_ratio = config['layer_aspect_ratio']
        self.min_size = config['min_size']
        self.max_size = config['max_size']
        self.relative_feature_size = config['relative_feature_size']
        self.clip = config['clip']

    def forward(self):
        box = []
        for k, f in enumerate(self.layer_shape):
            f_k_x = self.img_size[0]/self.relative_feature_size[k][0]
            f_k_y = self.img_size[1]/self.relative_feature_size[k][1]
            for i in range(self.layer_shape[k][0]):
                for j in range(self.layer_shape[k][1]):
                    cx = (i + 0.5)/f_k_x
                    cy = (j + 0.5)/f_k_y
                    s_k_x = self.min_size[k][0]/self.img_size[0]
                    s_k_y = self.min_size[k][1]/self.img_size[1]
                    box += [cx, cy, s_k_x, s_k_y]
                    s_k_x_plus = self.max_size[k][0]/self.img_size[0]
                    s_k_y_plus = self.max_size[k][1]/self.img_size[1]
                    box += [cx, cy, s_k_x_plus, s_k_y_plus]

                    for r in self.layer_aspect_ratio[k]:
                        box += [cx, cy, s_k_x * math.sqrt(r), s_k_y/math.sqrt(r)]
                        box += [cx, cy, s_k_x / math.sqrt(r), s_k_y*math.sqrt(r)]

        boxes = torch.tensor(box).view(-1, 4)
        if self.clip:
            boxes.clamp_(max=1, min=0)

        return boxes




prior_box_config = {
    'img_size': [320, 192],
    'layer_shape': [[40, 24], (20, 12), (10, 6), (5, 3), (3, 1), (1, 1)],
    'layer_priorbox': [4, 6, 6, 6, 4, 4],
    'layer_aspect_ratio': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_size': [[30, 18], [45, 27], [67, 40], [100, 60], [125, 75], [188, 94]],
    'max_size': [[60, 36], [90, 54], [135, 81], [162, 97], [194, 116], [240, 144]],
    'relative_feature_size': [[8, 8], [16, 16], [32, 32], [64, 64], [192, 192], [320, 192]],
    'clip': True
}




if __name__ == '__main__':
    config = {
        'layer_21': 4,
        'layer_28': 6,
        'layer_34': 6,
        'layer_36': 6,
        'layer_38': 4,
        'layer_40': 4
    }
    ssd = SSD_net(num_classes=7, box_config=config, prior_setting=prior_box_config)
    x = torch.randn(size=(1, 3, 320, 192))
    dict = ssd(x)
    p_box = PriorBox(prior_box_config)
    boxes = p_box()
    print("Total box num: ", boxes.shape)

    # multi layer shape
    # torch.Size([1, 16, 40, 24])
    # torch.Size([1, 24, 20, 12])
    # torch.Size([1, 24, 10, 6])
    # torch.Size([1, 24, 5, 3])
    # torch.Size([1, 24, 3, 1])
    # torch.Size([1, 16, 1, 1])
