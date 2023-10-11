import torch
import torch.nn as nn
import torch.nn.functional as F


class MapCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        map_channels = cfg.get('map_channels', 3)
        patch_size = cfg.get('patch_size', [100, 100])
        hdim = cfg.get('hdim', [32, 32])
        kernels = cfg.get('kernels', [3, 3])
        strides = cfg.get('strides', [3, 3])
        self.out_dim = out_dim = cfg.get('out_dim', 32)
        self.input_size = input_size = (map_channels, patch_size[0], patch_size[1])
        x_dummy = torch.randn(input_size).unsqueeze(0)

        for i, _ in enumerate(hdim):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hdim[i-1],
                                        hdim[i], kernels[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), out_dim)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class GlobalMapCNN(nn.Module):
    layer_types = {
        'conv2d': torch.nn.Conv2d,
        'maxpool': torch.nn.MaxPool2d
    }

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        cfg = {
            'map_channels': 3,
            'use_occlusion_map': True,
            'map_resolution': [400, 400],
            'output_dim': 256,
            'layers': [
                ['conv2d', 4, 7, 3],
                ['maxpool', 2, 2],
                ['conv2d', 8, 5, 2],
                ['maxpool', 2, 2],
                ['conv2d', 8, 3, 1],
                ['maxpool', 2, 2],
            ]
        }
        self.use_occlusion_map = cfg.get('use_occlusion_map', False)    # whether to add occlusion map as extra channel
        self.map_channels = cfg.get('map_channels', 3)                  # (R, G, B)
        self.input_channels = self.map_channels + int(self.use_occlusion_map)   # number of input channels
        self.map_size = cfg.get('map_resolution', [400, 400])       # resolution of the scene maps
        self.output_dim = cfg.get('output_dim', 256)                # dimension of the produced compressed state
        self.input_shape = (self.input_channels, *self.map_size)

        x_dummy = torch.randn(self.input_shape).unsqueeze(0)

        layers = cfg.get('layers')
        for layer in layers:
            layer_type = layer[0]
            layer_params = dict()
            if layer_type == 'conv2d':
                layer_params['in_channels'] = x_dummy.shape[1]
                layer_params['out_channels'] = layer[1]
                layer_params['kernel_size'] = layer[2]
                layer_params['stride'] = layer[3]
            elif layer_type == 'maxpool':
                layer_params['kernel_size'] = layer[1]
                layer_params['stride'] = layer[2]
            layer = self.layer_types[layer_type](**layer_params)
            self.layers.append(layer)
            x_dummy = layer(x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), self.output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    map_cnn = GlobalMapCNN(None)
    print(f"{map_cnn.__dict__=}")
    [print(f"{k}:\t\t{v}") for k, v in map_cnn.__dict__.items()]

    x = torch.randn([1, *map_cnn.input_shape])
    y = map_cnn(x)
    print(f"{y.shape=}")

