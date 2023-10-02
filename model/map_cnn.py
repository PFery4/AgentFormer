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
            'map_channels': 4,
            'map_resolution': [400, 400],
            'output_dim': 256,
            'layers': [
                (
                    'conv2d', {
                        # 'in_channels': None,
                        'out_channels': 4,
                        'kernel_size': 7,
                        'stride': 3,
                        # 'padding': None,
                        # 'padding_mode': None,
                        # 'dilation': None,
                        # 'groups': None,
                        # 'bias': None
                    }
                ),
                (
                    'maxpool', {
                        'kernel_size': 2,
                        'stride': 2,
                        # 'padding': None,
                        # 'dilation': None,
                        # 'return_indices': None,
                        # 'ceil_mode': None
                    }
                ),
                (
                    'conv2d', {
                        'out_channels': 8,
                        'kernel_size': 5,
                        'stride': 2,
                    }
                ),
                (
                    'maxpool', {
                        'kernel_size': 2,
                        'stride': 2,
                    }
                ),
                (
                    'conv2d', {
                        'out_channels': 8,
                        'kernel_size': 3,
                        'stride': 1,
                    }
                ),
                (
                    'maxpool', {
                        'kernel_size': 2,
                        'stride': 2,
                    }
                )
            ]
        }
        self.input_channels = cfg.get('map_channels', 4)            # (R, G, B, occlusion_map)
        self.map_size = cfg.get('map_resolution', [400, 400])       # resolution of the scene maps
        self.output_dim = cfg.get('output_dim', 256)                # dimension of the produced compressed state
        self.input_size = (self.input_channels, *self.map_size)

        x_dummy = torch.randn(self.input_size).unsqueeze(0)

        layers = cfg.get('layers')
        # print(f"{layers=}")
        for layer_type, layer_params in layers:
            if layer_type == 'conv2d':
                layer_params['in_channels'] = x_dummy.shape[1]
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
    # print(f"{map_cnn.__dict__=}")
    # [print(f"{k}:\t\t{v}") for k, v in map_cnn.__dict__.items()]

    x = torch.randn([1, 4, 400, 400])
    y = map_cnn(x)
    print(f"{y.shape=}")

