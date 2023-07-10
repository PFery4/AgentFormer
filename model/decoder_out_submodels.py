import torch.nn as nn
from model.common.mlp import MLP
from model.common.gauss import GaussianDensityModel


def point_out_module(model_dim: int, forecast_dim: int, **kwargs) -> nn.Sequential:
    modules = []
    hidden_dims = kwargs.get("hidden_dims", None)
    if hidden_dims:
        modules.append(MLP(model_dim, hidden_dims, 'relu'))
        modules.append(nn.Linear(hidden_dims[-1], forecast_dim))
    else:
        modules.append(nn.Linear(model_dim, forecast_dim))
    return nn.Sequential(*modules)


def gauss_out_module(model_dim: int, forecast_dim: int, **kwargs) -> nn.Sequential:
    hidden_dims = kwargs.get("hidden_dims", None)
    assert hidden_dims is not None
    return nn.Sequential(GaussianDensityModel(input_dim=model_dim, hidden_dims=hidden_dims, forecast_dim=forecast_dim))


if __name__ == '__main__':
    import torch

    print("Hello")
    module = point_out_module(256, 2, hidden_dims=[128, 128])
    print(module)
    x = torch.randn(256)
    y = module(x)
    print(x.shape)
    print(y.shape)

    module_2 = point_out_module(256, 2)
    print(module_2)
    x = torch.randn(256)
    y = module_2(x)
    print(x.shape)
    print(y.shape)

    module_3 = gauss_out_module(256, 2, hidden_dims=[128, 64])
    print(module_3)
    x = torch.randn(256)
    y = module_3(x)
    print(x.shape)
    print(y.shape)

