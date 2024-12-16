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
