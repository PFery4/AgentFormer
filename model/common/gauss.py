import torch.nn as nn
import torch
import numpy as np
from typing import List
from utils.utils import initialize_weights


class GaussianDensityModel(nn.Module):
    """
    implementation of a single Gaussian Density model. The model is inherently bivariate
    (produces mixtures of Gaussian distributions over 2D space).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], forecast_dim: int = 2, activation: str = "relu"):
        super().__init__()
        assert activation in ("tanh", "sigmoid", "relu"), ValueError(f"activation type unknown: {activation}")

        self.forecast_dim = forecast_dim                                        # dimensionality of the gaussian
        self.corr_dim = (self.forecast_dim - 1) * self.forecast_dim // 2        # amount of correlation parameters
        self.N_params = (self.forecast_dim + 3) * self.forecast_dim // 2        # total amount of parameters to be estimated (mu, sig, and rho)
        self.activation = getattr(torch, activation)  # default: torch.relu

        layer_dims = [input_dim, *hidden_dims]

        self.affine_layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.affine_layers.append(nn.Linear(in_dim, out_dim))

        self.layer_mu = nn.Linear(layer_dims[-1], self.forecast_dim)
        self.layer_sig = nn.Linear(layer_dims[-1], self.forecast_dim)
        self.layer_rho = nn.Linear(layer_dims[-1], self.corr_dim)

    @staticmethod
    def corr_matrix(rho):
        N = (1 + int(np.sqrt(1 + 8 * rho.shape[-1]))) // 2
        i, j = torch.triu_indices(N, N, offset=1)

        matrix = torch.ones([*rho.shape[:-1], N, N]).to(rho.device)
        matrix[..., i, j] = rho
        matrix.transpose(-2, -1)[..., i, j] = rho
        return matrix

    @staticmethod
    def sig_matrix(sig):
        return torch.diag_embed(sig)

    @classmethod
    def covariance_matrix(cls, sig, rho):
        return cls.sig_matrix(sig) @ cls.corr_matrix(rho) @ cls.sig_matrix(sig)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        # once we have reached the last hidden layer, we connect it to each of the individual layers
        # responsible for modelling different parameters
        mu = self.layer_mu(x)
        sig = torch.exp(self.layer_sig(x))      # strictly positive
        rho = torch.tanh(self.layer_rho(x))     # within (-1, 1)

        out = torch.cat([mu, sig, rho], dim=-1)
        return out

    def separate_prediction_parameters(self, pred):
        # pred is an output of self.forward
        mu = pred[..., 0:self.forecast_dim]
        sig = pred[..., self.forecast_dim:2*self.forecast_dim]
        rho = pred[..., 2*self.forecast_dim:]
        return mu, sig, rho
