import torch.nn as nn
import torch
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

        self.forecast_dim = forecast_dim
        self.activation = getattr(torch, activation)  # default: torch.relu

        layer_dims = [input_dim, *hidden_dims]

        self.affine_layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.affine_layers.append(nn.Linear(in_dim, out_dim))

        self.layer_mu = nn.Linear(layer_dims[-1], self.forecast_dim)
        self.layer_sig = nn.Linear(layer_dims[-1], self.forecast_dim)
        self.layer_rho = nn.Linear(layer_dims[-1], (self.forecast_dim - 1) * self.forecast_dim // 2)

    def corr_matrix(self, rho):
        i, j = torch.triu_indices(self.forecast_dim, self.forecast_dim, offset=1)

        matrix = torch.ones([*rho.shape[:-1], self.forecast_dim, self.forecast_dim])
        matrix[..., i, j] = rho
        matrix.transpose(-2, -1)[..., i, j] = rho
        return matrix

    @staticmethod
    def sig_matrix(sig):
        return torch.diag_embed(sig)

    def covariance_matrix(self, sig, rho):
        print(f"{sig.shape=}")
        print(f"{rho.shape=}")
        print(f"{self.sig_matrix(sig).shape=}")
        print(f"{self.corr_matrix(rho).shape=}")
        return self.sig_matrix(sig) @ self.corr_matrix(rho) @ self.sig_matrix(sig)

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


if __name__ == '__main__':

    from torch.distributions import MultivariateNormal
    from tqdm import tqdm
    from model.agentformer_loss import gaussian_twodee_nll, gaussian_twodee_nll_2, multivariate_gaussian_nll

    forecast_dim = 2
    dist_params = {
        "mu": torch.randn(forecast_dim),
        "sig": torch.exp(torch.randn(forecast_dim)),
        "rho": torch.tanh(torch.randn((forecast_dim - 1) * forecast_dim // 2))
    }
    tensor_shape = [30, 12]  # epochs, batches, datablobs...
    model_input_dim = 256
    model_hidden_dims = [128, 64]
    model_activation = "relu"
    n_training_steps = 1000


    gauss = GaussianDensityModel(
        input_dim=model_input_dim, hidden_dims=model_hidden_dims, forecast_dim=forecast_dim, activation=model_activation
    )
    initialize_weights(gauss.modules())
    optimizer = torch.optim.Adam(gauss.parameters())

    gt_mu = dist_params["mu"]
    gt_Sig = gauss.covariance_matrix(sig=dist_params["sig"], rho=dist_params["rho"])
    print(f"{gt_mu=}")
    print(f"{gt_Sig=}")

    distrib = MultivariateNormal(loc=gt_mu, covariance_matrix=gt_Sig)

    for i in tqdm(range(n_training_steps)):
        gauss.zero_grad()
        optimizer.zero_grad()

        y = distrib.sample(sample_shape=tensor_shape)
        x = torch.randn([*tensor_shape, model_input_dim])

        pred = gauss(x)
        mu, sig, rho = gauss.separate_prediction_parameters(pred)
        cov = gauss.covariance_matrix(sig, rho)

        # loss = gaussian_twodee_nll(mu=mu, sig=sig, rho=rho, targets=y)
        # loss = gaussian_twodee_nll_2(mu=mu, sig=sig, rho=rho, targets=y)
        loss = multivariate_gaussian_nll(mu=mu, Sig=cov, targets=y)

        loss.backward()

        optimizer.step()

        assert torch.all(sig >= 0.0)
        assert torch.all(-1.0 <= rho) and torch.all(rho <= 1.0)

    print("\"True\" Distribution parameters:")
    [print(f"{k}: {v}") for k, v in dist_params.items()]

    print("Mean of Generated distributions:")
    x = torch.randn([*tensor_shape, 256])
    pred = gauss(x)
    mu, sig, rho = gauss.separate_prediction_parameters(pred)
    print(f"{torch.mean(mu[..., 0])=}")
    print(f"{torch.mean(mu[..., 1])=}")
    print(f"{torch.mean(sig[..., 0])=}")
    print(f"{torch.mean(sig[..., 1])=}")
    print(f"{torch.mean(rho[..., 0])=}")
