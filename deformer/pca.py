import torch
import numpy as np

from torch import nn


class PCA(nn.Module):
    def __init__(self, data):
        """PCA: mean value, eigenvalues and eigenvectors (as
        columns in a matrix).

        Args:
            data: data tensor with k examples, each n-dimentional: k x n
                     matrix
        """
        super(PCA, self).__init__()

        data.require_grad = False
        with torch.no_grad():
            self.data, self.shape = self.flatten(data)

            n_examples, n_dimensions = self.data.shape[0:2]
            mean = torch.mean(self.data, 0)

            self.mean = torch.Tensor(
                mean.detach().cpu().numpy()).to(data.device).float()
            data_centered = self.data - self.mean

            cov_dual = torch.matmul(data_centered,
                                    data_centered.transpose(0, 1)) / (
                data_centered.shape[0] - 1)
            evals, evecs = torch.linalg.eigh(cov_dual)
            evecs = torch.matmul(data_centered.transpose(0, 1), evecs)
            # Normalize the col-vectors
            evecs = evecs / torch.sqrt(torch.sum(torch.square(evecs), 0))

            # Sort according to variances
            idx = torch.argsort(evals, descending=True)  # [::-1]
            evecs = evecs[:, idx]
            evals = evals[idx]

            # Remove eigenpair that should have zero eigenvalue
            diff = 1
            if n_examples > n_dimensions:
                diff = (n_examples - n_dimensions)

            self.variances = torch.Tensor(
                evals[:-diff].detach().cpu().numpy()).to(data.device).float()
            self.modes_norm = torch.Tensor(
                evecs[:, :-diff].detach().cpu().numpy()).to(data.device).float()

            # Compute the modes scaled by corresp. std. dev.
            self.modes_scaled = torch.multiply(
                self.modes_norm, torch.sqrt(self.variances)).to(data.device
                                                                ).float()

        for parameter in self.parameters():
            parameter.require_grad = False

        self.length = self.modes_norm.shape[1]
        self.weights = self.get_weights(data)

    def flatten(self, data):
        shape = data.shape
        data_flat = torch.flatten(data, start_dim=1)
        return data_flat, shape

    def unflatten(self, data, shape):
        data = data.reshape(shape)
        return data

    def forward(self, weights, num_modes=None):
        if num_modes:
            # restrict to max number of modes
            if num_modes > self.length:
                num_modes = self.length
            evecs = self.modes_norm[:, :num_modes]  # modes_norm
            weights = weights[:, :num_modes]
        else:
            evecs = self.modes_norm

        projection = self.mean + \
            torch.matmul(weights, evecs.transpose(0, 1))

        shape = [shape for shape in self.shape]
        shape[0] = len(weights)
        projection = self.unflatten(projection, shape)
        return projection

    def project_data(self, data, num_modes=None):
        """Project a data into the SSM."""
        data, shape = self.flatten(data)
        data_proj = data - self.mean
        if num_modes:
            # restrict to max number of modes
            if num_modes > self.length:
                num_modes = self.length
            evecs = self.modes_norm[:, :num_modes]
        else:
            evecs = self.modes_norm
        weights = torch.matmul(evecs.transpose(
            0, 1), data_proj.transpose(0, 1))
        data_proj = self.mean + \
            torch.matmul(weights.transpose(0, 1), evecs.transpose(0, 1))
        data_proj = self.unflatten(data_proj, shape)
        return data_proj

    def generate_random_samples(self, num_samples=1, num_modes=None):
        with torch.no_grad():
            if num_modes:
                # restrict to max number of modes
                if num_modes > self.length:
                    num_modes = self.length
                evecs = self.modes_scaled[:, :num_modes]
            else:
                evecs = self.modes_scaled
                num_modes = self.modes_scaled.shape[1]

            weights = torch.normal(0, 1, [num_samples, num_modes]).to(
                self.mean.device).float()

            samples = self.mean + \
                torch.matmul(weights, evecs.transpose(0, 1))
            shape = [shape for shape in self.shape]
            shape[0] = num_samples
            samples = self.unflatten(samples, shape)
        return samples

    def get_weights(self, data):
        data, _ = self.flatten(data)
        data_proj = data - self.mean

        # modes_norm
        return torch.matmul(self.modes_norm.transpose(
            0, 1), data_proj.transpose(0, 1)).transpose(0, 1)  # [N,#modes]

    def __len__(self):
        return self.length