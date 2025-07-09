""" Models to estimate latent representations.
"""
import torch
import torch.nn as nn

from deformer.pca import PCA


class LatentDist(nn.Module):
    def __init__(self):
        super(LatentDist, self).__init__()

    def flatten(self, data):
        self.shape = [level.shape for level in data]
        data = torch.cat([level.flatten(start_dim=1) for level in data], dim=1)
        return data

    def unflatten(self, data, n_samples=None):
        """ Only works for grid latents.
        """
        i = 0
        latent = []
        for shape in self.shape:
            if n_samples:
                shape = list(shape)
                shape[0] = n_samples
                shape = tuple(shape)
            length = shape[-1]**3 * shape[1]
            latent.append(data[:, i: i + length].reshape(shape))
            i += length

        return latent

    def sample_list(self, data):
        """ Only works for lod grid latents.
        """
        unflattened_lat = []
        for i in range(data[0].shape[0]):
            latent_level_list = []
            for lod in data:
                latent_level_list.append(lod[i].unsqueeze(0))
            unflattened_lat.append(latent_level_list)

        return unflattened_lat

    def project_data(self, data):
        """Identity mapping"""
        return data


class LoDPCA(LatentDist):
    def __init__(self, data):
        """LoD PCA wrapper.

        Args:
            data (tensor): data tensor with k examples, each n-dimentional: k x n matrix
        """
        super(LoDPCA, self).__init__()
        self.pcas = torch.nn.ModuleList([PCA(level) for level in data])

    def project_data(self, data, num_modes=None):
        """Project a lod data into the PCA.

        Args:
            data (tensor): data tensor with k examples, each n-dimentional: k x n matrix
            num_modes (int): number of modes.
        """
        data_proj = [pca.project_data(level, num_modes)
                     for level, pca in zip(data, self.pcas)]
        return data_proj

    def generate_random_samples(self, nsamples: int = 1, num_modes=None):
        """Generate random samples from distribution

        Args:
            nsamples (int): number of samples.
            num_modes (int): number of modes.
        """
        samples = [pca.generate_random_samples(
            nsamples, num_modes) for pca in self.pcas]
        return self.sample_list(samples)

    def __len__(self):
        return len(self.pcas[-1])

    def forward(self, weights, num_modes=None):
        data_proj = [pca(weigth, num_modes)
                     for weigth, pca in zip(weights, self.pcas)]
        return data_proj

    def get_weights(self, data):
        weights = [pca.get_weigths(dat) for dat, pca in zip(data, self.pcas)]
        return weights


