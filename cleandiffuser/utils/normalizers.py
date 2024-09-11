from typing import Optional

import numpy as np
import numpy as np
import scipy.interpolate as interpolate
import pdb
import torch

from .utils import at_least_ndim


class EmptyNormalizer:
    """ Empty Normalizer

    Does nothing to the input data.
    """

    def normalize(self, x: np.ndarray):
        return x

    def unnormalize(self, x: np.ndarray):
        return x


class GaussianNormalizer(EmptyNormalizer):
    """ Gaussian Normalizer

    Normalizes data to have zero mean and unit variance.
    For those dimensions with zero variance, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> normalizer = GaussianNormalizer(x_dataset, 1)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> normalizer = GaussianNormalizer(x_dataset, 2)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(self, X: np.ndarray, start_dim: int = -1):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = np.mean(X, axis=axes)
        self.std = np.std(X, axis=axes)
        self.std[self.std == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        return (x - at_least_ndim(self.mean, ndim, 1)) / at_least_ndim(self.std, ndim, 1)

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        return x * at_least_ndim(self.std, ndim, 1) + at_least_ndim(self.mean, ndim, 1)


class MinMaxNormalizer(EmptyNormalizer):
    """ MinMax Normalizer

    Normalizes data from range [min, max] to [-1, 1].
    For those dimensions with zero range, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1
        X_max: Optional[np.ndarray],
            Maximum value for each dimension. If None, it will be calculated from X. Default: None
        X_min: Optional[np.ndarray],
            Minimum value for each dimension. If None, it will be calculated from X. Default: None

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> x_min = np.random.randn(3, 10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 1, X_min=x_min)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> x_max = np.random.randn(10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 2, X_max=x_max)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(
            self, X: np.ndarray, start_dim: int = -1,
            X_max: Optional[np.ndarray] = None, X_min: Optional[np.ndarray] = None):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.max = np.max(X, axis=axes) if X_max is None else X_max
        self.min = np.min(X, axis=axes) if X_min is None else X_min
        self.mask = np.ones_like(self.max)
        self.range = self.max - self.min
        self.mask[self.max == self.min] = 0.
        self.range[self.range == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x - at_least_ndim(self.min, ndim, 1)) / at_least_ndim(self.range, ndim, 1)
        x = x * 2 - 1
        x = x * at_least_ndim(self.mask, ndim, 1)
        return x

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x + 1) / 2
        x = x * at_least_ndim(self.mask, ndim, 1)
        x = x * at_least_ndim(self.range, ndim, 1) + at_least_ndim(self.min, ndim, 1)
        return x



# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#


class DatasetNormalizer:
    def __init__(self, dataset, normalizer):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = flatten(dataset, dataset["path_lengths"])

        self.observation_dim = dataset["observations"].shape[1]
        self.action_dim = dataset["actions"].shape[1]
        self.log_keys = ["observations", "actions", "rewards"]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            self.normalizers[key] = normalizer(val, device)

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def update_statistics(self, dataset):
        dataset = flatten(dataset, dataset["path_lengths"])
        for key, val in dataset.items():
            self.normalizers[key].update_statistics(val)

    def normalize(self, x, key):
        if isinstance(x, np.ndarray):
            return self.normalizers[key].normalize(x)
        elif isinstance(x, torch.Tensor):
            return self.normalizers[key].normalize_torch(x)

    def unnormalize(self, x, key):
        if isinstance(x, np.ndarray):
            return self.normalizers[key].unnormalize(x)
        elif isinstance(x, torch.Tensor):
            return self.normalizers[key].unnormalize_torch(x)

    def get_field_normalizers(self):
        return self.normalizers

    def get_metrics(self):
        metrics = dict()
        for key, normalizer in self.normalizers.items():
            if key not in self.log_keys:
                continue
            norm_mets = normalizer.get_metrics()
            for k, v in norm_mets.items():
                metrics[f"{key}_{k}"] = v
        return metrics


def flatten(dataset, path_lengths):
    """
    flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
        to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    """
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate(
            [x[:length] for x, length in zip(xs, path_lengths)], axis=0
        )
    return flattened


# -----------------------------------------------------------------------------#
# -------------------------- single-field normalizers -------------------------#
# -----------------------------------------------------------------------------#


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X, device):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        self.mins_torch = torch.from_numpy(self.mins).float()
        self.maxs_torch = torch.from_numpy(self.maxs).float()
        self.device = device

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def normalize_torch(self, x):
        return NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize_torch(self, x):
        return NotImplementedError()

    def update_statistics(self, dataset):
        raise NotImplementedError()


class DummyNormalizer(Normalizer):
    """
    normalizes to zero mean and unit variance
    """

    def __repr__(self):
        return f"""[ Dummy Normalizer ]\n    """

    def normalize(self, x):
        return x

    def normalize_torch(self, x):
        return x

    def unnormalize(self, x):
        return x

    def unnormalize_torch(self, x):
        return x

    def get_metrics(self):
        return {}

    def update_statistics(self, dataset):
        pass


class online_GaussianNormalizer(Normalizer):
    """
    normalizes to zero mean and unit variance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.means_torch = torch.from_numpy(self.means).float().to(self.device)
        self.stds_torch = torch.from_numpy(self.stds).float().to(self.device)
        self.z = 1

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    def normalize(self, x):
        return (x - self.means) / (self.stds + 1e-6)

    def normalize_torch(self, x):
        return (x - self.means_torch) / (self.stds_torch + 1e-6)

    def unnormalize(self, x):
        return x * (self.stds + 1e-6) + self.means

    def unnormalize_torch(self, x):
        return x * (self.stds_torch + 1e-6) + self.means_torch

    def get_metrics(self):
        metrics = {str(i) + "_mean": self.means[i] for i in range(self.means.size)}
        metrics.update({str(i) + "_std": self.stds[i] for i in range(self.stds.size)})
        return metrics

    def update_statistics(self, dataset):
        self.X = dataset.astype(np.float32)
        self.means = np.mean(self.X, axis=0)
        self.stds = np.std(self.X, axis=0)
        self.means_torch = torch.from_numpy(self.means).float().to(self.device)
        self.stds_torch = torch.from_numpy(self.stds).float().to(self.device)
        return


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -self.limit, self.limit]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = 2

    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins + 1e-6)
        x = 2 * self.limit * x - self.limit
        return x

    def normalize_torch(self, x):
        x = (x - self.mins_torch) / (self.maxs_torch - self.mins_torch + 1e-6)
        x = 2 * self.limit * x - self.limit
        return x

    def unnormalize(self, x):
        x = (x + self.limit) / (2 * self.limit)
        return x * (self.maxs - self.mins) + self.mins

    def unnormalize_torch(self, x):
        x = (x + self.limit) / (2 * self.limit)
        return x * (self.maxs_torch - self.mins_torch) + self.mins_torch

    def update_statistics(self, dataset):
        self.X = dataset.astype(np.float32)
        self.mins = np.min(self.X, axis=0)
        self.maxs = np.max(self.X, axis=0)
        self.mins_torch = torch.from_numpy(self.mins).float().to(self.device)
        self.maxs_torch = torch.from_numpy(self.maxs).float().to(self.device)
        return

    def get_metrics(self):
        metrics = {str(i) + "_min": self.mins[i] for i in range(self.mins.size)}
        metrics.update({str(i) + "_max": self.maxs[i] for i in range(self.maxs.size)})
        return metrics