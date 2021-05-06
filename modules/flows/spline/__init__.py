
import torch


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    compare_tmp = inputs[..., None] >= bin_locations
    compare_sum = torch.sum(compare_tmp, dim=-1)
    return compare_sum - 1


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)

