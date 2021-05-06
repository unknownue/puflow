
import torch
import torch.nn.functional as F

import numpy as np

from modules.flows.spline import searchsorted

__all__ = ['rational_linear_spline']


def rational_linear_spline(inputs,
    unnormalized_width, unnormalized_height, unnormalized_derivatives, unnormalized_lambdas,
    inverse, tails, tail_bound, num_bins,
    min_bin_width, min_bin_height, min_derivative):

    # the mask of channel that is inside tail_bound
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    # the mask of channel that is outside tail_bound
    outside_interval_mask = ~inside_interval_mask

    outputs   = torch.zeros_like(inputs)
    log_det_J = torch.zeros_like(inputs)

    if tails == 'linear':
        # add leading and tailing channel padding
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1.0)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., 1] = constant  # fill the padding with a default derivative value

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        log_det_J[outside_interval_mask] = 0.0  # Assign log_det to 0 for channel that is outside tail_bound
    else:
        raise RuntimeError(f'{tails} tails are not implemented yet.')
    
    outputs[inside_interval_mask], log_det_J[inside_interval_mask] = _rational_linear_spline(inputs[inside_interval_mask],
            unnormalized_width[inside_interval_mask],
            unnormalized_height[inside_interval_mask],
            unnormalized_derivatives[inside_interval_mask],
            unnormalized_lambdas[inside_interval_mask],
            inverse, min_bin_width, min_bin_height, min_derivative,
            num_bins, -tail_bound, tail_bound, -tail_bound, tail_bound)

    return outputs, log_det_J


def _rational_linear_spline(inputs,
    unnormalized_width, unnormalized_height, unnormalized_derivatives, unnormalized_lambdas,
    inverse, min_bin_width, min_bin_height, min_derivative,
    num_bins, left, right, bottom, top):

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_width, dim=-1)  # Normalize width
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths  # ??

    cumwidths = torch.cumsum(widths, dim=-1)  # [N, num_bins]
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)  # Add 0 pad to leading widths, [N, num_bins + 1], value from 0.0 to 1.0
    cumwidths = (right - left) * cumwidths + left  # value from left to right

    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # Length of each bin

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)  # Apply the smooth ReLU to derivative, make them all positive

    heights = F.softmax(unnormalized_height, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    lambdas = 0.95 * torch.sigmoid(unnormalized_lambdas) + 0.025  # Constraint value to [0.025, 0.975]

    lam = lambdas.gather(-1, bin_idx)[..., 0]
    wa  = 1
    wb  = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa
    wc  = (lam * wa * input_derivatives + (1 - lam) * wb * input_derivatives_plus_one) / input_delta
    ya  = input_cumheights
    yb  = input_heights + input_cumheights
    yc  = ((1 - lam) * wa * ya + lam * wb * yb) / ((1 - lam) * wa + lam * wb)

    if inverse:

        numerator = (lam * wa * (ya - inputs)) * (inputs <= yc).float() \
                +  ((wc - lam * wb) * inputs + lam * wb * yb - wc * yc) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (inputs <= yc).float()\
                    + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator / denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (wa * wc * lam * (yc - ya) * (inputs <= yc).float()\
                            + wb * wc * (1 - lam) * (yb - yc) * (inputs > yc).float()) * input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet
    else:

        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * (theta <= lam).float()\
                + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * (theta > lam).float()

        denominator = (wa * (lam - theta) + wc * theta) * (theta <= lam).float()\
                    + (wc * (1 - theta) + wb * (theta - lam)) * (theta > lam).float()

        outputs = numerator / denominator

        derivative_numerator = (wa * wc * lam * (yc - ya) * (theta <= lam).float()\
                            + wb * wc * (1 - lam) * (yb - yc) * (theta > lam).float())/input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet
