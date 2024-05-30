import numpy as np
import torch

def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-6):
    """Gaussian kernel with dynamic bandwidth.

    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.

    Args:
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.

    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `tf.Tensor` object of shape (N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A 'tf.Tensor` object of shape (N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.

    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx, D = list(xs.shape)[-2:]
    Ky, D2 = list(ys.shape)[-2:]
    assert D == D2

    leading_shape = list(xs.shape)[:-2]

    # Compute the pairwise distances of left and right particles
    diff = torch.unsqueeze(xs, -2) - torch.unsqueeze(ys, -3)

    dist_sq = torch.sum(torch.square(diff), dim=-1)

    # Get median
    input_shape = leading_shape + [Kx * Ky]
    values, _ = torch.topk(
        torch.reshape(dist_sq, input_shape),
        k=(Kx * Ky // 2 + 1))
    medians_sq = values[..., -1]  # ... (shape) (last element is the median)




    h = medians_sq 

    if torch.isnan(h).any():
        print(h)
        raise ValueError("NaN detected in h")

    #clamp h > 1e-6 to ensure numerical stability
    h = torch.clamp(h, min=1e-6)
    h = h.detach()
    h_expanded_twice = torch.unsqueeze(torch.unsqueeze(h, -1), -1)

    kappa = torch.exp((-dist_sq *  np.log(Kx)) / h_expanded_twice)

    # Construct the gradient
    h_expanded_thrice = torch.unsqueeze(h_expanded_twice, -1)
    kappa_expanded = torch.unsqueeze(kappa, -1)

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded

    return {"output": kappa, "gradient": kappa_grad}




def adaptive_isotropic_gaussian_kernel_stable(xs,ys, h_min = 1e-6):
    Kx, D = list(xs.shape)[-2:]
    Ky, D2 = list(ys.shape)[-2:]
    assert D == D2

    leading_shape = list(xs.shape)[:-2]

    # Compute the pairwise distances of left and right particles
    diff = torch.unsqueeze(xs, -2) - torch.unsqueeze(ys, -3)
    dist_sq = torch.sum(torch.square(diff), dim=-1)

    # Get median
    input_shape = leading_shape + [Kx * Ky]
    values, _ = torch.topk(
        torch.reshape(dist_sq, input_shape),
        k=(Kx * Ky // 2 + 1))
    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    # Ensure numerical stability
    eps = 1e-6  # Small constant to ensure numerical stability
    h = medians_sq / (torch.log(torch.tensor(Kx, dtype=torch.float32)) + eps)
    h = torch.clamp(h, min=eps)
    h = h.detach()  # Ensure h is not involved in gradient computation

    h_expanded_twice = torch.unsqueeze(torch.unsqueeze(h, -1), -1)

    kappa = torch.exp(-dist_sq / h_expanded_twice)

    # Construct the gradient
    h_expanded_thrice = torch.unsqueeze(h_expanded_twice, -1)
    kappa_expanded = torch.unsqueeze(kappa, -1)

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded

    return {"output": kappa, "gradient": kappa_grad}