"""The Projected Gradient Descent attack."""

import numpy as np
import torch
import torch.utils.data as data_utils

def scale_tens(tens, tens_idx, norm, eps):
    # If ||tens||_p > eps, scales all tens values such that ||tens||_p = eps
    # This function follows the torch.nn.utils.clip_grad implementation of norm scaling
    # :tens (tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float or int): the maximum allowable norm value of tens
    # :returns: scaled_tens
    
    # Get the norm of tens across the specified indices
    tens_norm = torch.norm(tens, dim=tens_idx, keepdim=True, p=norm)
    
    # If eps > eta_norm in certain elements, don't upscale them
    scale_coef = torch.clamp(eps / (tens_norm + 1e-6), max=1.0)
    scaled_tens = tens * scale_coef
    
    return scaled_tens

def clip_tens(tens, tens_idx, norm, eps):
    # If ||tens||_inf > eps, projects tens tensor to the nearest point where ||tens||_inf = eps
    # :tens (Tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float): the maximum allowable inf norm value of tens
    # :returns clipped_tens (Tensor): 
    
    # Inf norm of tens cannot exceed eps - corresponds to clipping all values of tens beyond +/- eps
    if norm == np.inf:
        clipped_tens = torch.clamp(tens, min=-eps, max=eps)
    
    # Scaling isn't the same idea as clipping, but in practice, people will refer to scaling as clipping
    # I keep this part to ensure the function still works if people want to "clip" for other norms
    # Note that in 2-norm, clipping and scaling are identical processes
    else:
        clipped_tens = scale_tens(tens, tens_idx, norm, eps)
    
    return clipped_tens

def optimize_linear(grad, grad_idx, norm=np.inf):
    # Solves for the optimal input to a linear function under a norm constraint.
    # Optimal_perturbation = argmax_eta,||eta||_p<=1(dot(eta, grad))
    # i.e., Find eta s.t. pth norm of eta <= 1 such that dot(eta, grad) is maximized
    # :grad (Tensor): batch of gradients
    # :grad_idx (list): the list of grad dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (number): np.inf, 1, or 2. Order of norm constraint.
    # :returns eta (Tensor): optimal perturbation, the eta where ||eta||_p <= 1
    
    # dot(eta, grad) with ||eta||inf = max(abs(eta)) <= 1 is maximized when eta=sign(grad)
    # Optimal inf-norm constrained perturbation direction is the max magnitude grad value in every dimension
    if norm == np.inf:
        eta = torch.sign(grad)
    
    # dot(eta, grad) with ||eta||1 = sum(abs(eta_i)) <= 1 is maximized when eta is a +/- 1-hot corresponding to the maximum magnitude value of grad
    # Optimal 1-norm constrained perturbation direction is the max magnitude pixel value in any dimension
    elif norm == 1:
    
        # Absolute value and sign tensors of gradient, used later
        abs_grad = torch.abs(grad)

        # Get the maximum values of the tensor as well as their locations
        # Max is executed across all dimensions except the first (batch dim), such that one max value is found for each batch sample
        max_abs_grad = torch.amax(abs_grad, dim=grad_idx, keepdim=True)
        max_mask = abs_grad.eq(max_abs_grad).to(torch.float)
        
        # Count the number of tied values at maximum for each batch sample
        num_ties = torch.sum(max_mask, dim=grad_idx, keepdim=True)
        
        # Optimal perturbation for 1-norm is only along the maximum magnitude dimensions
        # Hack to apply smaller perturbation in >1 dimensions if num_ties > 1
        eta = torch.sign(grad) * max_mask / num_ties
    
    # dot(eta, grad) with ||eta||2 = sqrt(sum(eta_i^2)) <= 1 is maximized when eta is equal to the normalized gradient vector
    # Optimal 2-norm constrained perturbation direction is Euclidean scaling along the gradient vector
    elif norm == 2:
    
        # Get the 2 norm of the gradient vector for each batch sample and normalize
        eta = grad / torch.norm(grad, dim=grad_idx, keepdim=True, p=2)
        
    else:
        raise NotImplementedError('Only L-inf, L1 and L2 norms are currently implemented.')

    return eta


def fast_gradient_method(model_fn, x, alpha, norm, loss_fn=None, x_min=None, x_max=None, y=None, targeted=False):
    """
    Straightforward implementation of the Fast Gradient Method.
    This function assumes that model_fn and x are on same device
    :model_fn: a callable that takes an input tensor and returns the model logits.
    :x (float tensor): input tensor for model.
    :alpha (float): input variation parameter, see https://arxiv.org/abs/1412.6572.
    :norm (1, 2, or np.inf): Order of the norm.
    :loss_fn:
    :x_min (float, int) (optional): minimum value of x or x_adv (useful for ensuring images are useable)
    :x_max (float, int) (optional): maximum value of x or x_adv (useful for ensuring images are useable)
    :y: (optional) Tensor with true labels. If targeted is true, then provide target label. Otherwise, only
        provide this parameter if you'd like to use true labels when crafting adversarial samples. Otherwise, model
        predictions are used as labels to avoid the "label leaking" effect (explained in this paper:
        https://arxiv.org/abs/1611.01236). Default is None.
    :targeted (bool) (optional): Is the attack targeted or untargeted? Default is False.
            Untargeted will try to make the label incorrect away from true label
            Targeted will try to make the label incorrect towards target label
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to 0, got {} instead".format(alpha))
    if alpha == 0:
        return x
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    if x_min is not None and x_max is not None:
        if x_min > x_max:
            raise ValueError('x_max must be greater than x_min')

    # Not training model, set for inference
    model_fn.eval()

    # Copy input tensor, ensure it is detached, convert to float tensor and allow storing grad for backprop
    # .detach().requires_grad_(True) will clear and reset any previous gradients on x
    x = x.clone().detach().to(torch.float).requires_grad_(True)

    # Ensure x elements within appropriate bounds, then have to redo detach and requires_grad
    if x_min is not None or x_max is not None:
        x = torch.clamp(x, x_min, x_max)
        x = x.detach().requires_grad_(True)

    # Zero gradients of the model
    model_fn.zero_grad()

    # If target y is not specified, use model predictions as ground truth to avoid label leaking
    # Additionally set targeted False since we want to maximize the loss of ground truth label
    if y is None:
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)
        targeted = False

    # Calculate loss
    # If attack is targeted, define loss such that gradient dL/dx will point towards target class
    # else, define loss such that gradient dL/dx will point away from correct class
    loss = loss_fn(model_fn(x), y)
    if targeted:
        loss = -loss

    # Get loss gradient wrt input
    # eta is the norm-constrained direction that maximizes dot(perturbation, x.grad)
    # eta is detached, since x.grad does not carry autograd
    loss.backward()
    eta = optimize_linear(x.grad, list(range(1, len(x.grad.size()))), norm)

    # Add perturbation to original example to step away from correct label and obtain adversarial example
    # x_adv is detached, since x.data does not carry autograd
    x_adv = x.data + alpha * eta

    # Ensure x_adv elements within appropriate bounds - x_adv.requires_grad is False, no need to do detach after clamp
    if x_min is not None or x_max is not None:
        x_adv = torch.clamp(x_adv, x_min, x_max)

    return eta, x_adv

def projected_gradient_descent(model_fn, x, alpha, eps, nb_iter, norm, loss_fn=None, x_min=None, x_max=None,
                               y=None, targeted=False, rand_init=True, noise_minmax=None):
    """
    Implementation the Kurakin 2016 Basic Iterative Method (rand_init=False) or Madry 2017 PGD method (rand_init=True)
    This function assumes that model_fn and x are on same device
    :model_fn: a callable that takes an input tensor and returns the model logits.
    :x (float tensor): input tensor for model.
    :alpha (float): input variation parameter, see https://arxiv.org/abs/1412.6572.
    :eps (float): norm constraint bound for adversarial example
    :nb_iter (int): Number of attack iterations.
    :norm (1, 2, or np.inf): Order of the norm (mimics NumPy).
    :loss_fn:
    :x_min (float, int) (optional): minimum value of x or x_adv (useful for ensuring images are useable)
    :x_max (float, int) (optional): maximum value of x or x_adv (useful for ensuring images are useable)
    :y: (optional) Tensor with true labels. If targeted is true, then provide target label. Otherwise, only
        provide this parameter if you'd like to use true labels when crafting adversarial samples. Otherwise, model
        predictions are used as labels to avoid the "label leaking" effect (explained in this paper:
        https://arxiv.org/abs/1611.01236). Default is None.
    :targeted (bool) (optional): Is the attack targeted or untargeted? Default is False.
            Untargeted will try to make the label incorrect away from true label
            Targeted will try to make the label incorrect towards target label
    :rand_init (bool) (optional): Whether to start the attack from a randomly perturbed x.
    :noise_minmax (bool) (optional): Support of the continuous uniform distribution from which the random perturbation
        on x was drawn. Effective only when rand_init is True. Default equals to eps.
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if norm == 1:
        print('Warning: For PGD with norm=1, FGM may not be a good inner loop step, because norm=1 FGM only changes 1 pixel at a time')
    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to 0, got {} instead".format(alpha))
    if alpha == 0:
        return x
    if x_min is not None and x_max is not None:
        if x_min > x_max:
            raise ValueError('x_max must be greater than x_min')

    # Not training model, set for inference
    model_fn.eval()

    # Throughout this function, no variables carry autograd
    # In Fast_Gradient_Method, the input tensor will convert to carry autograd, but the output does not

    # Copy input tensor, ensure it is detached, convert to float tensor and ensure it will not store gradients
    x = x.clone().detach().to(torch.float).requires_grad_(False)

    # Apply random initial perturbation to input (or don't)
    # Madry 2017 apply random perturbations over many runs to see if adv results were very different - they were not
    if rand_init:
        if noise_minmax is None:
            noise_minmax = eps
        noise = torch.zeros_like(x).uniform_(-noise_minmax, noise_minmax)
        # Clip noise to ensure it does not violate x_adv norm constraint and then apply to x
        noise = clip_tens(noise, list(range(1, len(noise.size()))), norm, eps)
        x_adv = x + noise
    else:
        x_adv = x
    
    # Ensure x_adv elements within appropriate bounds
    if x_min is not None or x_max is not None:
        x_adv = torch.clamp(x_adv, x_min, x_max)

    # If target y is not specified, use model predictions as ground truth to avoid label leaking
    # Additionally set targeted False since we want to maximize the loss of ground truth label
    if y is None:
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)
        targeted = False
  
    # Run FGM iteratively
    for _ in range(nb_iter):
    
        # Get new adversarial tensor
        _, x_adv = fast_gradient_method(model_fn, x_adv, alpha, norm, loss_fn=loss_fn, x_min=x_min, x_max=x_max, y=y, targeted=targeted)

        # Clip total perturbation (measured from center x) to norm ball associated with x_adv limit
        # Then recreate x_adv using the clipped perturbation
        eta = x_adv - x
        eta = clip_tens(eta, list(range(1, len(eta.size()))), norm, eps)
        x_adv = x + eta
    
        # Ensure x_adv elements within appropriate bounds
        if x_min is not None or x_max is not None:
            x_adv = torch.clamp(x_adv, x_min, x_max)
    
    return (x_adv - x), x_adv
