import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils


def calc_local_lip(model, x, xp, top_norm, bot_norm, reduction='mean'):
    '''
    Calculate local Lipschitz value, ||f(x) - f(xp)||_p / ||x - xp||_q
    This function assumes that model, x, and xp are all on same device
    :model: a callable that takes an input tensor and returns the model logits
    :x (tensor): current input point, treated as "true"
    :xp (tensor): additional input point, treated as "pred"
    :top_norm (int or 'inf', or 'kl'): norm type to apply to ||f(x) - f(xp)||
    :bot_norm (int or 'inf'): norm type to apply to ||x - xp||
    :returns Lipschitz constant for each sample (float tensor with autograd):
    '''

    # Not training model, set for inference
    model.eval()

    # Calculate difference between input samples
    # Convert all tensor dimensions after batchsize dimension into a vector - N,C,H,W to N,C*H*W
    bot = torch.flatten(x - xp, start_dim=1)
    
    # Use KL divergence to calculate the difference between model outputs, then calculate Lipschitz
    # PyTorch KLDivLoss calculates reduction(ytrue*log(ytrue/ypred)) where reduction is some method of aggregating the results (sum, mean)
    # yt*log(yt/yp) = yt*(log(yt)-log(yp)) --> PyTorch expects yp to be in logspace already, such that you input log(yp) and yt
    if top_norm == 'kl':
        criterion_kl = nn.KLDivLoss(reduction='none')
        top = criterion_kl(F.log_softmax(model(xp), dim=1), F.softmax(model(x), dim=1))
        lolip = torch.sum(top, dim=1) / torch.norm(bot + 1e-6, dim=1, p=bot_norm)
    
    # Calculate Lipschitz constant using regular norms - the top just uses output logits (no softmax)
    else:
        top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
        lolip = torch.norm(top, dim=1, p=top_norm) / torch.norm(bot + 1e-6, dim=1, p=bot_norm)
    
    # Apply some method of aggregation Lipschitz values across multiple batch samples
    if reduction == 'mean':
        return torch.mean(lolip)
    elif reduction == 'sum':
        return torch.sum(lolip)
    else:
        raise ValueError(f"Not supported reduction: {reduction}")


def maximize_local_lip(model, X, top_norm, bot_norm, batch_size=16, perturb_steps=10, alpha=0.003, eps=0.01, device="cuda"):
    '''
    Uses Projected Gradient Descent to adjust an adversarial image in order to maximize the local Lipschitz constant
    :model (function): a callable that takes an input tensor and returns the model logits
    :X (tensor): a tensor of data where the first dimension is number of samples
    :top_norm (1, 2, np.inf, or 'kl'): the norm applied to the numerator of the Lipschitz constant
    :bot_norm (1, 2, np.inf): the norm applied to the denominator of the Lipschitz constant
    :batch_size (int): the number of samples to load from the dataloader
    :perturb_steps (int): the number of steps to perform PGD to maximize local Lipschitz constant
    :alpha (float): adversarial sample step size
    :eps (float): norm constraint bound for adversarial example
    :device ('cpu' or 'cuda'): device to store/run the model and samples
    :returns average lolip across samples and maximizing adversarial samples tensor:
    '''
    if top_norm not in [1, 2, np.inf, 'kl']:
        raise ValueError(f"Unsupported norm {top_norm}")
    if bot_norm not in [1, 2, np.inf]:
        raise ValueError(f"Unsupported norm {bot_norm}")

    # Not training model, set for inference
    model.eval()
    model = model.to(device)
    
    # Make a dataset and loader using the X tensor input
    dataset = data_utils.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the running variables, total lolip and
    total_lolip = 0.
    adv_samples_list = []

    for x in loader:

        # The loader returns a list with a single tensor of size N,C,H,W - get that and push to device
        x = x[0].to(device)
        
        # Initialize x_adv as some small random offset of x
        x_adv = x + 0.001 * torch.randn(x.shape).to(device)

        for _ in range(perturb_steps):

            # Reset the gradients of the adversarial input and model
            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad()

            # Calculate the local lipschitz constant using x and x_adv, then backpropagate to get gradients
            lolip = calc_local_lip(model, x, x_adv, top_norm, bot_norm)
            lolip.backward()
            
            # Calculate the new adversarial example given the new step - gradient ascent towards higher Lipschitz
            # x_adv detaches, since x_adv.data and x_adv.grad do not carry autograd
            x_adv = x_adv.data + alpha * x_adv.grad.sign()
            
            # Determine the total dist away from the original sample x and ensure it is within norm bounds of x
            eta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = x + eta
            
            # Ensure x_adv elements within appropriate bounds
            x_adv = torch.clamp(x_adv, 0, 1.0)

        # Calculate lolip for each sample and sum batch and append the adversarial tensor to list for concat later
        total_lolip += calc_local_lip(model, x, x_adv, top_norm, bot_norm, reduction='sum').item()
        adv_samples_list.append(x_adv.detach().cpu())

    # Calculate the average lolip across all samples and concatenate the adversarial output tensor
    avg_lolip = total_lolip / len(X)
    adv_tensor = torch.concatenate(adv_samples_list, dim=0)

    return avg_lolip, adv_tensor