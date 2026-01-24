import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import pyhessian
from pyhessian.utils import group_product, get_params_grad, group_add, normalization, hessian_vector_product, orthnormal

class GeneralizedModelHessian(pyhessian.hessian):
    # Wrapper for pyhessian.hessian class, used to bypass some weird requirements
    # inside the code, like dataloader tensor count in a batch
    def __init__(self, model, criterion, dataloader, device):
        self.model = model.eval()
        self.estimate_loss = criterion

        self.data = dataloader
        self.full_dataset = True
        self.device = device

        params, gradsH = get_params_grad(self.model)
        self.params = params

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0 

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for batch in self.data:
            self.model.zero_grad()
            batch = dict(zip(["x_feats", "y_true", "masks"], [tensor.to(device) for tensor in batch]))
            tmp_num_data = batch["x_feats"].size(0)

            loss = self.estimate_loss(batch)
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

def landscape_summary(density, grids, top_eigenvalues):
    kappa = np.max(np.abs(grids))/np.min(np.abs(grids))
    epsilon = top_eigenvalues[0]
    delta = np.sum(density[grids < 0]/np.sum(density))
    print("Kappa (max abs eigenvalue / min abs eigenvalue, ravine detection):", np.round(kappa, 3))
    print("Epsilon (eigenvalue magnitude, sharpness of minimum):", np.round(epsilon, 3))
    print("Delta (density of eigenvalues < 0, saddlepoint check):", np.round(delta, 3))
    print(f"Top {len(top_eigenvalues)} max eigenvalues:", np.round(top_eigenvalues, 3))

def visualize_eigenval_distr(density, grids, **semilogy_kwargs):
    plt.semilogy(grids, density + 1.0e-7, **semilogy_kwargs)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(grids) - 1, np.max(grids) + 1, None, None])
    plt.tight_layout()
    plt.grid()
    plt.show();

def get_loss_slice_vector(model, direction_mode, top_eigenvector = None, loss_fn = None, batch_dict = None, direction = None):
    controlled_param_list = list(filter(lambda param: param.requires_grad, model.parameters()))
    if "top_eigenvector" in direction_mode:
        magnitude_order = int(direction_mode.split("-")[-1])
        direction = top_eigenvector[magnitude_order]
    elif direction_mode == "gradient":
        loss = loss_fn(batch_dict, model)
        loss.backward()
        direction = [p.grad.data if p.grad is not None else 0.0 for p in controlled_param_list]
        model.zero_grad()
    elif direction_mode == "manual":
        direction = direction
    elif direction_mode == "random":
        direction = [torch.randn_like(p) for p in controlled_param_list]
    else:
        direction = [torch.randn_like(p) for p in controlled_param_list]
    direction = normalization(direction)
    return direction

def perturb_params(model_orig, model_perb, direction, alpha):
    model_orig_params = list(filter(lambda param: param.requires_grad, model_orig.parameters()))
    model_perb_params = list(filter(lambda param: param.requires_grad, model_perb.parameters()))
    for m_orig, m_perb, d in zip(model_orig_params, model_perb_params, direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def visualize_loss_slice(batch_dict, model, model_copy, direction, direction_mode,
                         loss_fn, lower_bound = -1.5, upper_bound = 1.5, steps = 51, **lineplot_kwargs):
    lams = np.linspace(lower_bound, upper_bound, steps).astype(np.float32)
    loss_list = np.zeros((steps, ))
    for i, lam in enumerate(lams):
        model_perb = perturb_params(model, model_copy, direction, lam)
        loss_list[i] = loss_fn(batch_dict, model_perb).item()
    
    plt.plot(lams, loss_list, **lineplot_kwargs)
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title(f'Slice of Loss Landscape, mode {direction_mode}')
    plt.grid();
