import os
import inspect
from collections import OrderedDict

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity, record_function

try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass

import plm_surrogate.commons as commons

### ADAPTATIONS FOR BAYESIAN INFERENCE
from types import SimpleNamespace
class ModelParameter(nn.Module):
    def __init__(self, name, shape, model_type = "linear_laplace", init_dict = {}, 
                 lower_upper = (-0.01, 0.01), prior = None):
        super().__init__()
        self.name = name
        self.shape = shape
        self.model_type = model_type
        
        self.mu = self.create_param(name, init_dict, lower_upper)
        if model_type in ["mean_field", "linear_laplace"]:
            if model_type == "mean_field":
                self.log_sigma = self.create_param(name+"_sigma", init_dict, (-5, -4))
            if prior is None:
                prior = (0*self.mu, 0.541325+0*self.mu)
            self.prior_mu = nn.Parameter(prior[0], requires_grad = False)
            self.prior_sigma = nn.Parameter(prior[1], requires_grad = False)
        
    def create_param(self, name, init_dict, lower_upper, verbose = False):
        shape = self.shape
        found_variable = init_dict.get(name, False)
        if found_variable is False:
            found_variable = np.random.uniform(lower_upper[0], lower_upper[1], shape)
            if verbose:
                print(f"WARNING: Parameter {name} not found in init_dict!")
                print(f"Initializing parameter {name} from scratch:")
                print(f"np.random.uniform({lower_upper[0]}, {lower_upper[1]})")
        
        try:
            found_variable = np.squeeze(found_variable).reshape(shape)
        except ValueError:
            if verbose:
                print("-"*70)
                print(f"WARNING: Parameter {name} has a shape mismatch:")
                print(f"{name}.shape = {found_variable.shape} vs parameter shape={shape}")
                print(f"Initializing parameter {name} from scratch:")
                print(f"np.random.uniform({lower_upper[0]}, {lower_upper[1]})")
            found_variable = np.random.uniform(lower_upper[0], lower_upper[1], shape)
            
        return nn.Parameter(torch.FloatTensor(found_variable), requires_grad = True)

    def sample_param(self):
        shape = self.shape
        bayesian = self.model_type == "mean_field"
        
        if bayesian:
            non_shifted = torch.randn(shape)
            log_sigma = self.log_sigma
            sigma = F.softplus(log_sigma)
            sample_w = non_shifted*sigma + self.mu

        else:
            sample_w = 0.0 + self.mu

        return sample_w

def create_parameter_list(names, shapes, model_type, init_dict):
    parameter_list = nn.ModuleList()
    for name, shape in zip(names, shapes):
        model_param = ModelParameter(name, shape, model_type, init_dict)
        parameter_list.add_module(name, model_param)
    return parameter_list

def unpack_parameter_list(parameter_list):
    local_params = SimpleNamespace()
    for param in parameter_list:
        local_params.__dict__[param.name] = param.sample_param()
    return local_params

def gaussian_kernel(pos_window, kernel_amplitude, kernel_mean, inv_kernel_var, kernel_period, kernel_phase, simplify_kernel = False):
    local_pos_window = pos_window.clone()
    if simplify_kernel:
        edge_size = pos_window.shape[0]//2
        std_dev = 1/(inv_kernel_var+1e-12)**0.5
        left_border = torch.min(torch.floor(kernel_mean - 3*std_dev + edge_size)).item()
        right_border = torch.max(torch.ceil(kernel_mean + 3*std_dev + edge_size)).item()
        left_border = max(int(left_border), 0)
        right_border = min(int(right_border), pos_window.shape[0])+1
        local_pos_window = local_pos_window[left_border:right_border]
    cos_part = torch.cos(kernel_period*local_pos_window + kernel_phase)
    exp_part = torch.exp(-inv_kernel_var*(local_pos_window-kernel_mean)**2)
    gaussian_filter = cos_part*exp_part*kernel_amplitude
    return gaussian_filter

kl_div_fn = lambda mu, sigma, mu_p, sigma_p: torch.log(sigma_p)-torch.log(sigma)+(sigma**2+(mu-mu_p)**2)/(2*sigma_p**2)-1/2
log_prior_fn = lambda mu, sigma, mu_p, sigma_p: 0.5*((mu-mu_p)**2)*(1/sigma_p**2)

### MODEL MODULE OBJECTS
class SharedVariableModule(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, module_count = 0, 
                 n_classes = 1, init_dict = {}):
        super().__init__()
        
        names = ["features2class", "conv_ampl_1"]
        shapes = [(1, n_features, hidden_state_dim), (1, n_features, hidden_state_dim)]
        if module_count > 0:
            names = names + [f"output_projection_{i}" for i in range(module_count)] + [f"latent_projection_{i}" for i in range(module_count)]
            shapes = shapes + [(1, hidden_state_dim, n_classes)]*module_count + [(1, hidden_state_dim, hidden_state_dim)]*module_count
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)

    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        for key, value in lp.__dict__.items():
            input_dict[key] = value
            
class InputOutputEmbeddingModule(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, cut_features = 0, 
                 additional_features = 0, init_dict = {}):
        super().__init__()
        n_features -= additional_features

        self.cut_features = cut_features
        self.additional_features = additional_features
        
        names = ["bos_emb", "eos_emb", "unk_emb", "rotate_embeds", 
                 "bias_embeds", "nat_embed", "class_unk_emb", 
                 "class_bos_emb", "class_eos_emb", "class_freq_emb"]
        shapes = [(1, 1, n_features), (1, 1, n_features), (1, 1, n_features), (1, n_features, n_features),
                  (1, 1, n_features), (1, len(commons.aa_alphabet)-1, n_features), (1, 1, hidden_state_dim),
                  (1, 1, hidden_state_dim), (1, 1, hidden_state_dim), (1, 1, hidden_state_dim)]
        if additional_features > 0:
            names.append("add_embed"), shapes.append((2, 1, additional_features))
            names.append("lda_class_map"), shapes.append((1, additional_features + cut_features, hidden_state_dim))
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        x_onehot, masks = input_dict["x"], input_dict["masks"]

        orig_embed_full = x_onehot[..., :20]@lp.nat_embed@lp.rotate_embeds + lp.bias_embeds
        special_tokens_embeds = x_onehot[..., 20:]@torch.cat([lp.unk_emb, lp.bos_emb, lp.eos_emb], dim = 1)
        orig_embed_full = orig_embed_full + special_tokens_embeds
        if self.additional_features > 0:
            mean_add_feats, std_add_feats = lp.add_embed[:1], lp.add_embed[1:]
            orig_embed_other = (input_dict["x_other"][..., self.cut_features:]-mean_add_feats)*std_add_feats
            orig_embed_full = torch.cat([orig_embed_full, orig_embed_other], dim = -1)
        
        input_dict["x_embeds"] = orig_embed_full

        special_tokens_logits = x_onehot[..., 20:]@torch.cat([lp.class_unk_emb, lp.class_bos_emb, lp.class_eos_emb], dim = 1)
        class_logits = special_tokens_logits + lp.class_freq_emb*masks
        if self.additional_features > 0:
            class_logits = class_logits + input_dict["x_other"]@lp.lda_class_map
        return class_logits

class PhysioChemical2Class(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, max_std_base, init_dict = {}):
        super().__init__()
        
        self.init_dict = {"max_std_base":max_std_base}
        names = ["mean_dist_attn", "std_dist_attn", "period_cos", 
                 "phase_cos", "w_bias"]
        shapes = [(1, n_features, hidden_state_dim), (1, n_features, hidden_state_dim),
                  (1, n_features, hidden_state_dim), (1, n_features, hidden_state_dim),
                  (1, n_features, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        x_onehot, orig_embed_full, pos_embed = input_dict["x"], input_dict["x_embeds"], input_dict["pos_embed"]
        masks, features2class = input_dict["masks"], input_dict["features2class"]
        
        std_dist_attn = lp.std_dist_attn**2 + 1/self.init_dict["max_std_base"]**2
        conv_w = gaussian_kernel(pos_embed, features2class, lp.mean_dist_attn, std_dist_attn, lp.period_cos, lp.phase_cos)
        dist_filt = F.conv1d(input = orig_embed_full.clone().permute(0, 2, 1), 
                             weight = conv_w.permute(2, 1, 0), 
                             stride = 1, padding = "same").permute(0, 2, 1)
        
        axis_two_mask = masks.clone() #B, L, 1
        collect_bos_eos = torch.sum(x_onehot[..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+collect_bos_eos #B, L, 1
        masked_filt = torch.sum(orig_embed_full.clone()*axis_one_mask, dim = 1, keepdim = True) #B, 1, 15
        masked_filt = masked_filt*axis_two_mask #B, L, 15
        #B, L, 15 -> B, L, 9
        w_bias_dist_filt = masked_filt@(features2class*lp.w_bias)
        
        dist_filt = w_bias_dist_filt + dist_filt
        class_logits = dist_filt*masks
        return class_logits

class InterClassBiasCorrection(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, init_dict = {}):
        super().__init__()
        
        names = ["seq_w_bias"]
        shapes = [(1, n_features, 1)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
    
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        orig_embed_full, masks, features2class = input_dict["x_embeds"],  input_dict["masks"], input_dict["features2class"]
        
        axis_two_mask = torch.reshape(masks.clone(), masks.shape+(1, 1))
        collect_bos_eos = torch.sum(input_dict["x"].clone()[..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)

        seq_w_base = orig_embed_full@lp.seq_w_bias
        seq_w_base = torch.reshape(seq_w_base, seq_w_base.shape+(1, 1))
        seq_to_classes = torch.unsqueeze(torch.unsqueeze(orig_embed_full.clone()@features2class, dim = -2), dim = -2)
        seq_w_base = seq_to_classes*seq_w_base
        
        masked_part_one = torch.transpose(axis_one_mask[..., 0, 0], 1, 2)@seq_w_base[:, :, 0, 0]
        class_logits = (axis_two_mask[..., 0, 0]@masked_part_one)*masks
        return class_logits
        
class SolubilityModule(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, starting_period, max_std_pos_conv, init_dict = {}):
        super().__init__()
        
        self.init_dict = {"starting_period":starting_period, 
                          "max_std_pos_conv":max_std_pos_conv}
        for key, value in init_dict.items():
            self.init_dict[key] = value
        names = ["true_period", "solvent_access_w", "conv_ampl_0", 
                 "period_conv", "phase_conv", "mean_pos_conv", 
                 "std_pos_conv", "converge_w_bias"]
        shapes = [(1, 1, hidden_state_dim), (1, n_features, 1), (1, n_features, 1), 
                  (1, 1, hidden_state_dim),  (1, 1, hidden_state_dim), (1, 1, hidden_state_dim), 
                  (1, 1, hidden_state_dim), (1, 1, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        orig_embed_full, masks = input_dict["x_embeds"], input_dict["masks"]
        pos_embed, conv_ampl_1 = input_dict["pos_embed"], input_dict["conv_ampl_1"]

        max_std_pos_conv = self.init_dict["max_std_pos_conv"]
        std_pos_conv = lp.std_pos_conv**2 + 1/max_std_pos_conv**2
        period_conv = self.precompute_period_conv(input_dict)

        conv_ampl_filter = lp.conv_ampl_0 + lp.solvent_access_w
        conv_pairwise_0 = gaussian_kernel(pos_embed, 1, lp.mean_pos_conv, std_pos_conv, period_conv, lp.phase_conv)
        conv_pairwise_0 = conv_ampl_filter@conv_pairwise_0
        local_part_0 = F.conv1d(input = orig_embed_full.clone().permute(0, 2, 1), 
                                weight = conv_pairwise_0.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        local_part_1 = orig_embed_full@conv_ampl_1
        class_logits = (local_part_0+lp.converge_w_bias)*local_part_1*masks
        return class_logits

    def precompute_period_conv(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        true_period = lp.true_period
        true_period = 2*torch.pi/true_period
        fixed_periods = F.sigmoid(true_period*100)
        
        starting_period = self.init_dict["starting_period"]
        period_conv = lp.period_conv + starting_period
        
        period_conv = true_period*fixed_periods + period_conv*(1-fixed_periods)
        return period_conv
        
class AggregateNeighbourComparison(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, max_std2conv, init_dict = {}):
        super().__init__()
        
        self.init_dict = {"max_std2conv":max_std2conv}
        for key, value in init_dict.items():
            self.init_dict[key] = value
        names = ["conv_ampl_2", "std2conv", "converge_w_bias_2"]
        shapes = [(1, n_features, hidden_state_dim), (1, 1, hidden_state_dim), (1, 1, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        orig_embed_full, pos_embed = input_dict["x_embeds"], input_dict["pos_embed"]
        masks, conv_ampl_1 = input_dict["masks"], input_dict["conv_ampl_1"]

        max_std2conv = self.init_dict["max_std2conv"]
        std2conv = lp.std2conv**2 + 1/max_std2conv**2
        conv_pairwise_0 = gaussian_kernel(pos_embed, lp.conv_ampl_2, 0, std2conv, 0, 0)
        
        local_part_2 = F.conv1d(input = orig_embed_full.clone().permute(0, 2, 1), 
                                weight = conv_pairwise_0.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        
        conv_pairwise_1 = gaussian_kernel(pos_embed, conv_ampl_1, 0, std2conv, 0, 0)
        local_part_3 = F.conv1d(input = orig_embed_full.clone().permute(0, 2, 1), 
                                weight = conv_pairwise_1.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        class_logits = (local_part_2+lp.converge_w_bias_2)*local_part_3*masks
        return class_logits 

class SingleAminoAcidInteraction(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, init_dict = {}):
        super().__init__()
        
        names = ["pos_self_sim_w", "pos_self_sim_b"]
        shapes = [(2, 1, n_features, hidden_state_dim), (1, 1, hidden_state_dim)]      
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
    
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        orig_embed_full, masks = input_dict["x_embeds"], input_dict["masks"]
        
        diag_dist_filt_0 = orig_embed_full@lp.pos_self_sim_w[0] + lp.pos_self_sim_b
        diag_dist_filt_1 = orig_embed_full@lp.pos_self_sim_w[1]
        
        class_logits = diag_dist_filt_0*diag_dist_filt_1*masks 
        return class_logits

class BetweenClassCoherence(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, init_std_class_attn, 
                 max_std_class_attn, init_dict = {}):
        super().__init__()
        
        self.init_dict = {"init_std_class_attn":init_std_class_attn,
                          "max_std_class_attn":max_std_class_attn}
        for key, value in init_dict.items():
            self.init_dict[key] = value
        names = ["conv_class_inter"]
        shapes = [(4, 1, hidden_state_dim, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        masks, pos_embed = input_dict["masks"], input_dict["pos_embed"]
        modules_collect = ["SolubilityModule", "AggregateNeighbourComparison", "SingleAminoAcidInteraction"]
        class_logits_vec = 0
        for module in modules_collect:
            class_logits_vec += input_dict[f"class_logits_{module}"]*masks

        init_std_class_attn = self.init_dict["init_std_class_attn"]
        max_std_class_attn = self.init_dict["max_std_class_attn"]
        mean_class_attn, std_class_attn, ampl_class, zero_corr_class = lp.conv_class_inter
        std_class_attn = (std_class_attn+1/init_std_class_attn)**2 + 1/max_std_class_attn**2
        conv_class_inter = gaussian_kernel(pos_embed, ampl_class, mean_class_attn, std_class_attn, 0, 0)
        
        class_conv_dist_filt = F.conv1d(input = class_logits_vec.clone().permute(0, 2, 1), 
                                        weight = conv_class_inter.permute(2, 1, 0), 
                                        stride = 1, padding = "same").permute(0, 2, 1)
        class_conv_dist_filt = class_conv_dist_filt + class_logits_vec@zero_corr_class
        class_logits = class_conv_dist_filt*masks
        return class_logits
        
class WiderBetweenClassCoherence(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, init_std_class_attn_0, 
                 max_std_class_attn_0, init_std_class_attn_1, 
                 max_std_class_attn_1, init_std_class_attn_2, 
                 max_std_class_attn_2, init_dict = {}):
        super().__init__()
        
        self.init_dict = {"init_std_class_attn_0" : init_std_class_attn_0, 
                          "max_std_class_attn_0" : max_std_class_attn_0, 
                          "init_std_class_attn_1" : init_std_class_attn_1, 
                          "max_std_class_attn_1" : max_std_class_attn_1, 
                          "init_std_class_attn_2" : init_std_class_attn_2, 
                          "max_std_class_attn_2" : max_std_class_attn_2}
        for key, value in init_dict.items():
            self.init_dict[key] = value
        names = ["class_pairwise", "class_ampl_0"]
        shapes = [(3, 4, 1, hidden_state_dim, hidden_state_dim), (1, hidden_state_dim, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
    
    def forward(self, input_dict):
        masks = input_dict["masks"]
        lp = unpack_parameter_list(self.parameter_list) #local_params
        
        class_local_dist_filt_0 = self.only_opposing_aa_neighbours(input_dict, lp)
        class_local_dist_filt_1 = self.aggregate_neighbour_comparison(input_dict, lp)
        
        class_logits = (class_local_dist_filt_0 + class_local_dist_filt_1)*masks
        return class_logits
        
    def only_opposing_aa_neighbours(self, input_dict, lp):
        masks, pos_embed = input_dict["masks"], input_dict["pos_embed"]
        class_logits_vec = input_dict["class_logits_BetweenClassCoherence"]*masks
        class_pairwise_0, class_pairwise_1, class_pairwise_2 = lp.class_pairwise
        proj_class_filter_0 = lp.class_ampl_0
        
        init_std_class_attn_0 = self.init_dict["init_std_class_attn_0"]
        max_std_class_attn_0 = self.init_dict["max_std_class_attn_0"]
        mean_class_attn, std_class_attn, ampl_class, zero_corr_class = class_pairwise_0
        std_class_attn = (std_class_attn+1/init_std_class_attn_0)**2 + 1/max_std_class_attn_0**2
        class_pairwise_0 = gaussian_kernel(pos_embed, ampl_class, mean_class_attn, std_class_attn, 0, 0)
        
        local_part_0 = F.conv1d(input = class_logits_vec.permute(0, 2, 1), 
                                weight = class_pairwise_0.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        local_part_0 = local_part_0 + class_logits_vec@zero_corr_class
        local_part_1 = class_logits_vec@proj_class_filter_0

        return local_part_0*local_part_1 
        
    def aggregate_neighbour_comparison(self, input_dict, lp):
        masks, pos_embed = input_dict["masks"], input_dict["pos_embed"]
        class_logits_vec = input_dict["class_logits_BetweenClassCoherence"]*masks
        class_pairwise_0, class_pairwise_1, class_pairwise_2 = lp.class_pairwise
        proj_class_filter_0 = lp.class_ampl_0
        
        init_std_class_attn_1 = self.init_dict["init_std_class_attn_1"]
        max_std_class_attn_1 = self.init_dict["max_std_class_attn_1"]
        mean_class_attn, std_class_attn, ampl_class, zero_corr_class = class_pairwise_1
        std_class_attn = (std_class_attn+1/init_std_class_attn_1)**2 + 1/max_std_class_attn_1**2
        class_pairwise_1 = gaussian_kernel(pos_embed, ampl_class, mean_class_attn, std_class_attn, 0, 0)
        
        local_part_2 = F.conv1d(input = class_logits_vec.permute(0, 2, 1), 
                                weight = class_pairwise_1.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        local_part_2 = local_part_2 + class_logits_vec@zero_corr_class

        init_std_class_attn_2 = self.init_dict["init_std_class_attn_2"]
        max_std_class_attn_2 = self.init_dict["max_std_class_attn_2"]
        mean_class_attn, std_class_attn, ampl_class, zero_corr_class = class_pairwise_2
        std_class_attn = (std_class_attn+1/init_std_class_attn_2)**2 + 1/max_std_class_attn_2**2
        class_pairwise_2 = gaussian_kernel(pos_embed, ampl_class, mean_class_attn, std_class_attn, 0, 0)
        
        local_part_3 = F.conv1d(input = class_logits_vec.permute(0, 2, 1), 
                                weight = class_pairwise_2.permute(2, 1, 0), 
                                stride = 1, padding = "same").permute(0, 2, 1)
        local_part_3 = local_part_3 + class_logits_vec@zero_corr_class
        
        return local_part_2*local_part_3
        
class SquareAttnCorrection(nn.Module):
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, init_dict = {}):
        super().__init__()
        
        names = ["class_x_class_lora"]
        shapes = [(1, hidden_state_dim, 4)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        orig_embed_full, class_logits = input_dict["x_embeds"], input_dict["latent_outputs"]
        masks, features2class = input_dict["masks"], input_dict["features2class"]
        
        square_w = lp.class_x_class_lora
        implicit_dim = square_w.shape[-1]//2
        square_w_old = square_w[..., :implicit_dim]@torch.transpose(square_w[..., implicit_dim:], 1, 2)

        axis_two_mask = masks.clone().unsqueeze(-1).unsqueeze(-1)
        collect_bos_eos = torch.sum(input_dict["x"][..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)

        class_p = F.softmax(class_logits, dim = -1)
        class_w_0 = (axis_one_mask[..., 0, 0]*class_p)@square_w_old
        class_w_1 = class_p*axis_two_mask[..., 0, 0]
        class_filter = torch.transpose(class_w_0, 1, 2)@orig_embed_full
        
        class_logits = (class_w_1@class_filter@features2class)*masks
        return class_logits

class SeqWideImportantRegions(nn.Module):
    def __init__(self, n_features, hidden_state_dim, model_type, init_dict = {}):
        super().__init__()
        
        names = ["pos_independent_w", "pos_independent_b"]
        shapes = [(1, hidden_state_dim, hidden_state_dim), (1, 1, hidden_state_dim)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
        
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        class_logits, masks = input_dict["latent_outputs"], input_dict["masks"]
        orig_embed_full, features2class = input_dict["x_embeds"], input_dict["features2class"]
        
        pi_logit_input = class_logits - torch.max(class_logits, dim = -1, keepdim = True).values
        pi_logit_w = F.sigmoid(pi_logit_input@lp.pos_independent_w + lp.pos_independent_b)
        pi_dist_filt = (orig_embed_full@features2class)*pi_logit_w

        axis_two_mask = masks.clone().unsqueeze(-1).unsqueeze(-1)
        collect_bos_eos = torch.sum(input_dict["x"][..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)
        
        pi_dist_filt = torch.transpose(axis_one_mask[..., 0, 0], 1, 2)@pi_dist_filt
        class_logits = (axis_two_mask[..., 0, 0]@pi_dist_filt)*masks
        return class_logits

class AttnDecompClassCoherence(nn.Module): 
    def __init__(self, n_features, hidden_state_dim, model_type, max_std_sigma, 
                 downsample_feats = None, decompose_axa = False, init_dict = {}): 
        super().__init__()         
        
        self.init_dict = {"max_std_sigma":max_std_sigma}
        self.decompose_axa = decompose_axa
        feat_i_classes = n_features + hidden_state_dim          
        if downsample_feats is None:
            downsample_feats = feat_i_classes
            
        names = ["inp_decomp_vecs", "out_decomp_vecs", 
                 "all_x_all_bias", "knn_activations"]
        shapes = [(5, downsample_feats, 1), (5, 1, hidden_state_dim), 
                  (3, 1, hidden_state_dim), (2, feat_i_classes, downsample_feats)]
                  
        if not decompose_axa:
            names = ["all_x_all_ampl"] + names
            shapes = [(5, downsample_feats, hidden_state_dim)] + shapes
        else:
            names = ["all_x_all_base", "all_x_all_weights"] + names
            shapes = [(1, downsample_feats, hidden_state_dim), (5, 1, 1)] + shapes
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
    
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        all_x_all_ampl = lp.all_x_all_ampl if not self.decompose_axa else lp.all_x_all_base*(lp.all_x_all_weights+1)
        class_attn_rec_0, class_attn_rec_1 = all_x_all_ampl[0], all_x_all_ampl[1], 
        class_attn_rec_2, class_attn_rec_3 = all_x_all_ampl[2], all_x_all_ampl[3:]   
        b_attn_0, b_attn_1, b_attn_2 = lp.all_x_all_bias

        device = input_dict["device"] 
        masks = input_dict["masks"]
        class_logits = input_dict["latent_outputs"]
        orig_embed_full, pos_embed = input_dict["x_embeds"], input_dict["pos_embed"]
        class_logits = F.softmax(class_logits, dim = -1)*masks
        orig_embed_fuller = torch.cat([orig_embed_full, class_logits], dim = -1)

        conv_feat_vecs_0 = orig_embed_fuller.clone()@lp.knn_activations[0]
        conv_feat_vecs_1 = orig_embed_fuller.clone()@lp.knn_activations[1]
    
        ampl_attn_rec = class_attn_rec_0 
        mean_attn_rec = self.recompose_weights(lp, 0)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 1))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_acts_0 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        ampl_attn_rec_copy = class_attn_rec_1
        std_attn_rec = torch.abs(self.recompose_weights(lp, 2))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec_copy, 0, std_attn_rec, 0, 0)
        class_rec_acts_1 = F.conv1d(input = conv_feat_vecs_1.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        rec_dist_filt_0 = (class_rec_acts_0+b_attn_0)*class_rec_acts_1
        
        ampl_attn_rec = class_attn_rec_2
        mean_attn_rec = self.recompose_weights(lp, 3)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 4))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_acts_2 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        class_rec_per_pos_0 = conv_feat_vecs_1@ampl_attn_rec_copy + b_attn_2
        rec_dist_filt_1 = class_rec_acts_2*class_rec_per_pos_0

        class_rec_per_pos_1 = conv_feat_vecs_0@class_attn_rec_3[0] + b_attn_1
        class_rec_per_pos_2 = conv_feat_vecs_1@class_attn_rec_3[1]
        rec_dist_filt_2 = class_rec_per_pos_1*class_rec_per_pos_2

        rec_dist_filt = rec_dist_filt_0 + rec_dist_filt_1 + rec_dist_filt_2
        class_logits = rec_dist_filt*masks 
        return class_logits

    @staticmethod
    def recompose_weights(lp, index, style = "*", 
                          inp_vec_name = "inp_decomp_vecs",
                          out_vec_name = "out_decomp_vecs"):
        inp_proj = lp.__dict__[inp_vec_name][index]
        out_proj = lp.__dict__[out_vec_name][index]
        if style == "*":
            return inp_proj * out_proj
        elif style == "+":
            return inp_proj + out_proj
        elif style == "@":
            return inp_proj @ out_proj
        else:
            return inp_proj * out_proj * 0

class WiderAttnDecompClassCoherence(nn.Module): 
    def __init__(self, n_features, hidden_state_dim, model_type, 
                 max_std_sigma_0, max_std_sigma_1, decompose_axa = False,
                 downsample_feats = None, init_dict = {}): 
        super().__init__() 
        
        self.init_dict = {"max_std_sigma_0":max_std_sigma_0,
                          "max_std_sigma_1":max_std_sigma_1}
        feat_i_classes = n_features + hidden_state_dim
        if downsample_feats is None:
            downsample_feats = feat_i_classes
        self.decompose_axa = decompose_axa
            
        names = ["inp_decomp_vecs_1", "out_decomp_vecs_1", "all_x_all_bias_1", 
                 "pos_dep_ampl_weights_1", "knn_activations_1"] 
        shapes = [(5, downsample_feats, 1), (5, 1, hidden_state_dim), (5, 1, hidden_state_dim), 
                  (1, feat_i_classes, hidden_state_dim), (2, feat_i_classes, downsample_feats)] 
        if not decompose_axa:
            names = ["all_x_all_ampl_1"] + names
            shapes = [(5, downsample_feats, hidden_state_dim)] + shapes
        else:
            names = ["all_x_all_base_1", "all_x_all_weights_1"] + names
            shapes = [(1, downsample_feats, hidden_state_dim), (5, 1, 1)] + shapes
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
    
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        all_x_all_ampl = lp.all_x_all_ampl_1 if not self.decompose_axa else lp.all_x_all_base_1*(lp.all_x_all_weights_1+1)
        class_attn_rec_0, class_attn_rec_1 = all_x_all_ampl[0], all_x_all_ampl[1], 
        class_attn_rec_2, class_attn_rec_3 = all_x_all_ampl[2], all_x_all_ampl[3:]     
        b_attn_0, b_attn_1, b_attn_2, b_attn_3, class_multiplier = lp.all_x_all_bias_1

        device = input_dict["device"] 
        masks = input_dict["masks"] 
        class_logits = input_dict["latent_outputs"]*masks
        orig_embed_full, pos_embed = input_dict["x_embeds"], input_dict["pos_embed"]
        class_logits = F.softmax(class_logits, dim = -1)*masks
        orig_embed_fuller = torch.cat([orig_embed_full, class_logits], dim = -1)

        conv_feat_vecs_0 = orig_embed_fuller.clone()@lp.knn_activations_1[0]
        conv_feat_vecs_1 = orig_embed_fuller.clone()@lp.knn_activations_1[1]
    
        ampl_attn_rec = class_attn_rec_0 
        mean_attn_rec = self.recompose_weights(lp, 0)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 1))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_acts_0 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        ampl_attn_rec_copy = class_attn_rec_1
        std_attn_rec = torch.abs(self.recompose_weights(lp, 2))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec_copy, 0, std_attn_rec, 0, 0)
        class_rec_acts_1 = F.conv1d(input = conv_feat_vecs_1.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        rec_dist_filt_0 = (class_rec_acts_0+b_attn_0)*class_rec_acts_1
        
        ampl_attn_rec = class_attn_rec_2
        mean_attn_rec = self.recompose_weights(lp, 3)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 4))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, ampl_attn_rec, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_acts_2 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        class_rec_per_pos_0 = conv_feat_vecs_1@ampl_attn_rec_copy + b_attn_2
        rec_dist_filt_1 = class_rec_acts_2*class_rec_per_pos_0

        class_rec_per_pos_1 = conv_feat_vecs_0@class_attn_rec_3[0] + b_attn_1
        class_rec_per_pos_2 = conv_feat_vecs_1@class_attn_rec_3[1]
        rec_dist_filt_2 = class_rec_per_pos_1*class_rec_per_pos_2

        axis_two_mask = torch.reshape(masks.clone(), masks.shape+(1, 1))
        collect_bos_eos = torch.sum(input_dict["x"][..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)
        new_masks = axis_one_mask[..., 0, 0]
        attn_mask = torch.unsqueeze(new_masks@torch.transpose(new_masks, 1, 2), 1)
        positions = torch.arange(orig_embed_full.shape[1]).to(device) 
        rel_positions = positions[None, None, None, :] - positions[None, None, :, None] # [1, 1, L, L] 

        class_multiplier = class_multiplier.unsqueeze(0).unsqueeze(0)
        sigma = (class_multiplier.permute(0, 3, 2, 1))**2 + 1/self.init_dict["max_std_sigma_1"]**2 #[B, C, L, 1] 
        G_v = torch.exp(-sigma*rel_positions**2) #[B, C, L, L]
        V = G_v*attn_mask #[B, C, L, L]
        V_correction = torch.sum(V, dim = -1).permute(0, 2, 1)

        rec_dist_filt = rec_dist_filt_0 + rec_dist_filt_1 + rec_dist_filt_2
        per_pos_ampl = orig_embed_fuller@lp.pos_dep_ampl_weights_1 + b_attn_3
        class_logits = V_correction*per_pos_ampl*rec_dist_filt*masks
        return class_logits

    @staticmethod
    def recompose_weights(lp, index, style = "*", 
                          inp_vec_name = "inp_decomp_vecs_1",
                          out_vec_name = "out_decomp_vecs_1"):
        inp_proj = lp.__dict__[inp_vec_name][index]
        out_proj = lp.__dict__[out_vec_name][index]
        if style == "*":
            return inp_proj * out_proj
        elif style == "+":
            return inp_proj + out_proj
        elif style == "@":
            return inp_proj @ out_proj
        else:
            return inp_proj * out_proj * 0
            
class EvenWiderAttnDecompClassCoherence(nn.Module): 
    def __init__(self, n_features, hidden_state_dim, model_type, 
                 max_std_sigma_0, max_std_sigma_1, decompose_axa = False, 
                 downsample_feats = None, init_dict = {}): 
        super().__init__() 
        
        self.init_dict = {"max_std_sigma_0":max_std_sigma_0,
                          "max_std_sigma_1":max_std_sigma_1}
        feat_i_classes = n_features + hidden_state_dim
        if downsample_feats is None:
            downsample_feats = feat_i_classes
        self.decompose_axa = decompose_axa
                     
        names = ["inp_decomp_vecs_2", "out_decomp_vecs_2", "all_x_all_bias_2", 
                 "pos_dep_ampl_weights_2", "knn_activations_2"] 
        shapes = [(5, downsample_feats, 1), (5, 1, hidden_state_dim), (5, 1, hidden_state_dim), 
                  (1, feat_i_classes, hidden_state_dim), (2, feat_i_classes, downsample_feats)] 
        if not decompose_axa:
            names = ["all_x_all_ampl_2"] + names
            shapes = [(5, downsample_feats, hidden_state_dim)] + shapes
        else:
            names = ["all_x_all_base_2", "all_x_all_weights_2"] + names
            shapes = [(1, downsample_feats, hidden_state_dim), (5, 1, 1)] + shapes
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
    
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        all_x_all_ampl = lp.all_x_all_ampl_2 if not self.decompose_axa else lp.all_x_all_base_2*(lp.all_x_all_weights_2+1)
        class_attn_rec_0, class_attn_rec_1 = all_x_all_ampl[0], all_x_all_ampl[1], 
        class_attn_rec_2, class_attn_rec_3 = all_x_all_ampl[2], all_x_all_ampl[3:]
        b_attn_0, b_attn_1, b_attn_2, b_attn_3, class_multiplier = lp.all_x_all_bias_2

        device = input_dict["device"] 
        masks = input_dict["masks"] 
        class_logits = input_dict["latent_outputs"]
        orig_embed_full, pos_embed = input_dict["x_embeds"], input_dict["pos_embed"]
        class_logits = F.softmax(class_logits, dim = -1)*masks
        orig_embed_fuller = torch.cat([orig_embed_full, class_logits], dim = -1)

        conv_feat_vecs_0 = orig_embed_fuller.clone()@lp.knn_activations_2[0]
        conv_feat_vecs_1 = orig_embed_fuller.clone()@lp.knn_activations_2[1]
    
        ampl_attn_rec = class_attn_rec_0 
        mean_attn_rec = self.recompose_weights(lp, 0)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 1))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, 1, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_kernel = (class_rec_kernel/torch.sum(class_rec_kernel, dim = 0, keepdim = True))*ampl_attn_rec
        class_rec_acts_0 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        ampl_attn_rec_copy = class_attn_rec_1
        std_attn_rec = torch.abs(self.recompose_weights(lp, 2))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, 1, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_kernel = (class_rec_kernel/torch.sum(class_rec_kernel, dim = 0, keepdim = True))*ampl_attn_rec_copy
        class_rec_acts_1 = F.conv1d(input = conv_feat_vecs_1.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        rec_dist_filt_0 = (class_rec_acts_0+b_attn_0)*class_rec_acts_1
        
        ampl_attn_rec = class_attn_rec_2
        mean_attn_rec = self.recompose_weights(lp, 3)
        std_attn_rec = torch.abs(self.recompose_weights(lp, 4))**0.5
        std_attn_rec = std_attn_rec**2 + 1/self.init_dict["max_std_sigma_0"]**2
        class_rec_kernel = gaussian_kernel(pos_embed, 1, mean_attn_rec, std_attn_rec, 0, 0)
        class_rec_kernel = (class_rec_kernel/torch.sum(class_rec_kernel, dim = 0, keepdim = True))*ampl_attn_rec
        class_rec_acts_2 = F.conv1d(input = conv_feat_vecs_0.clone().permute(0, 2, 1), 
                                    weight = class_rec_kernel.permute(2, 1, 0), 
                                    stride = 1, padding = "same").permute(0, 2, 1)
        class_rec_per_pos_0 = conv_feat_vecs_1@ampl_attn_rec_copy + b_attn_2
        rec_dist_filt_1 = class_rec_acts_2*class_rec_per_pos_0

        class_rec_per_pos_1 = conv_feat_vecs_0@class_attn_rec_3[0] + b_attn_1
        class_rec_per_pos_2 = conv_feat_vecs_1@class_attn_rec_3[1]
        rec_dist_filt_2 = class_rec_per_pos_1*class_rec_per_pos_2

        axis_two_mask = torch.reshape(masks.clone(), masks.shape+(1, 1))
        collect_bos_eos = torch.sum(input_dict["x"][..., 21:], dim = -1, keepdim = True)
        axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)
        new_masks = axis_one_mask[..., 0, 0]
        attn_mask = torch.unsqueeze(new_masks@torch.transpose(new_masks, 1, 2), 1)
        positions = torch.arange(masks.shape[1]).to(device) 
        rel_positions = positions[None, None, None, :] - positions[None, None, :, None] # [1, 1, L, L] 

        class_multiplier = class_multiplier.unsqueeze(0).unsqueeze(0)
        sigma = (class_multiplier.permute(0, 3, 2, 1))**2 + 1/self.init_dict["max_std_sigma_1"]**2 #[B, C, L, 1] 
        G_v = torch.exp(-sigma*rel_positions**2) #[B, C, L, L]
        V = G_v*attn_mask #[B, C, L, L]
        V_correction = torch.sum(V, dim = -1).permute(0, 2, 1)

        rec_dist_filt = rec_dist_filt_0 + rec_dist_filt_1 + rec_dist_filt_2
        per_pos_ampl = orig_embed_fuller@lp.pos_dep_ampl_weights_2 + b_attn_3
        class_logits = V_correction*per_pos_ampl*rec_dist_filt*masks
        return class_logits

    @staticmethod
    def recompose_weights(lp, index, style = "*", 
                          inp_vec_name = "inp_decomp_vecs_2",
                          out_vec_name = "out_decomp_vecs_2"):
        inp_proj = lp.__dict__[inp_vec_name][index]
        out_proj = lp.__dict__[out_vec_name][index]
        if style == "*":
            return inp_proj * out_proj
        elif style == "+":
            return inp_proj + out_proj
        elif style == "@":
            return inp_proj @ out_proj
        else:
            return inp_proj * out_proj * 0
            
class ESMMimicryModule(nn.Module):
    def __init__(self, n_features, hidden_state_dim, model_type, downsampling_feats, 
                 value_feats, n_heads, n_layers, n_landmarks, max_laplace_std_attn,
                 dsample_heads = "none", init_dict = {}):
        super().__init__()
        
        self.init_dict = {"max_laplace_std_attn": max_laplace_std_attn, 
                          "n_landmarks": n_landmarks}
        feat_i_classes = n_features + hidden_state_dim
        assert dsample_heads in ["none", "query", "value"], ("Incorrect downsampling mode for heads selected")
        q_heads = n_heads if dsample_heads in ["none", "value"] else 1
        v_heads = n_heads if dsample_heads in ["none", "query"] else 1
                     
        names = ["collect_classes", "query_linear", "key_linear", 
                 "value_linear", "project_original", "aggregation_params"]
        shapes = [(1, value_feats*n_heads, hidden_state_dim),
                  (n_layers, 1, q_heads, feat_i_classes+1, downsampling_feats),
                  (n_layers, 1, 1, feat_i_classes+1, downsampling_feats),
                  (n_layers, 1, v_heads, feat_i_classes+1, value_feats),
                  (n_layers, 1, 1, feat_i_classes+1, value_feats),
                  (n_layers, 1, n_heads, 1, 1)]
        if n_layers > 1:
            names = names + ["collect_feats"]
            shapes = shapes + [(n_layers-1, value_feats*n_heads, feat_i_classes)]
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict)
    
    def forward(self, input_dict):
        lp = unpack_parameter_list(self.parameter_list) #local_params
        
        device = input_dict["device"]
        class_logits, masks = input_dict["latent_outputs"], input_dict["masks"]
        orig_embed_full = input_dict["x_embeds"]
        class_logits = F.softmax(class_logits, dim = -1)*masks
        orig_embed_fuller = torch.cat([orig_embed_full, class_logits], dim = -1).unsqueeze(1)

        n_layers = lp.query_linear.shape[0]
        value_accum = orig_embed_fuller.clone()
        for layer in range(1, n_layers+1):
            value_accum = self.compute_accum(value_accum, input_dict, lp, layer-1)
            project_orig_w, project_orig_b = lp.project_original[layer-1, ..., :-1, :], lp.project_original[layer-1, ..., -2:-1, :]
            value_accum = value_accum + (orig_embed_fuller@project_orig_w + project_orig_b)/n_layers
            value_accum = torch.flatten(value_accum.permute(0, 2, 1, 3), start_dim = -2, end_dim = -1)
            
            if layer < n_layers:
                collect_weights = lp.collect_feats[layer-1]
            else:
                collect_weights = lp.collect_classes
            value_accum = (value_accum@collect_weights).unsqueeze(1)
        
        esm_dist_filt = torch.squeeze(value_accum, dim = 1)
        class_logits = esm_dist_filt*masks
        return class_logits

    def compute_accum(self, values, input_dict, lp, index):
        device = input_dict["device"]
        masks, pos_embed = input_dict["masks"], input_dict["pos_embed"]
        
        seq_len, n_landmarks = values.shape[-2], self.init_dict["n_landmarks"]
        if n_landmarks is not None:
            right_pad = int(np.ceil(seq_len/n_landmarks))*n_landmarks - seq_len
            values = F.pad(values, (0, 0, 0, right_pad))
            feat_dim, seq_len = lp.query_linear.shape[-1], right_pad + seq_len

        E_q = values@lp.query_linear[index, ..., :-1, :] + lp.query_linear[index, ..., -2:-1, :]
        E_k = values@lp.key_linear[index, ..., :-1, :] + lp.key_linear[index, ..., -2:-1, :]
        E_v = values@lp.value_linear[index, ..., :-1, :] + lp.value_linear[index, ..., -2:-1, :]
        
        positions = torch.arange(masks.shape[1]).to(device)
        rel_positions = positions[None, None, None, :] - positions[None, None, :, None]  # [1, 1, L, L]
        sigmas = torch.abs(lp.aggregation_params[index]) + self.init_dict["max_laplace_std_attn"]
        pos_mask = -sigmas*torch.abs(rel_positions)
        E_q = E_q*(sigmas*0+1)
        
        if n_landmarks is None:
            
            axis_two_mask = torch.reshape(masks.clone(), masks.shape+(1, 1))
            collect_bos_eos = torch.sum(input_dict["x"][..., 21:], dim = -1, keepdim = True)
            axis_one_mask = axis_two_mask+torch.reshape(collect_bos_eos, axis_two_mask.shape)
            new_masks = axis_one_mask[..., 0, 0]
            attn_mask = torch.unsqueeze(new_masks@torch.transpose(new_masks, 1, 2), 1)
            attn_mask = torch.where(attn_mask == 0, -1e06, 0) + pos_mask
            
            accumulate_values = F.scaled_dot_product_attention(E_q, E_k, E_v, attn_mask=attn_mask)
        
        else:        
            
            E_q = E_q/np.sqrt(np.sqrt(feat_dim))
            E_k = E_k/np.sqrt(np.sqrt(feat_dim))
            
            q_landmarks = torch.unflatten(E_q, -2, (n_landmarks, seq_len//n_landmarks)).mean(dim=-2)    
            k_landmarks = torch.unflatten(E_k, -2, (n_landmarks, seq_len//n_landmarks)).mean(dim=-2)    

            pos_select = torch.unflatten(pos_mask, -2, (n_landmarks, seq_len//n_landmarks)).mean(dim=-2)    
            kernel_1 = F.softmax(E_q@k_landmarks.transpose(-1, -2)+pos_select.transpose(-1, -2), dim=-1) 
            pos_select_2 = torch.unflatten(pos_select, -1, (n_landmarks, seq_len//n_landmarks)).mean(dim=-1)    
            kernel_2 = F.softmax(q_landmarks@k_landmarks.transpose(-1, -2) + pos_select_2, dim=-1) 
            attention_scores = q_landmarks@E_k.transpose(-1, -2)
            attention_mask = torch.transpose(torch.where(masks.unsqueeze(1) == 0, -1e06, 0), -1, -2)
            attention_mask = F.pad(attention_mask, (0, right_pad), "constant", -1e06)
            attention_scores = attention_scores + attention_mask + pos_select
            
            kernel_3 = F.softmax(attention_scores, dim=-1) 
            attention_probs = kernel_1@self.iterative_inv(kernel_2)
            updated_E_v = kernel_3@E_v 
            accumulate_values = attention_probs@updated_E_v
            
        return accumulate_values

    def iterative_inv(self, mat, n_iter=6, init_option="original"):
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat
        # The entries of key are positive and ||key||_{\infty} = 1 due to softmax
        if init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)
        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(
                0.25 * value,
                13 * identity
                - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)),
            )
        return value

all_module_class_calls = [SharedVariableModule, InputOutputEmbeddingModule, PhysioChemical2Class, InterClassBiasCorrection, 
                          SolubilityModule, AggregateNeighbourComparison, SingleAminoAcidInteraction, 
                          BetweenClassCoherence, WiderBetweenClassCoherence, SquareAttnCorrection, 
                          SeqWideImportantRegions, AttnDecompClassCoherence, WiderAttnDecompClassCoherence,
                          EvenWiderAttnDecompClassCoherence, ESMMimicryModule]
module_inits = dict([(call.__name__, call) for call in all_module_class_calls])

### DEFINITION OF THE MODEL OBJECT
class GeneralizedStructureModel(nn.Module):
    #MODEL CONSTRUCTORS
    def __init__(self, n_features, n_classes, hidden_state_dim, module_init_dict, additional_features = 0, 
                 window_size = None, active_modules = None, bilinear = False, model_type = "linear_laplace", model_name = "Model", 
                 init_dict = {}, load_state_dict = "", set_model_grad = {}, prior_sigma = 1, 
                 load_model_kwargs = {}, set_grad_kwargs = {}):
        super().__init__()
        assert model_type in ["linear_laplace", "mean_field", "deterministic"], ("Incorrect model type!")
        
        self.n_features = n_features
        self.additional_features = additional_features
        self.window_size = window_size
        self.hidden_state_dim = hidden_state_dim
        self.n_classes = n_classes
        
        self.model_type = model_type
        self.model_name = model_name
        self.active_modules = active_modules
        self.bilinear = bilinear
        self.construct_model(module_init_dict, init_dict)
        
        if len(load_state_dict) > 0:
            self.load_model(load_state_dict, **load_model_kwargs)
        if len(set_model_grad) > 0:
            self.set_model_grad(set_model_grad, **set_grad_kwargs)
        if model_type in ["linear_laplace", "mean_field"]:
            self.set_prior_precision(prior_sigma)
        self.calibrated = False

    def construct_model(self, module_init_dict, init_dict={}):
        model_type = self.model_type
        n_features = self.n_features
        additional_features = self.additional_features
        hidden_state_dim = self.hidden_state_dim
        base_n_classes = self.n_classes
        
        full_features = n_features + additional_features
        bilinear, need_projection = self.bilinear, self.hidden_state_dim != self.n_classes
        module_count = len(module_init_dict) if (need_projection or bilinear) else 0
        
        execution_order = nn.ModuleDict()  
        execution_order["SharedVariableModule"] = SharedVariableModule(full_features, hidden_state_dim, model_type, 
                                                                       module_count, base_n_classes, init_dict)
        for module_name, (module_fn, module_params) in module_init_dict.items():
            execution_order[module_name] = module_fn(n_features=full_features, hidden_state_dim=hidden_state_dim, 
                                                     model_type=model_type, init_dict=init_dict, **module_params)
        self.execution_order = execution_order
        if self.active_modules is None or len(self.active_modules) < len(execution_order):
            print("Warning! No active_modules provided. Check the model arguments or ignore, if this was planned")
            active_modules = [1, 1] + [0]*(len(execution_order)-2)
            self.active_modules = dict(zip(list(execution_order.keys()), active_modules))

    #MODEL CALLERS
    def forward(self, x_feats, masks, return_logits = True, return_prev_compute = False):
        device = x_feats.device
        active_modules = self.active_modules
        execution_order = self.execution_order
        batch_size, max_length = x_feats.shape[0], x_feats.shape[1]
        base_n_classes, bilinear = self.n_classes, self.bilinear
        need_projection = self.hidden_state_dim != self.n_classes
        
        window_size = self.window_size
        window_size = window_size if window_size != None else x_feats.shape[1]//2*2+1
        edge_pos = window_size//2
        pos_embed = torch.linspace(-edge_pos, edge_pos, window_size, dtype = torch.float32).to(device)
        pos_embed = torch.unsqueeze(torch.unsqueeze(pos_embed, -1), -1)
        
        if bilinear:
            output_vec = torch.zeros((batch_size, max_length, max_length, base_n_classes)).to(device)
        else:
            output_vec = torch.zeros((batch_size, max_length, base_n_classes)).to(device)
        latent_vec = torch.zeros((batch_size, max_length, self.hidden_state_dim)).to(device)

        data_dict = OrderedDict(device = device, x = x_feats[..., :23], x_other = x_feats[..., 23:], masks = masks, 
                                pos_embed = pos_embed, class_outputs = output_vec, latent_outputs = latent_vec)

        execution_order.SharedVariableModule(data_dict)
        for module_order, module_name in enumerate(list(execution_order.keys())[1:]):
            use_module, module_fn = active_modules[module_name], execution_order[module_name]
            if use_module > 1e-08:
                with record_function(module_name):
                    module_output = module_fn(data_dict)*use_module
                    data_dict[f"class_logits_{module_name}"] = module_output
                    
                    if need_projection or bilinear:
                        extract_projection = data_dict[f"output_projection_{module_order}"]
                        latent_projection = data_dict[f"latent_projection_{module_order}"]
                        class_module_output = module_output@extract_projection
                        latent_module_output = module_output@latent_projection
                    else:
                        class_module_output = module_output.clone()
                        latent_module_output = module_output.clone()
                        
                    if bilinear:
                        class_module_output = module_output@data_dict[f"class_logits_{module_name}"].permute(0, 2, 1)
                        class_module_output = (class_module_output + class_module_output.permute(0, 2, 1))/2

                    data_dict["latent_outputs"] = data_dict["latent_outputs"] + latent_module_output
                    data_dict["class_outputs"] = data_dict["class_outputs"] + class_module_output
        del data_dict["device"]

        if not return_logits and base_n_classes > 1:
            data_dict["class_outputs"] = F.softmax(data_dict["class_outputs"], dim = -1)
        if not return_logits and base_n_classes == 1:
            data_dict["class_outputs"] = F.sigmoid(data_dict["class_outputs"])
        if return_prev_compute == False:
            return data_dict["class_outputs"]
        return data_dict

    def call_model(self, x_feats, masks, return_logits = True, return_prev_compute = False):
        return self.forward(x_feats, masks, return_logits, return_prev_compute)

    def inference_model(self, input_data, return_logits = False, 
                        enable_grad = False, batch_fn = None, use_tqdm = True):
        base_n_classes = self.n_classes
        if enable_grad:
            grad_context = torch.set_grad_enabled(True)
        else:
            grad_context = torch.no_grad()
            
        with grad_context:
            device = next(self.parameters()).device
            device_type = device.type

            n_dims = base_n_classes if return_logits else 1
            predictions, y_trues, masks_count = [], [], []

            prog_datalist = tqdm(input_data) if use_tqdm else input_data
            for input_batch in prog_datalist:
                supervised = len(input_batch) > 2
                if supervised:
                    x_feats, y_true, masks = input_batch
                else:
                    x_feats, masks = input_batch
                x_feats, masks = x_feats.to(device), masks.to(device)
                y_pred = self(x_feats, masks, return_logits = return_logits)
                
                if batch_fn is not None:
                    batch_fn(x_feats, masks, y_pred)

                select_parts_pred = y_pred.reshape(-1, base_n_classes)
                if not return_logits and base_n_classes > 1:
                    select_parts_pred = torch.argmax(select_parts_pred, dim = -1, keepdim = True)
                elif not return_logits and base_n_classes == 1:
                    select_parts_pred = (select_parts_pred > 0.5).to(int)
                    
                predictions.append(select_parts_pred)
                real_mask = masks if not self.bilinear else masks*masks.permute(0, 2, 1)
                masks_count.append(real_mask.reshape(-1, ))
                if supervised:
                    y_trues.append(y_true.to(device).reshape(-1, 1))
                
                if device_type == "xla":
                    xm.mark_step()

            predictions = torch.cat(predictions, 0)
            masks_count = torch.cat(masks_count, 0)
            predictions = predictions[masks_count == 1].cpu().detach().numpy()
            
            if supervised:
                y_trues = torch.cat(y_trues, 0)
                y_trues = y_trues[masks_count == 1].cpu().detach().numpy()
                return predictions, y_trues
            return predictions
      
    def score_model(self, *dataloaders):
        model_name = self.model_name
        determine_if_run = lambda x: self.active_modules[x] > 1e-08
        all_modules = list(self.execution_order.keys())
        all_active_modules = list(filter(determine_if_run, all_modules))

        print("-"*70)
        fn_outputs, accs_text = [], [f"Active Modules: {', '.join(all_active_modules)}"]
        for i, dataloader in enumerate(dataloaders):
            if hasattr(dataloader, "name"):
                dloader_name = dataloader.name
            else:
                dloader_name = dataloader._loader.name
            pred_dloader, true_dloader = self.inference_model(dataloader)
            base_accuracy = np.mean(np.array(pred_dloader) == np.array(true_dloader))*100
            format_accuracy = round(float(base_accuracy), 3)
            acc_string = f"{model_name} {dloader_name} Accuracy: {format_accuracy} %"
            fn_outputs.append(pred_dloader), fn_outputs.append(true_dloader), accs_text.append(acc_string)

        accs_text.append("-"*70)
        print("\n".join(accs_text))
        return fn_outputs
    
    def linearize_model(self, dataloader):
        assert hasattr(self, "cov_params") ("No covariance matrix found. You need to compute it before linearization!")

    #MODEL STATE MANIPULATION
    def save_model(self, file_name):
        new_state_dict = {}
        use_module = self.active_modules

        calibration_filter = lambda name: ".mu" in name if not self.calibrated else True
        use_module_filter = lambda module_name: use_module[module_name] > 1e-08
        
        for param_name, param_value in self.state_dict().items():
            module_name = param_name.split(".")[1]
            if calibration_filter(param_name) and use_module_filter(module_name):
                new_state_dict[param_name] = param_value

        export_dict = dict(state_dict = new_state_dict)
        torch.save(export_dict, file_name)

    def load_model(self, file_name, device = None, strict = False, file_delete = False):
        rename_dict = {"PhysioChemical2Class.parameter_list.final_bias":"InputOutputEmbeddingModule.parameter_list.class_frequency_embed", 
                       ".EmbeddingModule":".InputOutputEmbeddingModule", "square_w":"class_x_class_lora", "feat_square":"pos_self_sim_w", "window_sum":"pos_self_sim_b"}
        if device is None:
            device = next(self.parameters()).device
            
        loaded_data = torch.load(file_name, map_location = device)
        state_dict = loaded_data["state_dict"]
        refactored_state_dict = {}
        for key in list(state_dict.keys()):
            refactored_key = str(key)
            for replace_key, replace_value in rename_dict.items():
                refactored_key = refactored_key.replace(replace_key, replace_value)
            refactored_state_dict[refactored_key] = state_dict.pop(key)
            
        new_state_dict = {}
        for key, vector in self.state_dict().items():
            new_vector = refactored_state_dict.get(key, vector).to(device)
            if not torch.numel(vector) == torch.numel(new_vector):
                new_vector = vector
            else:
                new_vector = new_vector.reshape(vector.shape)
            new_state_dict[key] = new_vector
        self.load_state_dict(new_state_dict, strict = strict, assign = True)

        if file_delete:
            os.remove(file_name)

    def reset_model(self):
        self.construct_model({})
        
    #SETTERS AND GETTERS 
    def set_model_grad(self, params_to_freeze_dict, use_old_style = True, verbose = True):
        conditional_print = lambda x: print(x) if verbose else x
        conditional_print("-"*70)
        
        for key, vector in self.named_parameters():
            if use_old_style:
                name, var_type = key.split(".")[-2:] #freeze learned mu and log_sigma!
            else:
                name = key
                
            if name in set(params_to_freeze_dict.keys()):
                enable_grad = params_to_freeze_dict[name]
                is_from_prior = "prior" in var_type
                vector.requires_grad = enable_grad and not is_from_prior
                key_name_print = key.replace("execution_order.", "").replace("parameter_list.", "")
                
                conditional_print(f"Successfully assigned {key_name_print}.requires_grad = {vector.requires_grad}")
                    
        conditional_print("-"*70)

    def get_model_grad(self, use_old_style = False):
        grad_dict = {}
        all_named_params = self.named_parameters()    
        
        for key, vector in all_named_params.items():
            if use_old_style:
                name, var_type = key.split(".")[-2:] 
            else:
                name = key
            grad_dict[name] = vector.requires_grad
            
        return grad_dict
                
    def get_model_params(self, param_names_tuple, use_old_style = True, numpify = True):     
        vectors = []
        for param_name, param_type in param_names_tuple:
            if use_old_style:
                select_param_name = f"{param_name}_{param_type}"
                for key, vector in self.state_dict().items():
                    key_name = "_".join(key.split(".")[-2:])
                    if key_name == select_param_name:
                        break
            else:
                vector = self.named_parameters()[param_name]  
            if numpify:
                vector = vector.cpu().detach().numpy()
            vectors.append(vector)
            
        if len(param_names_tuple) == 1:
            vectors = vectors[0]
        return vectors
    
    def get_model_size(self, detailed = True, use_old_style = True, types = ["mu"]):
        execution_order = self.execution_order
        active_modules = self.active_modules 
        param_size = OrderedDict()
        
        for module_name in list(execution_order.keys()):
            use_module, module_fn = active_modules[module_name], execution_order[module_name]
            if use_module < 1e-08:
                continue
                
            for key, vector in module_fn.named_parameters():
                if use_old_style:
                    name = "_".join(key.split(".")[-2:])
                else:
                    name = f"{module_name}.{key}"
                if vector.requires_grad and key.split(".")[-1] in types:
                    param_size[name] = torch.numel(vector)
                    
        if not detailed:
            return sum(list(param_size.values()))
        else:
            return param_size

    def pretty_parameter_count(self):
        model_size_table = self.get_model_size(detailed = True, use_old_style = False)
        model_size_table = pd.DataFrame([item[0].split(".") + [item[1]]for item in list(model_size_table.items())])[[0, 2, 4]]
        model_size_table.columns = ["module", "name", "size"]
        
        printable_table = model_size_table.to_markdown()
        borders = printable_table.split("\n")[1].replace(":", "-")
        count_param_str = f"| Count the number of parameters: {sum(model_size_table['size'])}" 
        count_param_str = count_param_str + (len(borders)-len(count_param_str) - 1)*" "+"|"
        
        print()
        print(f"Parameter Count of {self.model_name} per Module:")
        print()
        print(borders)
        print(printable_table)
        print(borders)
        print(count_param_str)
        print(borders)
        
    def get_prior_loss(self):
        if not self.model_type in ["linear_laplace", "mean_field"]:
            return 0.0
            
        prior_loss_val = 0.0
        active_modules, execution_order = self.active_modules, self.execution_order
        prior_loss_fn = log_prior_fn if self.model_type == "linear_laplace" else kl_div_fn
        
        for module_name in list(execution_order.keys()):
            use_module, module_fn = active_modules[module_name], execution_order[module_name]
            if use_module < 1e-08:
                continue
                
            for param_obj in module_fn.parameter_list:
                mu_pred, sigma_pred = param_obj.mu, 0.0 if self.model_type == "linear_laplace" else F.softplus(param_obj.log_sigma)
                mu_prior, sigma_prior = param_obj.prior_mu, F.softplus(param_obj.prior_sigma)
                
                prior_loss_new = torch.sum(prior_loss_fn(mu_pred, sigma_pred, mu_prior, sigma_prior))
                prior_loss_val = prior_loss_val + prior_loss_new
                    
        return prior_loss_val

    #LINEARIZED LAPLACE SUPPORTING FUNCTIONS
    #method by Alex Immer et al: https://arxiv.org/pdf/2008.08400
    #detailed formulas taken from: arxiv.org/pdf/2010.14689
    def set_prior_precision(self, prior_sigma):
        prior_sigma_softplus = np.log(np.exp(prior_sigma)-1) #inverse softplus
        for module_fn in self.execution_order.values():
            for param_obj in module_fn.parameter_list:
                param_obj.prior_sigma.fill_(prior_sigma_softplus)
                
    def get_prior_precision(self):
        if not self.model_type in ["linear_laplace", "mean_field"]:
            raise RuntimeError ("The model is not bayesian: impossible to get priors!")

        all_sigma_priors = []
        for module_fn in self.execution_order.values():
            for param_obj in module_fn.parameter_list:
                prior_prec = 1/F.softplus(param_obj.prior_sigma.flatten())**2
                all_sigma_priors.append(prior_prec)
        return torch.diag(torch.cat(all_sigma_priors))

    def get_jacobian(self, x_input, masks):
        return None

    @staticmethod
    def get_hessian(y_pred):
        probs = torch.clamp(F.softmax(y_pred, dim = -1), 1e-07, 1 - 1e-07)
        hessian = torch.diag_embed(probs) - torch.einsum('ij,ik->ijk', probs, probs)
        return hessian
        
    def get_batch_ggn(self, x_input, masks):
        jacobian, y_pred = self.get_jacobian(x_input, masks) #B, D, O
        hessian = get_hessian(y_pred) #B, O, O
        
        return torch.einsum("mpk,mkl,mql->pq", [jacobian, hessian, jacobian])
    
    def get_generalized_gauss_newton(self, train_dataloader):
        ggn_output = self.get_prior_precision()
    
        for (x_input, _, masks) in tqdm(train_dataloader):
            x_input, masks = x_input.to(device), masks.to(device)
            ggn_output = ggn_output + self.get_batch_ggn(x_input, masks)
            
        return ggn_output
    
    def get_covariance(self, ggn_output = None, train_dataset = None):
        input_is_provided = not(ggn_output is None and train_dataset is None) 
        assert input_is_provided, ("No way to estimate Generalized Gauss-Newton matrix provided.")
        
        if ggn_output is None:
            train_dataloader = create_dataloader(train_dataset, 1, dataloader_name = "Train")
            ggn_output = self.get_generalized_gauss_newton(model, train_dataloader)
                
        chol = torch.linalg.cholesky(ggn_output)
        self.cov_params = torch.cholesky_inverse(chol, upper = False)
            
        return cov_params
        
    #DUNDER FUNCTIONS
    def __call__(self, x_feats, masks, return_logits = True, return_prev_compute = False):
        return self.call_model(x_feats, masks, return_logits, return_prev_compute)

    def __len__(self):
        return self.get_model_size(detailed = False)                    
