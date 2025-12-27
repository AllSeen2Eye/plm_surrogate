import torch
import shap

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import AutoTokenizer
checkpoint_str = "facebook/esm2_t30_150M_UR50D" #"facebook/esm2_t30_150M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(checkpoint_str)

from commons import *

def tokenizer_mine(seqs, device):
    seq_size = max([len(seq.split(" ")) for seq in seqs])
    max_len = seq_size+2
    
    x_feats = torch.zeros((len(seqs), max_len, 23)).to(device)
    x_feats[:, 0, 21] = 1
    masks = torch.zeros((len(seqs), max_len, 1)).to(device)
    for i, seq in enumerate(seqs):
        seq = [aa if aa in list(aa_alphabet) else "X" for aa in seq.split(" ")]
        x_feats[i, 1:seq_size+1, :21] = torch.FloatTensor(np.reshape(np.array(seq), (-1, 1)) == x_tokens)
        x_feats[i, seq_size+1, 22] = 1
        masks[i, 1:seq_size+1] = 1
    return x_feats, masks

def get_embeddings(model, tokenizer_mine, device, inp):
    x_feats, masks = tokenizer_mine(inp, device)
    results = model(x_feats, masks)
    return results

def factorize(number):
    candidate_factors, actual_factors = range(1, number), []
    for factor in candidate_factors:
        if number % factor == 0:
            actual_factors.append(factor)
          
    closest_1 = int(len(actual_factors) // 2)
    closest_0 = closest_1 - int(len(actual_factors) % 2 == 0)
    return actual_factors[closest_1], actual_factors[closest_0]

def visualize_shap(seq, model, tokenizer, device, full_class_str, figure_dims = ()):
    n_classes = len(full_class_str) 
    n_classes = n_classes if n_classes > 2 else 1
    if len(figure_dims) < 2:
        figure_dims = factorize(n_classes)
    n_rows, n_cols = figure_dims
  
    x_ =  [" ".join(list(seq))]
    shap_values = np.zeros((len(x_[0].split(" ")), len(x_[0].split(" ")), n_classes))
    for class_id in range(0, n_classes):
        explainer = shap.Explainer(lambda inp: get_embeddings(model, tokenizer, device, inp)[..., class_id], esm_tokenizer)
        shap_values[..., class_id] = explainer(x_, fixed_context=1).values[0, 1:-1, 1:-1].T
    
    min_val, max_val = np.percentile(shap_values, [5, 95])
    max_val = np.max(np.abs([min_val, max_val]))
    min_val = -1*max_val
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (n_cols*8, n_rows*8))
    for class_id in range(n_classes):
      residual_shap_matrix = shap_values[..., class_id]
      ax[class_id//4, class_id%4].matshow(residual_shap_matrix, vmin = min_val, vmax = max_val, cmap = "coolwarm")
      ax[class_id//4, class_id%4].set_title(f"{full_class_str[class_id]} {min_val:.3f} to {max_val:.3f}")

def visualize_physicochemical(structure_model, feat_id, class_id, window_size = 121):
    all_pos = np.reshape(list(range(0, window_size)), (-1, 1))
    pos_embed = np.array([all_pos - all_pos.T])
    pos_embed = np.expand_dims(pos_embed, axis = -1)
    pos_embed = np.expand_dims(pos_embed, axis = -1)
    
    params_to_extract = ["mean_dist_attn", "std_dist_attn", 
                         "period_cos", "phase_cos", "w_bias", 
                         "features2class", "rotate_embeds"]
    params_to_extract = list(zip(params_to_extract, len(params_to_extract)*["mu"]))
    
    mean_dist_attn, std_dist_attn, period_cos, phase_cos, w_bias_true, features2class, rotate_embeds = structure_model.get_model_params(params_to_extract)
    std_dist_attn = std_dist_attn**2
    cos_part = np.cos(period_cos*pos_embed + phase_cos)
    exp_part = np.exp(-(std_dist_attn*(pos_embed-mean_dist_attn)**2))
    full_w = rotate_embeds@((cos_part*exp_part + w_bias_true)*features2class)
    
    fig, ax = plt.subplots(figsize = (10, 10))
    mat2display = full_w[0, :, :, feat_id, class_id]
    cbar = ax.matshow(mat2display, cmap = "coolwarm", vmin = -np.max(np.abs(mat2display)), vmax = np.max(np.abs(mat2display)))
    fig.colorbar(cbar);

def visualize_solubilitymodule(structure_model, window_size = 21, title_legend = [], legend = [], figure_dims = ()):
    edge_pos = window_size//2
    pos_embed = np.linspace(-edge_pos, edge_pos, window_size).astype(float)
    pos_embed = np.expand_dims(np.expand_dims(pos_embed, -1), -1)
    n_features = structure_model.n_features + structure_model.additional_features - structure_model.cut_features
    hidden_state_dim = structure_model.hidden_state_dim

    params_to_extract = ["conv_ampl_0", "conv_ampl_1", "period_conv", 
                         "phase_conv", "mean_pos_conv", "std_pos_conv", 
                         "true_period", "solvent_access_w", "rotate_embeds"]
    params_to_extract = list(zip(params_to_extract, len(params_to_extract)*["mu"]))
    conv_ampl_0, conv_ampl_1, period_cos, 
    phase_cos, mean_pos, std_pos_conv, true_period, 
    solvent_access_w, rotate_embeds = structure_model.get_model_params(params_to_extract)
    solubility_module = structure_model.execution_order.SolubilityModule
    max_std_pos_conv = solubility_module.init_dict["max_std_pos_conv"]
    starting_period = solubility_module.init_dict["starting_period"]
    
    std_pos_conv = std_pos_conv**2 + 1/max_std_pos_conv**2
    period_conv = period_cos + starting_period
    true_period = 2*np.pi/np.array(true_period)
    fixed_periods = 1/(1+np.exp(-true_period*100))
    period_conv = true_period*fixed_periods + period_conv*(1-fixed_periods)
    
    exp_part = np.exp(-std_pos_conv*(pos_embed-mean_pos)**2)
    cos_part = np.cos(period_conv*pos_embed+phase_cos)
    conv_ampl_0 = conv_ampl_0[:1] + solvent_access_w
    conv_pairwise_0 = rotate_embeds@conv_ampl_0@(exp_part*cos_part)

    if len(figure_dims) < 2:
        figure_dims = factorize(hidden_state_dim)
    n_rows, n_cols = figure_dims
    
    jet_class = plt.get_cmap("jet", n_features)
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (n_rows*8, n_cols*8))
    for j in range(hidden_state_dim): 
        for k in range(n_features):
            ax[j//n_cols, j%n_cols].plot(conv_pairwise_0[..., k, j], c = jet_class(k))
        ax[j//n_cols, j%n_cols].set_xticks(np.arange(0, window_size, 2), np.arange(0, window_size, 2) - edge_pos)
        if len(title_legend) > 0:
            ax[j//n_cols, j%n_cols].set_title(title_legend[j])
        ax[j//n_cols, j%n_cols].grid()
    if len(legend) > 0:
        plt.legend(legend, loc = "lower center", bbox_to_anchor = [1.25, 1.2]);

def visualize_betweenclasscoherence(structure_model, window_size = 21, title_legend = [], legend = [], figure_dims = ()):
    edge_pos = window_size//2
    pos_embed = np.linspace(-edge_pos, edge_pos, window_size).astype(float)
    pos_embed = np.expand_dims(np.expand_dims(pos_embed, -1), -1)
    n_features = structure_model.n_features + structure_model.additional_features - structure_model.cut_features
    hidden_state_dim = structure_model.hidden_state_dim

    bcc_module = structure_model.execution_order.BetweenClassCoherence
    max_std_class_attn = bcc_module.init_dict["max_std_class_attn"]
    init_std_class_attn = bcc_module.init_dict["init_std_class_attn"]
    
    mean_class_attn, std_class_attn, ampl_class, zero_corr_class = structure_model.get_model_params([("class_pairwise", "mu")])[0]
    std_class_attn = (std_class_attn+1/init_std_class_attn)**2 + 1/max_std_class_attn**2
    exp_part_class = np.exp(-(std_class_attn*(pos_embed-mean_class_attn)**2))
    class_pairwise_ = exp_part_class*ampl_class
    class_pairwise_[edge_pos] = zero_corr_class

    if len(figure_dims) < 2:
        figure_dims = factorize(hidden_state_dim)
    n_rows, n_cols = figure_dims

    jet_class = plt.get_cmap("jet", hidden_state_dim)
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (n_rows*8, n_cols*8))
    for j in range(hidden_state_dim): 
        for k in range(hidden_state_dim):
            ax[j//n_cols, j%n_cols].plot(class_pairwise_[..., k, j], c = jet_class(k))
        ax[j//n_cols, j%n_cols].set_xticks(np.arange(0, window_size, 2), np.arange(0, window_size, 2) - edge_pos)
        if len(title_legend) > 0:
            ax[j//n_cols, j%n_cols].set_title(title_legend[j])
        ax[j//n_cols, j%n_cols].grid()
        if len(legend) > 0:
            ax[j//n_cols, j%n_cols].legend(legend, loc = "upper left");

def visualize_logits(seq, model, device, to_probs = False):
    max_length = len(seq)+2
    x_feats = np.zeros((1, max_length, 23), dtype = np.float32)
    x_feats[:, 0, 21] = 1
    x_feats[0, 1:len(seq)+1, :21] = np.reshape(np.array(list(seq)), (-1, 1)) == x_tokens
    x_feats[0, len(seq)+1, 22] = 1
    masks = np.zeros((1, max_length, 1), dtype = np.float32)
    masks[0, 1:len(seq)+1] = 1
    debug_dict = model(torch.FloatTensor(x_feats).to(device), torch.FloatTensor(masks).to(device), debugging = True)

    count_actives = 0
    fig, ax = plt.subplots(sum(model.active_modules.values())-1, 2, figsize = (sum(model.active_modules.values())*2, 20), width_ratios = [10, 1])
    for key in debug_dict.keys():
        if "class_logits" in key and "SharedVariable" not in key:
            if to_probs:
                matrix = F.softmax(debug_dict[key], dim = -1)
                manipulation_name = " as probabilities"
            else:
                matrix = F.log_softmax(debug_dict[key], dim = -1)
                manipulation_name = " as normalized logits"
            ax_i = ax[count_actives, 0]
            show_matrix = matrix.cpu().detach().numpy()[0, 1:-1].T
            percentiles = np.round(np.percentile(show_matrix, [2.5, 97.5]), 2)
            mappable = ax_i.matshow(show_matrix, vmin = percentiles[0],
                                    vmax = percentiles[1], cmap = "jet")
            fig.colorbar(mappable, pad = 0.025)
            ax_i.set_title(" ".join(key.split("_")[2:])+manipulation_name, fontsize = 7)
            ax_i.set_xticks([])
            ax_i.set_yticks([])
            
            percentiles_class = -np.percentile(show_matrix, [2.5, 50, 97.5], -1)
            ax_distr = ax[count_actives, 1]
            n_classes = len(percentiles_class[0])
            ax_distr.bar(np.arange(n_classes), percentiles_class[0], color = "r", alpha = 1)
            ax_distr.bar(np.arange(n_classes), percentiles_class[1], color = "g", alpha = 1)
            ax_distr.bar(np.arange(n_classes), percentiles_class[2], color = "b", alpha = 1)
    
            count_actives += 1
    fig.subplots_adjust(wspace = -0.2, hspace = 0.5);
