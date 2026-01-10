### IMPORTS
import os
import copy
import json
import inspect
import warnings
from collections import OrderedDict

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.profiler import profile, ProfilerActivity, record_function

import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, DistributedSampler

try:
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    xla_device_available = True
except ModuleNotFoundError:
    xla_device_available = False

from pytorch_optimizer import SAM
import plm_surrogate.commons as commons

LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR = "./logs", "./models", "./checkpoints", "./trace", "./tmp"
folders_to_create = [LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR]
for select_DIR in folders_to_create:
    if not os.path.exists(select_DIR):
        os.mkdir(select_DIR)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)
print(f"Using {device} device")
scaler = torch.amp.GradScaler(device)

### DEFINE AND IMPORT CONSTANTS
def get_constants(aaprop_file_name, wp_file_name, n_features = 15):
    aa_data = pd.read_csv(aaprop_file_name, index_col="Name")

    for column_name in aa_data.columns:
        aa_data[column_name] = aa_data[column_name].astype(str).str.replace(",", ".")
    
    class_names = ["Unknown"]+sorted(aa_data["Class"].value_counts().index)
    aa_data["Class"] = list(map(lambda x: class_names.index(x), aa_data["Class"].tolist()))
    
    aa_data = aa_data.astype(float)
    aa_data['p(AA)'] = np.log(aa_data['p(AA)'])
    
    aa_data.columns = [column.replace("\\n", " ") for column in aa_data.columns]
    key_list = ['Atom Count', 'Mass [Dalton]', 'Volume [Å3]', 'Surface [Å2]', 'Hydrophobicity (pH 7)', 'p(AA)', 
                'pI at 25°C', 'pKa1', 'Dipole moment', 'pKa2', 'Aromaphilicity', 'EIIP', 'kappa1', 'kappa2', 
                'kappa3', 'In–out propensity', 'Amino acid propensity in Loops', 'Amino acid propensity in hinge', 
                'Amino acid propensity in alpha helices', 'Amino acid propensity in Beta sheets', 'Class']
    key_list = key_list[:n_features]
    orig_key_list = key_list.copy()

    aa_data_sorted = aa_data[key_list]
    aa_data_sorted_mean = aa_data_sorted.mean()
    aa_data_sorted_std = aa_data_sorted.std()
    aa_data_norm = (aa_data_sorted - aa_data_sorted_mean)/aa_data_sorted_std
    nat_embed = np.array(aa_data_norm, dtype = float)

    derive = len(wp_file_name) == 0
    if derive:
        pca = PCA(n_components=n_features)
        pca.fit(aa_data_norm)
        wp = pca.components_.T
    else:
        wp = np.load(wp_file_name) 
    return nat_embed, wp


### DATASET OBJECT AND FUNCTIONS
class StructureDataset(Dataset):
    def __init__(self, df, class_tokenizer, given_distr = False, max_len = None, precompute = False):
        self.df = df.copy()
        self.class_tokenizer = class_tokenizer
        self.n_tokens = np.sum(df["seq"].str.len())
        self.max_len = max_len
        self.precompute = precompute
        self.given_distr = given_distr
        if precompute:
            self.x, self.y, self.mask = [], [], []
            for idx in range(len(df)):
                outputs = self.compute_tensor(idx)
                self.x.append(outputs[0])
                self.y.append(outputs[1])
                self.mask.append(outputs[2])
            self.x = torch.stack(self.x, 0)
            self.y = torch.stack(self.y, 0)
            self.mask = torch.stack(self.mask, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.precompute:
            return self.compute_tensor(idx)
        else:
            return (self.x[idx], self.y[idx], self.mask[idx])

    def compute_tensor(self, idx):
        seq_col, y_col, other_col = ["seq", "label", "other"]

        patch = self.df.iloc[idx]
        seq, label, others = patch[seq_col], patch[y_col], patch[other_col]
        seq_len = len(seq)
        if self.max_len is None:
            max_len = seq_len+2
        else:
            max_len = self.max_len+2

        other_feats = others.shape[-1] if len(others) > 0 else 0
        x_feats = torch.zeros((max_len, 23+other_feats))
        masks = torch.zeros((max_len, 1))

        x_feats[0, 21] = 1
        x_feats[seq_len+1, 22] = 1
        x_feats[1:seq_len+1, :21] = torch.from_numpy(np.reshape(np.array(list(seq)), (-1, 1)) == commons.x_tokens).to(float)
        if other_feats > 0:
            x_feats[1:seq_len+1, 23:] = torch.from_numpy(others).to(float)
        
        y_true = self.class_tokenizer.tokenize(label, seq_len, max_len)
        masks[1:seq_len+1] = 1
        
        return (x_feats, y_true, masks)

class ClassTokenizer():
    def __init__(self, full_class_str, y_tokens, given_distr, 
                 bilinear = False, name = "Tokenizer"):
        self.full_class_str = full_class_str
        self.y_tokens = y_tokens
        self.given_distr = given_distr
        self.bilinear = bilinear
        self.n_dims = np.prod(y_tokens.shape) if given_distr else 1
        self.name = name
        
    def sequence_tokenizer(self, y_input, seq_len, max_len):
        y_true = torch.zeros((max_len, self.n_dims)).squeeze(dim=-1)
        if not self.given_distr: 
            if type(y_input) != str: 
                argmaxed_y = np.argmax(y_input, -1).tolist()
                y_input_list = list(map(lambda y:self.full_class_str[y], argmaxed_y))
                y_input = "".join(y_input_list) 
            y_onehot = np.reshape(np.array(list(y_input)), (-1, 1)) == self.y_tokens
            decide_label = np.argmax(y_onehot, -1) 
            y_input_true = torch.from_numpy(decide_label).to(int)
        else: 
            y_input_true = torch.FloatTensor(y_input).softmax(-1)
        y_true[1:seq_len+1] = y_input_true
        return y_true

    def bilinear_tokenizer(self, y_input, seq_len, max_len):
        y_true = torch.zeros((max_len, max_len, self.n_dims)).squeeze(dim=-1)
        y_input = np.fromstring(y_input)
        real_len = int(y_input.shape[0]**0.5)
        y_input = y_input.reshape(real_len, real_len, self.n_dims)
        if not self.given_distr: 
            y_input_true = torch.from_numpy(y_input).to(int)
        else: 
            y_input_true = torch.FloatTensor(y_input).softmax(-1)
        y_true[1:seq_len+1, 1:seq_len+1] = y_input_true
        return y_true
    
    def tokenize(self, y_input, seq_len, max_len):
        if self.bilinear:
            return self.bilinear_tokenizer(y_input, seq_len, max_len)
        else:
            return self.sequence_tokenizer(y_input, seq_len, max_len)
            
    def collate_fn(self, tensor_tuple):
        x_feats, y_true, masks = zip(*tensor_tuple)
        x_feats = pad_sequence(x_feats, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
            
        if self.bilinear:
            lenghts = [y_.shape[0] for y_ in y_true]
            pads = [max(lengths)-giv_len for giv_len in lengths]
            pads = [(0, 0, 0, pad_size, 0, pad_size) for pad_size in pads]
            y_true = torch.stack([F.pad(y_, p_, "constant", 0) for (y_, p_) in zip(y_true, pads)], 0)
        else:
            y_true = pad_sequence(y_true, batch_first=True, padding_value=0)
                
        return (x_feats, y_true, masks)

def create_dataset(dataset, tokenizer, sampler_fn = SequentialSampler, given_distr = False):
    dataset_obj = StructureDataset(dataset, tokenizer, given_distr = given_distr)
    sampler = sampler_fn(dataset_obj)
    return dataset_obj, sampler
    
def create_dataloader(dataset_obj, batch_size, sampler = None, 
                      num_workers = 0, dataloader_name = "Dataset"):
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, sampler=sampler,
                            collate_fn=dataset_obj.class_tokenizer.collate_fn, num_workers = num_workers)
    dataloader.name = dataloader_name
    return dataloader

def prepare_data(dataset, tokenizer, batch_size, given_distr = False, 
                 sampler_fn = SequentialSampler, num_workers = 0,
                 dataloader_name = "Dataset"):
    dataset_obj, sampler = create_dataset(dataset, tokenizer, sampler_fn, given_distr)
    dataloader = create_dataloader(dataset_obj, batch_size, sampler, num_workers, dataloader_name)
    return dataloader
    
    
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
            non_shifted = torch.randn(shape).to(device)
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
    device_type = pos_window.device.type
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


### ADDITIONAL FUNCTIONS (MODEL)    
def adjust_optim(optimizer_fn, lr, wd):
    optimizer_fn.param_groups[0]['weight_decay'] = wd
    optimizer_fn.param_groups[0]['lr'] = lr
    
def evaluate_model(y_pred, y_true, full_class_str, state_map_list = [], ax = None,
                   print_result = True, display_matrices = False):
    n_classes = len(full_class_str)
    if len(state_map_list) != n_classes:
        state_map_list = range(n_classes)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    state_map = dict(zip(range(n_classes), state_map_list))
    state_labels = sorted(list(set(state_map_list)))
    state_names = [full_class_str[state] for state in state_labels]
    predictions, y_trues = pd.Series(y_pred).map(state_map), pd.Series(y_true).map(state_map),
    accuracy = accuracy_score(y_trues, predictions)
    class_report = classification_report(y_trues, predictions, zero_division=0,
                                         labels = state_labels, target_names = state_names)
    if print_result:
        print("Dataset Accuracy:", accuracy)
        print("\nClassification Report:\n")
        print(class_report)
    if display_matrices:
        assert not (ax is None)
        display_labels = [full_class_str[state] for state in set(list(predictions) + list(y_trues))]
        cm = confusion_matrix(y_trues, predictions)
        cmdisp = ConfusionMatrixDisplay(cm, display_labels = display_labels)
        cmdisp.plot(cmap = "magma", ax = ax, colorbar = False, values_format='')
        return ax
    return class_report
   
   
### ADDITIONAL FUNCTIONS FOR LOGGING, DEBUGGING AND TRACING
from functools import wraps
def interruption_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt: cancelling function {func.__name__}")
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if "debug_dict" in bound_args.arguments:
                return bound_args.arguments["debug_dict"]
            else:
                raise ValueError(f"Argument 'debug_dict' not found in function call.")
    return wrapper

def check_for_NaNs(given_dict):
    errors_not_found = True
    for key, value in given_dict.items():
        if torch.sum(value.isnan()) > 0:
            if errors_not_found:
                print("---------------- ERRORS FOUND!!! ----------------")
                errors_not_found = False
            error_slices = value.cpu().detach().numpy()
            error_ids = np.argwhere(np.isnan(error_slices))
            print(key, "| dims =", error_slices.ndim)
            for dim in range(error_slices.ndim):
                set_of_error_ids = set(error_ids[:, dim])
                if len(set_of_error_ids) <= batch_size:
                    print("Shape:", error_slices.shape, "| Dims:", dim, "| Error IDs:", set_of_error_ids)
            print("-------------------------------------------------")
    if errors_not_found:
        print("RESULT: Errors NOT Found!!!")

def diagnosing_NaN(model_obj, error_dict):
    check_for_NaNs(error_dict)
    for key, value in model_obj.state_dict().items():
        if torch.sum(value.isnan()) > 0:
            error_slices = value.cpu().detach().numpy()
            error_ids = np.argwhere(np.isnan(error_slices))
            print(key, "dims = ", error_slices.ndim)
            for dim in range(error_slices.ndim):
                set_of_error_ids = set(error_ids[:, dim])
                if len(set_of_error_ids) <= batch_size:
                    print(error_slices.shape, dim, set_of_error_ids)
    check_for_NaNs(model_obj.state_dict())
    for key, val in model_obj.state_dict().items():
        print(key, torch.max(torch.abs(val)).item(), torch.std(val).item())

def call_profiler(model, dataset_profiler, profiler_options):
    with profile(**profiler_options) as profiler_context:
            with record_function("model_inference"):
                model.inference_model(dataset_profiler)
    sort_by = f"{device.type.lower()}_time_total"
    with open(f"{TRC_DIR}/model_profile.log", "w") as profiler_logs:
        profiler_logs.write(profiler_context.key_averages().table(sort_by=sort_by))
    profiler_context.export_chrome_trace(f"{TRC_DIR}/model_trace.json")
    

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
    def __init__(self, n_features, hidden_state_dim, 
                 model_type, max_std_sigma, init_dict = {}): 
        super().__init__()         
        
        self.init_dict = {"max_std_sigma":max_std_sigma}
        feat_i_classes = n_features + hidden_state_dim
        downsample_feats = feat_i_classes
        names = ["all_x_all_ampl", "inp_decomp_vecs", 
                 "out_decomp_vecs", "all_x_all_bias", 
                 "knn_activations"] 
        shapes = [(5, downsample_feats, hidden_state_dim), (5, downsample_feats, 1), 
                  (5, 1, hidden_state_dim), (3, 1, hidden_state_dim), (2, feat_i_classes, downsample_feats)] 
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
    
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        class_attn_rec_0, class_attn_rec_1 = lp.all_x_all_ampl[0], lp.all_x_all_ampl[1], 
        class_attn_rec_2, class_attn_rec_3 = lp.all_x_all_ampl[2], lp.all_x_all_ampl[3:]   
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
                 max_std_sigma_0, max_std_sigma_1, 
                 init_dict = {}): 
        super().__init__() 
        
        self.init_dict = {"max_std_sigma_0":max_std_sigma_0,
                          "max_std_sigma_1":max_std_sigma_1}
        feat_i_classes = n_features + hidden_state_dim
        downsample_feats = feat_i_classes
        names = ["all_x_all_ampl_1", "inp_decomp_vecs_1", 
                 "out_decomp_vecs_1", "all_x_all_bias_1", 
                 "pos_dep_ampl_weights_1", "knn_activations_1"] 
        shapes = [(5, downsample_feats, hidden_state_dim), (5, downsample_feats, 1), 
                  (5, 1, hidden_state_dim), (5, 1, hidden_state_dim), 
                  (1, feat_i_classes, hidden_state_dim), (2, feat_i_classes, downsample_feats)] 
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
        
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        class_attn_rec_0, class_attn_rec_1 = lp.all_x_all_ampl_1[0], lp.all_x_all_ampl_1[1], 
        class_attn_rec_2, class_attn_rec_3 = lp.all_x_all_ampl_1[2], lp.all_x_all_ampl_1[3:]   
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
                 max_std_sigma_0, max_std_sigma_1, 
                 init_dict = {}): 
        super().__init__() 
        
        self.init_dict = {"max_std_sigma_0":max_std_sigma_0,
                          "max_std_sigma_1":max_std_sigma_1}
        feat_i_classes = n_features + hidden_state_dim
        downsample_feats = feat_i_classes
        names = ["all_x_all_ampl_2", "inp_decomp_vecs_2", 
                 "out_decomp_vecs_2", "all_x_all_bias_2", 
                 "pos_dep_ampl_weights_2", "knn_activations_2"] 
        shapes = [(5, downsample_feats, hidden_state_dim), (5, downsample_feats, 1), 
                  (5, 1, hidden_state_dim), (5, 1, hidden_state_dim), 
                  (1, feat_i_classes, hidden_state_dim), (2, feat_i_classes, downsample_feats)] 
        self.parameter_list = create_parameter_list(names, shapes, model_type, init_dict) 
    
    def forward(self, input_dict): 
        lp = unpack_parameter_list(self.parameter_list) #local_params 
        class_attn_rec_0, class_attn_rec_1 = lp.all_x_all_ampl_2[0], lp.all_x_all_ampl_2[1], 
        class_attn_rec_2, class_attn_rec_3 = lp.all_x_all_ampl_2[2], lp.all_x_all_ampl_2[3:]   
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
                 value_feats, n_heads, n_layers, n_landmarks, max_laplace_std_attn, init_dict = {}):
        super().__init__()

        self.init_dict = {"max_laplace_std_attn": max_laplace_std_attn, 
                          "n_landmarks": n_landmarks}
        feat_i_classes = n_features + hidden_state_dim
        names = ["collect_classes", "query_linear", "key_linear", 
                 "value_linear", "project_original", "aggregation_params"]
        shapes = [(1, value_feats*n_heads, hidden_state_dim),
                  (n_layers, 1, n_heads, feat_i_classes+1, downsampling_feats),
                  (n_layers, 1, 1, feat_i_classes+1, downsampling_feats),
                  (n_layers, 1, n_heads, feat_i_classes+1, value_feats),
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


### CONFIG FOR MODEL AND TRAINING
class StructureModelConfig():
    def __init__(self, config = {}):
        self.config = config
        
    def from_dict(self, model_kwargs, dataset_kwargs, training_kwargs):
        self.config["model"] = copy.deepcopy(model_kwargs)
        self.config["dataset"] = copy.deepcopy(dataset_kwargs)
        self.config["training"] = copy.deepcopy(training_kwargs)
        return self
    
    def to_dict(self):
        return self.config["model"], self.config["dataset"], self.config["training"]
    
    def from_json(self, file_name):
        with open(file_name, "r") as json_file:
            self.config = json.load(json_file)
        return self
    
    def to_json(self, file_name, **dump_kwargs):
        with open(file_name, "w") as json_file:
            json.dump(self.config, json_file, **dump_kwargs)
            
    def get_init_dict(self, nat_embed, rotate_embeds, true_period, solvent_access_w):
        n_features = self["model/n_features"]
        additional_features = self["model/additional_features"]
        bias_embeds = np.array([0]*n_features)
        
        embeds_dict = {"nat_embed":nat_embed, "rotate_embeds":rotate_embeds, "bias_embeds":bias_embeds, 
                       "true_period":true_period, "solvent_access_w":solvent_access_w, }  
        init_dict_file = self["model/init_dict_file"]
        if len(init_dict_file) > 0:
            init_dict = dict(np.load(init_dict_file)) 
        else:
            init_dict = {}
        init_dict = init_dict | embeds_dict
        init_dict = {key:np.array(value).astype(np.float64) for key, value in init_dict.items()}
        
        return init_dict
        
    def get_module_init_dict(self):
        module_order = self["model/module_order"]
        module_hyperparams = self["model/module_hyperparams"]
        module_init_dict = OrderedDict()
        
        for module_name in module_order:
            local_module_params = module_hyperparams.get(module_name, {})
            local_module_init = module_inits[module_name]
            module_init_dict[module_name] = (local_module_init, local_module_params)
            
        return module_init_dict
        
    def config_model_init(self):
        model_init_dict = copy.deepcopy(self.config["model"])
        embedding_configs = model_init_dict.pop("embedding_configs")
        model_init_dict["init_dict"] = self.get_init_dict(**embedding_configs)
        model_init_dict["module_init_dict"] = self.get_module_init_dict()
        
        not_needed = ["init_dict_file", "module_order", "module_hyperparams"]
        for key in not_needed:
            del model_init_dict[key]
        
        return model_init_dict

    def get_per_epoch_data(self):
        set_model_grad = self["model/set_model_grad"]
        
        active_list = self["training/active_list"]
        train_list = self["training/train_list"]
        reset_state_list = self["training/reset_state_list"]
        epochs_list = self["training/epochs_list"]
        
        learning_rates = self["training/learning_rates"]
        learning_rates_multipliers = self["training/learning_rates_multipliers"]
        weight_decays = self["training/weight_decays"]
        weight_decays_multipliers = self["training/weight_decays_multipliers"]
        
        fake_model_init_dict = self.config_model_init()
        fake_model_init_dict["set_grad_kwargs"]["verbose"] = False 
        fake_model = GeneralizedStructureModel(**fake_model_init_dict)
        all_module_names = ["SharedVariableModule"] + self["model/module_order"]
        
        use_module_per_epoch, set_grad_array, reset_state, lrs, wds = [], [], [], [], []
        for element_num in range(len(active_list)):
            new_active_modules = active_list[element_num]
            new_trained_modules = train_list[element_num]
            epochs = epochs_list[element_num]
            set_grad_dict = {}
            allow_module_to_train = dict(zip(all_module_names, new_trained_modules))
            
            for module_name, use_module in allow_module_to_train.items():
                module_fn = fake_model.execution_order[module_name]
                set_grad_dict_keys = [key.split(".")[-2] for key in set(list(module_fn.state_dict().keys()))]
                set_grad_dict_keys = set(set_grad_dict_keys) - set(["features2class", "conv_ampl_1"])
                set_grad_dict_vals = len(set_grad_dict_keys)*[use_module > 1e-08]
                set_grad_dict = set_grad_dict | dict(zip(list(set_grad_dict_keys), set_grad_dict_vals))
                set_grad_dict["features2class"] = allow_module_to_train["PhysioChemical2Class"] > 1e-08
                set_grad_dict["conv_ampl_1"] = allow_module_to_train["SolubilityModule"] > 1e-08
        
            set_grad_array = set_grad_array + [set_grad_dict | set_model_grad]*epochs
            use_module_dict = dict(zip(all_module_names, new_active_modules))
            use_module_per_epoch = use_module_per_epoch + [use_module_dict]*epochs
            reset_state = reset_state + [reset_state_list[element_num]]*epochs
            
        for i, epochs in enumerate(epochs_list):
            slice_lrs = np.array(learning_rates[:epochs])
            slice_multiplier = learning_rates_multipliers[i]
            lrs = lrs + (slice_lrs*slice_multiplier).tolist()
        
        for i, epochs in enumerate(epochs_list):
            slice_wds = np.array(weight_decays[:epochs])
            slice_multiplier = weight_decays_multipliers[i]
            wds = wds + (slice_wds*slice_multiplier).tolist()
        per_epoch_args = {"lrs":lrs, "wds":wds, "reset_state":reset_state, 
                          "use_module_per_epoch":use_module_per_epoch, 
                          "set_grad_array":set_grad_array}
        return per_epoch_args

    def preprocess_fn(self, dataset_files, dataset_labels = ["train", "score", "valid", "test"]):
        full_dataset = pd.read_csv(dataset_files["main_dataset"], index_col = "Index")
        full_dataset.index = list(map(str, full_dataset.index))
        full_dataset = full_dataset[full_dataset["seq"].str.len() < 1022]

        full_dataset["other"] = [[]]*len(full_dataset)
        other_feat_files = dataset_files["other_feats"]
        if len(other_feat_files) > 0:
            other_feat_dict = {}
            load_npzs = [np.load(file_name, allow_pickle = True) for file_name in other_feat_files] 
            for key in full_dataset.index:
                full_feats = np.concatenate([npz[key] for npz in load_npzs], axis = -1)
                other_feat_dict[key] = full_feats
            full_dataset["other"] = pd.Series(other_feat_dict).loc[full_dataset.index]

        all_datasets = []
        for dataset_label in dataset_labels:
            ds_slice = full_dataset[full_dataset["subset_type"] == dataset_label][["seq", "label", "other"]]
            all_datasets.append(ds_slice)

        return all_datasets
        
    def config_dataset_init(self):
        dataset_init_dict = copy.deepcopy(self["dataset"])
        dataset_files = dataset_init_dict.pop("dataset_files")
        ds_train, ds_score, ds_valid, ds_test = self.preprocess_fn(dataset_files)

        given_distr = dataset_init_dict.pop("given_distr")
        if given_distr: 
            dataset_esm = dict(np.load(dataset_files["distr_data"]))
            ds_train["label"] = pd.Series(dataset_esm).loc[ds_train.index]

        full_class_str = dataset_init_dict.pop("full_class_str")
        y_tokens = np.array(dataset_init_dict.pop("y_tokens"))
        bilinear = dataset_init_dict.pop("bilinear")
        all_datasets = [ds_train, ds_score, ds_valid, ds_test]
        for j, dataset in enumerate(all_datasets):
            dataset_given_distr = given_distr and j == 0
            tokenizer = ClassTokenizer(full_class_str, y_tokens, dataset_given_distr, bilinear)
            dataset_init_dict["class_tokenizer"] = tokenizer
            all_datasets[j] = StructureDataset(dataset, given_distr = given_distr and j == 0, **dataset_init_dict)
        
        return all_datasets
        
    def config_training_init(self, sent_device, num_workers):
        training_init_dict = copy.deepcopy(self.config["training"])
        batch_size = training_init_dict.pop("batch_size")
        train_obj, scoring_train_obj, valid_obj, test_obj = self.config_dataset_init()
        
        train_dataloader = create_dataloader(train_obj, batch_size, RandomSampler(train_obj), num_workers = num_workers, dataloader_name = "Train")
        scoring_train_dataloader = create_dataloader(scoring_train_obj, batch_size, SequentialSampler(scoring_train_obj), dataloader_name = "Train")
        valid_dataloader = create_dataloader(valid_obj, batch_size, SequentialSampler(valid_obj), dataloader_name = "Valid")
        test_dataloader = create_dataloader(test_obj, batch_size, SequentialSampler(test_obj), dataloader_name = "Test")
    
        if sent_device.type == "xla":
            train_dataloader = pl.MpDeviceLoader(train_dataloader, sent_device)
            scoring_train_dataloader = pl.MpDeviceLoader(scoring_train_dataloader, sent_device)
            valid_dataloader = pl.MpDeviceLoader(valid_dataloader, sent_device)
            test_dataloader = pl.MpDeviceLoader(test_dataloader, sent_device)
    
        training_init_dict["datasets"] = (train_dataloader, scoring_train_dataloader, valid_dataloader, test_dataloader)    
        per_epoch_args = self.get_per_epoch_data()
        for key, value in per_epoch_args.items():
            training_init_dict[key] = value
    
        not_needed = ["output_name", "active_list", "train_list", 
                      "reset_state_list", "epochs_list",
                      "learning_rates", "learning_rates_multipliers", 
                      "weight_decays", "weight_decays_multipliers"]
        for key in not_needed:
            del training_init_dict[key]
        return training_init_dict
    
    def __getitem__(self, idx):
        dirs_from_idx = idx.split("/")
        opened_dir = self.config
        for search_dir in dirs_from_idx:
            opened_dir = opened_dir[search_dir]
        return opened_dir
        
    def __setitem__(self, idx, new_value):
        dirs_from_idx = idx.split("/")
        opened_dir = self.config
        final_element = len(dirs_from_idx)-1
        for elem_num, search_dir in enumerate(dirs_from_idx):
            if elem_num == final_element:
                opened_dir[search_dir] = new_value
            else:
                opened_dir = opened_dir[search_dir]
    
    def __len__(self, split = False):
        config_lens = {key:len(self[key]) for key in self.config.keys()}
        if split: 
            return config_lens
        else:
            return sum(config_lens.values())
   
   
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

        data_dict = {"x":x_feats[..., :23], "x_other":x_feats[..., 23:], 
                     "masks":masks, "pos_embed":pos_embed, "class_outputs":output_vec, 
                     "latent_outputs":latent_vec, "device":device}

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
        
    def inference_model(self, dataloader, return_logits = False, enable_grad = False, batch_fn = None):
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
            
            for (x_feats, y_true, masks) in tqdm(dataloader):
                    
                x_feats, y_true, masks = x_feats.to(device), y_true.to(device), masks.to(device)
                y_pred = self(x_feats, masks, return_logits = return_logits)
                
                if batch_fn is not None:
                    batch_fn(x_feats, masks, y_pred)

                select_parts_pred = y_pred.reshape(-1, base_n_classes)
                if not return_logits and base_n_classes > 1:
                    select_parts_pred = torch.argmax(select_parts_pred, dim = -1, keepdim = True)
                elif not return_logits and base_n_classes == 1:
                    select_parts_pred = (select_parts_pred > 0.5).to(int)
                predictions.append(select_parts_pred)
                y_trues.append(y_true.reshape(-1, 1))
                real_mask = masks if not self.bilinear else masks*masks.permute(0, 2, 1)
                masks_count.append(real_mask.reshape(-1, ))

                if device_type == "xla":
                    xm.mark_step()

            predictions, y_trues = torch.cat(predictions, 0), torch.cat(y_trues, 0)
            masks_count = torch.cat(masks_count, 0)
            predictions = predictions[masks_count == 1].cpu().detach().numpy()
            y_trues = y_trues[masks_count == 1].cpu().detach().numpy()
            return predictions, y_trues
      
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

        calibration_filter = lambda name: "mu" in name if not self.calibrated else True
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
        param_size = {}
        
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
        

### TRAIN HANDLING
@interruption_handler
def train_model(model, datasets, lrs, wds, reset_state, use_sam, use_module_per_epoch, set_grad_array = [], 
                sent_device = torch.device("cpu"), class_weights = None, score_beginning = True,
                print_interm = False, ckpt_config = {}, profiler_options = {}, debug_dict = None, rank = 0):
    train_dataloader, scoring_train_dataloader, valid_dataloader, test_dataloader = datasets
    
    if sent_device.type != "xla":
        get_train_dataset = train_dataloader.dataset
        cast_dtype = torch.float16
    else:
        get_train_dataset = train_dataloader._loader.dataset
        cast_dtype = torch.bfloat16
    if use_sam:
        cast_dtype = torch.float32 

    all_rows = len(get_train_dataset)
    N_tokens = get_train_dataset.n_tokens
    given_distr = get_train_dataset.given_distr

    unwrapped_model = lambda model: model.module if isinstance(model, nn.DataParallel) else model
    base_n_classes = unwrapped_model(model).n_classes
    model_type = unwrapped_model(model).model_type
    bilinear = unwrapped_model(model).bilinear
    
    if class_weights is not None:
        class_weights = torch.from_numpy(np.array(class_weights)).to(float).to(sent_device)
    if not given_distr and base_n_classes > 1:
        loss_fn = nn.CrossEntropyLoss(reduction = "none", weight = class_weights)
    elif not given_distr and base_n_classes == 1:
        loss_fn = nn.BCEWithLogitsLoss(reduction = "none", weight = class_weights)
    else: 
        loss_fn = nn.KLDivLoss(reduction = "none")
        
    if not use_sam:
        optimizer_fn = torch.optim.AdamW(model.parameters(), lr = 0, weight_decay = 0)
    else:
        base_optimizer = torch.optim.AdamW
        optimizer_fn = SAM(model.parameters(), base_optimizer, lr = 0, weight_decay = 0)
        
    epochs = min(len(lrs), len(wds), len(use_module_per_epoch), len(reset_state))
    if print_interm is False:
        print_interm = epochs+1
    ckpt_frequency = ckpt_config.get("ckpt_frequency", epochs+1)
    ckpt_filename = ckpt_config.get("ckpt_filename", unwrapped_model(model).model_name)
    ckpt_filename = f"{CKPT_DIR}/{ckpt_filename}_epoch.pt"
    debugging = type(debug_dict) == dict
    
    if score_beginning:
        unwrapped_model(model).score_model(scoring_train_dataloader, valid_dataloader, test_dataloader)
        
    for epoch in range(epochs):
        cumloss, cumacc, cumcount, cumpriorloss, done_rows = 0, 0, 0, 0, 0
        if reset_state[epoch]:
            optimizer_fn = type(optimizer_fn)(params = optimizer_fn.param_groups, defaults = optimizer_fn.defaults)
        else:
            adjust_optim(optimizer_fn, lrs[epoch], wds[epoch])
        if len(set_grad_array) == epochs:   
            unwrapped_model(model).set_model_grad(set_grad_array[epoch], verbose = False)
        
        old_active_modules = unwrapped_model(model).active_modules
        epoch_active_modules = use_module_per_epoch[epoch]
        new_is_same_as_old = True
        for module_name in list(old_active_modules.keys()):
            old_module_state = old_active_modules[module_name]
            new_module_state = epoch_active_modules[module_name]
            new_is_same_as_old *= (old_module_state == new_module_state)
        if not new_is_same_as_old or epoch == 0:
            unwrapped_model(model).active_modules = epoch_active_modules

        for (x_feats, y_true, masks) in train_dataloader:
            x_feats, y_true, masks = x_feats.to(sent_device), y_true.to(sent_device), masks.to(sent_device)
            for step in range(int(use_sam)+1):
                with autocast(device_type = sent_device.type, enabled = True, dtype = cast_dtype):
                    y_pred = model(x_feats, masks, return_prev_compute = debugging)
                    if debugging:
                        debug_dict |= y_pred
                        y_pred = y_pred["class_outputs"]
                    if not given_distr and base_n_classes > 1:
                        y_true = y_true.to(int)
                    elif not given_distr and base_n_classes == 1:
                        y_true = y_true.to(cast_dtype)
                    else:
                        y_pred = F.log_softmax(y_pred, dim = -1)
    
                    metrics_mask = masks.clone().flatten(0, 1) if not bilinear else (masks*masks.permute(0, 2, 1)).flatten()
                    given_distr_shape = (-1, ) if not given_distr else (-1, base_n_classes)
                    loss_val_unmasked = loss_fn(y_pred.reshape((-1, base_n_classes)).squeeze(-1), 
                                                y_true.reshape(given_distr_shape))
                    loss_val_unmasked = loss_val_unmasked.reshape((metrics_mask.shape[0], -1))
                    loss_val = torch.sum(loss_val_unmasked*metrics_mask)/torch.sum(metrics_mask)
                    prior_loss_component = unwrapped_model(model).get_prior_loss()
                    loss_val_optim = loss_val*N_tokens + prior_loss_component
                
                if sent_device.type != "xla":
                    loss_is_nan = loss_val.isnan()
                    if loss_is_nan:
                        if debugging:
                            print ("\nWARNING: Loss is NaN")
                            return debug_dict
                        else:
                            raise ValueError ("Loss is NaN")
                
                if sent_device.type in {"cuda", "cpu", "mps"}:
                    scaler.scale(loss_val_optim).backward()
                    if not use_sam:
                        scaler.step(optimizer_fn)
                    else: 
                        scaler.unscale_(optimizer_fn)
                        if step == 0:
                            optimizer_fn.first_step()
                        else:
                            optimizer_fn.second_step()
                    scaler.update()
                    optimizer_fn.zero_grad()
    
                elif sent_device.type == "xla":
                    loss_val_optim.backward()
                    xm.optimizer_step(optimizer_fn)
                    optimizer_fn.zero_grad()
                    xm.mark_step()

            if given_distr:
                y_true = torch.argmax(y_true, axis = -1)

            y_pred_to_idx = torch.argmax(y_pred, dim = -1) if base_n_classes > 1 else (y_pred > 0.0).to(int)[..., 0]
            real_masks = masks[..., 0] if not bilinear else masks*masks.permute(0, 2, 1)
            acc_val = torch.sum((y_true == y_pred_to_idx)*real_masks)
            cumacc += acc_val
            cumcount += torch.sum(real_masks)
            done_rows += y_true.shape[0]
            add_format_string = ""
            
            if model_type in ["mean_field", "linear_laplace"]:
                cumpriorloss += prior_loss_component*torch.sum(real_masks)/N_tokens
                local_pl = torch.round((cumpriorloss/cumcount).to(torch.float64), decimals = 3)
                add_format_string = f", with average prior loss: {local_pl:.3f}"
            cumloss += loss_val*torch.sum(real_masks)
            
            local_loss = torch.round((cumloss/cumcount).to(torch.float64), decimals = 3)
            local_acc = torch.round((cumacc/cumcount).to(torch.float64), decimals = 3)
            if rank == 0:
                print(f"\rLoss for epoch {epoch+1} row {done_rows} out of {all_rows}: {local_loss:.3f}, accuracy: {local_acc:.3f}" + add_format_string, end = "")
                    
        if ((epoch+1) % print_interm) == 0:
            print("\n")
            unwrapped_model(model).score_model(scoring_train_dataloader, valid_dataloader, test_dataloader)
            if len(profiler_options) > 0 and (epoch+1) == print_interm:
                call_profiler(unwrapped_model(model), test_dataloader, profiler_options)
        
        if ((epoch+1) % ckpt_frequency) == 0:
            epoch_ckpt_name = ckpt_filename.replace("epoch", str(epoch+1))
            unwrapped_model(model).save_model(epoch_ckpt_name)
            print(f"Saved checkpoint for model at {epoch_ckpt_name}")
            
    debug_dict["model"] = unwrapped_model(model)   
    return {}
    
def generalized_train_fn(rank, world_size, sent_device, config_obj, debug_dict):
    if sent_device is None:
        sent_device = xm.xla_device()
    device_type = sent_device.type
    multiple_gpus = (world_size > 1) and device_type == "cuda"

    model_init_dict = config_obj.config_model_init()
    structure_model = GeneralizedStructureModel(**model_init_dict)
    structure_model = structure_model.to(sent_device)
    if multiple_gpus:
        print("DataParallel triggered!")
        structure_model = nn.DataParallel(structure_model)

    training_init_dict = config_obj.config_training_init(sent_device, world_size)
    output_name = config_obj["training/output_name"]
    json_output = ".".join(output_name.split(".")[:-1] + ["json"])
    config_obj.to_json(json_output)
    debug_dict = train_model(model = structure_model, sent_device = sent_device, 
                             debug_dict = debug_dict, **training_init_dict)
    
    if "error_y_pred" in debug_dict.keys():
        diagnosing_NaN(structure_model, debug_dict)
        raise ValueError ("Loss is NaN!")
    if multiple_gpus:
        structure_model.module.save_model(output_name)
    else:
        structure_model.save_model(output_name)
    debug_dict["model"] = structure_model 
    return debug_dict

def train_handler(config_obj, debug_dict):
    print("Train Handler is on!")
    sent_device = None if xla_device_available else device
    device_type = "xla" if xla_device_available else sent_device.type

    if device_type == "cuda":
        world_size = torch.cuda.device_count()
    elif device_type == "xla":
        world_size = 1
    else:
        world_size = 1

    print("Current world_size:", world_size, "for device", device_type)
    arg_list = (world_size, sent_device, config_obj, debug_dict)
    if not device_type == "xla":
        return generalized_train_fn(0, *arg_list)
    else:
        xmp.spawn(generalized_train_fn, args=arg_list, nprocs=world_size)
    return {}
