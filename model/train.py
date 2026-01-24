### IMPORTS
import os
import inspect

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.profiler import profile, ProfilerActivity, record_function

try:
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ModuleNotFoundError:
    pass

from pytorch_optimizer import SAM
from plm_surrogate.model.architecture import GeneralizedStructureModel

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

def call_profiler(model, dataset_profiler, profiler_options, trc_dir):
    with profile(**profiler_options) as profiler_context:
            with record_function("model_inference"):
                model.inference_model(dataset_profiler)
    device_name = next(model.parameters()).device.type.lower()
    sort_by = f"{device_name}_time_total"
    with open(f"{trc_dir}/model_profile.log", "w") as profiler_logs:
        profiler_logs.write(profiler_context.key_averages().table(sort_by=sort_by))
    profiler_context.export_chrome_trace(f"{trc_dir}/model_trace.json")

@interruption_handler
def train_model(model, datasets, lrs, wds, reset_state, use_sam, use_module_per_epoch, set_grad_array = [], sent_device = torch.device("cpu"), 
                class_weights = None, score_beginning = True, print_interm = False, ckpt_config = {}, profiler_options = {}, 
                folder_options = {"ckpt_dir":"./ckpt", "trc_dir":"./trc"}, debug_dict = None, rank = 0):
    train_dataloader, scoring_train_dataloader, valid_dataloader, test_dataloader = datasets
    
    if sent_device.type != "xla":
        get_train_dataset = train_dataloader.dataset
        cast_dtype = torch.float16
        scaler = torch.amp.GradScaler(sent_device)
    else:
        get_train_dataset = train_dataloader._loader.dataset
        cast_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(torch.device("cpu"))
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
        
    optim_params = dict(params = model.parameters(), lr = 0, weight_decay = 0)
    if not use_sam:
        optimizer_fn = torch.optim.AdamW(**optim_params)
    else:
        optim_params["base_optimizer"] = torch.optim.AdamW
        optim_params["adaptive"] = True
        optim_params["rho"] = 0.1
        optimizer_fn = SAM(**optim_params)
        
    epochs = min(len(lrs), len(wds), len(use_module_per_epoch), len(reset_state))
    if print_interm is False:
        print_interm = epochs+1
    ckpt_frequency = ckpt_config.get("ckpt_frequency", epochs+1)
    ckpt_filename = ckpt_config.get("ckpt_filename", unwrapped_model(model).model_name)
    ckpt_filename = f"{folder_options['ckpt_dir']}/{ckpt_filename}_epoch.pt"
    debugging = type(debug_dict) == dict
    
    if score_beginning:
        unwrapped_model(model).score_model(scoring_train_dataloader, valid_dataloader, test_dataloader)
        
    for epoch in range(epochs):
        cumloss, cumacc, cumcount, cumpriorloss, done_rows = 0, 0, 0, 0, 0
        if reset_state[epoch]:
            optim_params["params"] = model.parameters()
            optimizer_fn = type(optimizer_fn)(**optim_params)
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
                call_profiler(unwrapped_model(model), test_dataloader, profiler_options, folder_options['trc_dir'])
        
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
    output_dir = training_init_dict["folder_options"]["output_dir"]
    model_fname = config_obj["training/output_name"]
    output_name = f'{output_dir}/{model_fname}'
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

def train_handler(config_obj, debug_dict, device_type):
    print("Train Handler is on!")
    sent_device = None if device_type == "xla" else torch.device(device_type)

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
