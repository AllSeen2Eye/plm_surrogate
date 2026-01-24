import copy
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, DistributedSampler
try:
    import torch_xla.distributed.parallel_loader as pl
except ModuleNotFoundError:
    pass     
from plm_surrogate.model.architecture import GeneralizedStructureModel
from plm_surrogate.data_prep.dataset import StructureDataset, ClassTokenizer, create_dataloader

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
        file_format = dataset_files["main_dataset"].split(".")[-1]
        if file_format == "csv":
            full_dataset = pd.read_csv(dataset_files["main_dataset"], index_col = "Index")
        elif file_format == "pkl":
            full_dataset = pd.read_pickle(dataset_files["main_dataset"])
            full_dataset.index.name = "Index"
        else:
            raise ValueError ("Incorrect file format for dataset import! Only .csv and .pkl are currently supported")
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
