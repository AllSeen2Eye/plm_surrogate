import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import plm_surrogate.commons as commons


class StructureDataset(Dataset):
    def __init__(self, data_source, class_tokenizer, given_distr = False, max_len = None, precompute = False):
        self.data_source = data_source.copy()
        self.class_tokenizer = class_tokenizer
        self.n_tokens = np.sum(data_source["seq"].str.len())
        self.max_len = max_len
        self.precompute = precompute
        self.given_distr = given_distr
        self.supervised = class_tokenizer is not None
        
        if precompute:    
            n_vectors = 2+int(supervised)
            self.data = [[]]*n_vectors
            for idx in range(len(data_source)):
                data_slice = self.compute_tensor(idx)
                [self.data[i].append(data_slice[i]) for i in range(n_vectors)]
            self.data = [torch.stack(self.data[i], 0) for i in range(n_vectors)]

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if not self.precompute:
            return self.compute_tensor(idx)
        else:
            return [self.data[i][idx] for i in range(n_vectors)]
             

    def compute_tensor(self, idx):
        seq_col, y_col, other_col = ["seq", "label", "other"]

        patch = self.data_source.iloc[idx]
        seq, label, others = patch[seq_col], patch[y_col], patch[other_col]
        if self.max_len is None:
            max_len = len(seq)+2
        else:
            max_len = self.max_len+2

        x_feats, masks = commons.tokenize_aminoacid_sequence(seq, others, max_len)
        if self.supervised:
            y_true = self.class_tokenizer.tokenize(label, len(seq), max_len)        
            return (x_feats, y_true, masks)
        return (x_feats, masks)

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
        y_input = np.array(y_input)
        real_len = y_input.shape[0]
        y_input = y_input.reshape(real_len, real_len, self.n_dims)
        if not self.given_distr: 
            y_input_true = torch.from_numpy(y_input).to(int)
        else: 
            y_input_true = torch.FloatTensor(y_input).softmax(-1)
        y_true[1:seq_len+1, 1:seq_len+1] = y_input_true.squeeze(dim=-1)
        return y_true
    
    def tokenize(self, y_input, seq_len, max_len):
        if self.bilinear:
            return self.bilinear_tokenizer(y_input, seq_len, max_len)
        else:
            return self.sequence_tokenizer(y_input, seq_len, max_len)

def collate_fn(tensor_tuple, bilinear):
    outputs = zip(*tensor_tuple)
    supervised = len(outputs) > 2
    if supervised:
        x_feats, y_true, masks = outputs
    else:
        x_feats, masks = outputs
    x_feats = pad_sequence(x_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    if supervised:
        if bilinear:
            lenghts = [y_.shape[0] for y_ in y_true]
            pads = [max(lengths)-giv_len for giv_len in lengths]
            pads = [(0, 0, 0, pad_size, 0, pad_size) for pad_size in pads]
            y_true = torch.stack([F.pad(y_, p_, "constant", 0) for (y_, p_) in zip(y_true, pads)], 0)
        else:
            y_true = pad_sequence(y_true, batch_first=True, padding_value=0)
        return x_feats, y_true, masks
        
    return x_feats, masks

def create_dataset(dataset, tokenizer, sampler_fn = SequentialSampler, given_distr = False):
    dataset_obj = StructureDataset(dataset, tokenizer, given_distr = given_distr)
    sampler = sampler_fn(dataset_obj)
    return dataset_obj, sampler
    
def create_dataloader(dataset_obj, batch_size, sampler = None, 
                      num_workers = 0, dataloader_name = "Dataset"):
    bilinear = dataset_obj.class_tokenizer.bilinear if dataset_obj.class_tokenizer is not None else False
    local_collate = lambda x: collate_fn(x, bilinear)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, sampler=sampler,
                            collate_fn=local_collate, num_workers = num_workers)
    dataloader.name = dataloader_name
    return dataloader

def prepare_data(dataset, tokenizer, batch_size, given_distr = False, 
                 sampler_fn = SequentialSampler, num_workers = 0,
                 dataloader_name = "Dataset"):
    dataset_obj, sampler = create_dataset(dataset, tokenizer, sampler_fn, given_distr)
    dataloader = create_dataloader(dataset_obj, batch_size, sampler, num_workers, dataloader_name)
    return dataloader
