import os
import torch

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence

aa_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
x_tokens = np.reshape(np.array(list(aa_alphabet)), (1, -1))

def tokenize_aminoacid_sequence(seq, others, max_len):
    other_feats = others.shape[-1] if len(others) > 0 else 0
    x_feats = torch.zeros((max_len, 23+other_feats))
    masks = torch.zeros((max_len, 1))
    
    seq_len = len(seq)
    x_feats[0, 21] = 1
    x_feats[seq_len+1, 22] = 1

    if type(seq) != list:
        seq = list(seq)
    x_feats[1:seq_len+1, :21] = torch.from_numpy(np.reshape(np.array(seq), (-1, 1)) == x_tokens).to(float)
    if other_feats > 0:
        x_feats[1:seq_len+1, 23:] = torch.from_numpy(others).to(float)
    masks[1:seq_len+1] = 1
    return x_feats, masks

def unsupervised_tokenizer(seqs, device, split_token = ""):
    real_seqs = [seq.split(split_token) for seq in seqs]
    seq_size = max(list(map(len, real_seqs)))
    max_len = seq_size+2
    
    tensor_tuple = [tokenize_aminoacid_sequence(seq, [], max_len) for seq in real_seqs]
    x_feats, masks = zip(*tensor_tuple)
    x_feats = pad_sequence(x_feats, batch_first=True, padding_value=0).to(device)
    masks = pad_sequence(masks, batch_first=True, padding_value=0).to(device)
    return x_feats, masks

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
    
def init_folders(folders_to_create):
    for select_DIR in folders_to_create:
        if not os.path.exists(select_DIR):
            os.mkdir(select_DIR)
