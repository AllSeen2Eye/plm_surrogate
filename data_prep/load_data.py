import requests
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import mdtraj as md

def load_files_from_pdb(pdb_id_list, save_dir):
    for dict_item in tqdm(pdb_id_list):
        pdb_code, _ = dict_item
        pdb_data = requests.get(f"https://files.rcsb.org/download/{pdb_code}.pdb").text
        with open(f"{save_dir}/{pdb_code}.pdb", "w") as pdb_file_pointer:
            pdb_file_pointer.write(pdb_data)

def get_secondary_from_pdb(pdb_id_list, save_dir, use_chain_id = True):
    dataset_dict, real_idx = {}, []
    for dict_item in pdb_id_list:
        pdb_code, chain_id = dict_item
        try:
            traj = md.load(f"{save_dir}/{pdb_code}.pdb")
            slice_chain = np.argwhere([atom.residue.chain.chain_id == chain_id[0] for atom in list(traj.topology.atoms)])[:, 0]
            if use_chain_id:
                traj = traj.atom_slice(slice_chain)
            traj = traj.atom_slice(traj.topology.select("protein"))
            dssp_list = md.compute_dssp(traj, simplified=False)[0].tolist()
            if "NA" in dssp_list:
                dssp_list = list(filter(lambda x: x != "NA", dssp_list))
            dssp_seq = "".join(dssp_list).replace(" ", "C").strip()
            aminoacid_seq = traj.topology.to_fasta()[0]
            dataset_dict[f"{pdb_code}_{chain_id}"] = [aminoacid_seq, dssp_seq]
            real_idx.append(pdb_code+chain_id)
        except (ValueError, IndexError) as e:
            print(e)
    dataset_df = pd.DataFrame(dataset_dict, index = ["seq", "label"]).T
    dataset_df.index = real_idx
    return dataset_df
