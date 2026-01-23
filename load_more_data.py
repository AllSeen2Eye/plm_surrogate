import requests
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import mdtraj as md

def get_secondary_from_pdb(pdb_id_list, tmp_dir):
    dataset_dict = {}
    for dict_item in tqdm(pdb_id_list):
        pdb_code, chain_id = dict_item
        pdb_data = requests.get(f"https://files.rcsb.org/download/{pdb_code}.pdb").text
        with open(f"{tmp_dir}/tmp_file.pdb", "w") as pdb_file_pointer:
            pdb_file_pointer.write(pdb_data)
        try:
            traj = md.load(f"{tmp_dir}/tmp_file.pdb")
            slice_chain = np.argwhere([atom.residue.chain.chain_id == chain_id for atom in list(traj.topology.atoms)])[:, 0]
            traj = traj.atom_slice(slice_chain)
            traj = traj.atom_slice(traj.topology.select("protein"))
            dssp_list = md.compute_dssp(traj, simplified=False)[0].tolist()
            if "NA" in dssp_list:
                dssp_list = list(filter(lambda x: x != "NA", dssp_list))
            dssp_seq = "".join(dssp_list).replace(" ", "C")
            aminoacid_seq = traj.topology.to_fasta()[0]
            dataset_dict[f"{pdb_code}_{chain_id}"] = [aminoacid_seq, dssp_seq]
        except (ValueError, IndexError) as e:
            print(e)
    dataset_df = pd.DataFrame(dataset_dict, index = ["seq", "label"]).T
    return dataset_df
