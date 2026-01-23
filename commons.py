import os
import numpy as np

def init_folders(folders_to_create):
    for select_DIR in folders_to_create:
        if not os.path.exists(select_DIR):
            os.mkdir(select_DIR)

aa_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
x_tokens = np.reshape(np.array(list(aa_alphabet)), (1, -1))
