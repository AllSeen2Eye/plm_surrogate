import os
import numpy as np

LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR = "./logs", "./models", "./checkpoints", "./trace", "./tmp"
folders_to_create = [LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR]
for select_DIR in folders_to_create:
    if not os.path.exists(select_DIR):
        os.mkdir(select_DIR)

aa_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
x_tokens = np.reshape(np.array(list(aa_alphabet)), (1, -1))
