import os
import torch
import numpy as np

try:
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    xla_device_available = True
except ModuleNotFoundError:
    xla_device_available = False

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

LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR = "./logs", "./models", "./checkpoints", "./trace", "./tmp"
folders_to_create = [LOG_DIR, OUTPUT_DIR, CKPT_DIR, TRC_DIR, TMP_DIR]
for select_DIR in folders_to_create:
    if not os.path.exists(select_DIR):
        os.mkdir(select_DIR)

aa_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
x_tokens = np.reshape(np.array(list(aa_alphabet)), (1, -1))
