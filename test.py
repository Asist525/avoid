import torch
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))