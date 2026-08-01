import torch

if not torch.cuda.is_available():
    print("CUDA not available")
    exit()

device = torch.device("cuda")
x = torch.randn(1000, 1000).to(device)
y = x @ x.T
print("GPU computation OK")
print("Device:", torch.cuda.get_device_name(0))
