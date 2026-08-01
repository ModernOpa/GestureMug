import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    t = torch.randn(1000, 1000).cuda()
    print("Tensor on CUDA:", t.is_cuda)
