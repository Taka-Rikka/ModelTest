import torch
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
print("CUDA版本：", torch.version.cuda)  # 输出None则为CPU版本
print("GPU数量：", torch.cuda.device_count())
