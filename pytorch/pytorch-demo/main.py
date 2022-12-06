import torch
import torchvision

"""
  @author 潘维吉
  @date 2022/12/05 13:22
  @email 406798106@qq.com
  @description PyTorch机器学习
"""

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

torch.cuda.is_available()
x = torch.rand(5, 3)

print(x)
