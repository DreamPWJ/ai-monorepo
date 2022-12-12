### PyTorch机器学习框架Demo项目

### 安装步骤

- PyTorch官网初始化 https://pytorch.org
- 安装包管理工具Anaconda 需要添加环境变量 https://www.anaconda.com/products/distribution#windows
- (集成显卡无需设置 使用CPU训练)安装CUDA显卡NVIDIA运算平台
  查看cuda版本命令：nvidia-smi https://developer.nvidia.com/cuda-downloads
- 管理员身份运行: conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
- conda --version
- conda list | findstr torch
- 注意: PyCharm内设置Anaconda目录下python.exe作为拦截器才能找到torch模块