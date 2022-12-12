import os

import torch
import torchvision
from PIL import Image
from torch import nn

"""
  @author 潘维吉
  @date 2022/12/05 13:22
  @email 406798106@qq.com
  @description PyTorch机器学习 验证
"""

i = 0  # 识别图片计数
root_path = "data/test/cat"  # 待测试文件夹
names = os.listdir(root_path)
for name in names:
    print(name)
    i = i + 1
    data_class = ['猫']  # 按文件索引顺序排列
    image_path = os.path.join(root_path, name)
    image = Image.open(image_path)
    print(image)
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
                                                 torchvision.transforms.ToTensor()])
    image = transforms(image)
    print(image.shape)

    model_ft = torchvision.models.resnet18()  # 需要使用训练时的相同模型
    # print(model_ft)
    in_features = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(in_features, 36),
                                nn.Linear(36, 6))  # 此处也要与训练模型一致

    model = torch.load("best_model_panweiji.pth", map_location=torch.device("cpu"))  # 选择训练后得到的模型文件
    # print(model)
    image = torch.reshape(image, (1, 3, 64, 64))  # 修改待预测图片尺寸，需要与训练时一致
    model.eval()
    with torch.no_grad():
        output = model(image)
    print(output)  # 输出预测结果
    print(int(output.argmax(1)))
    # print("第{}张图片预测为：{}".format(i, data_class[int(output.argmax(1))]))  # 对结果进行处理，使直接显示出预测的植物种类
