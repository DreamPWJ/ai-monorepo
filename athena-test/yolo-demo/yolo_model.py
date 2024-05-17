from ultralytics import YOLO

"""
  @author 潘维吉
  @date 2022/12/05 13:22
  @email 406798106@qq.com
  @description PyTorch机器学习YOLO目标检测
  參考文章: https://docs.ultralytics.com/usage/python/
"""

# Load a model  执行python运行 自动下载预训练模型和配置
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model 训练模型开启
#model.train(data="coco8.yaml", epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set

# 运行预测
results = model(["images/bus.jpg"])  # predict on an image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk

#path = model.export(format="onnx")  # export the model to ONNX format
