from ultralytics import YOLO

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
print("========================= 目标识别结果 ==============================")
print(results)
print("===================================================================")
