from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")
model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
