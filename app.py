import torch
import cv2
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)  # or yolov5n - yolov5x6, custom

# Images
# image = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread("./data/images/b.jpg")

result = model(image)
result.print()
print(result.xyxy)
cv2.imshow("pyTorch", np.squeeze(result.render()))
cv2.waitKey(0)