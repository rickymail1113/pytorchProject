import torch
import cv2
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)  # or yolov5n - yolov5x6, custom
cap = cv2.VideoCapture(0)
while cap.isOpened():
    if 27 == 0xFF & cv2.waitKey(1):
        break

    success, image = cap.read()
    if not success:
        print("camera frame error")
        continue

    image = cv2.resize(image, (800,600))
    result = model(image)
    result.print()
    print(result.xyxy)
    cv2.imshow("pyTorch", np.squeeze(result.render()))

cap.release()
cv2.destroyAllWindows()