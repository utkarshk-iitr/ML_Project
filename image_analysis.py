import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

img = cv2.imread('input/image1.jpg')
results = model.predict(source=img, imgsz=640, conf=0.25)
r = results[0]

percentage_damage = 0.0
if r.masks is not None:
    masks = r.masks.data.cpu().numpy()
    image_area = img.shape[0] * img.shape[1]
    total_area = 0
    for mask in masks:
        binary_mask = (mask > 0).astype(np.uint8) * 255
        contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area += cv2.contourArea(contour[0])
    percentage_damage = (total_area / image_area) * 100

annotated = r.plot(boxes=False)
cv2.putText(annotated, f'Road Damage: {percentage_damage:.2f}%',(40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
cv2.imwrite('output/out_img1.jpg', annotated)
print(f'Damage: {percentage_damage:.2f}%')