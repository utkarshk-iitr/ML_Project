import cv2
import numpy as np
from ultralytics import YOLO

def generate(num):
    model = YOLO('best.pt')
    img = cv2.imread(f'input/image{num}.jpeg')
    results = model.predict(source=img, imgsz=640, conf=0.25)
    r = results[0]

    annotated = img.copy()
    percentage_damage = 0.0

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        height, width = img.shape[:2]
        image_area = height * width
        total_area = 0
        
        for mask in masks:
            mask_resized = cv2.resize((mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            binary_mask = mask_resized * 255
            contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contour:
                continue
            
            cnt = max(contour, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            total_area += area
            
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            
            distance_weight = 1.0 + (cY / height)
            trajectory_weight = 1.5 if (width / 3 < cX < 2 * width / 3) else 1.0
            
            mask_indices = binary_mask > 0
            if np.any(mask_indices):
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray_img[mask_indices])
                depth_weight = 1.5 if mean_intensity < 80 else 1.0
            else:
                depth_weight = 1.0

            severity_score = area * distance_weight * trajectory_weight * depth_weight
            relative_severity = (severity_score / image_area) * 100
            
            color = (0, 255, 0) if relative_severity < 0.1 else (0, 255, 255) if relative_severity < 0.5 else (0, 0, 255)
            
            colored_mask = np.zeros_like(img)
            colored_mask[mask_indices] = color
            cv2.addWeighted(colored_mask, 0.4, annotated, 1.0, 0, annotated)
            
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        
        percentage_damage = (total_area / image_area) * 100

    text = f'Road Damage: {percentage_damage:.2f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.0, 2)[0]

    x_pos, y_pos, padding = 15, 45, 8
    box_x1, box_y1 = x_pos - padding, y_pos - text_size[1] - padding
    box_x2, box_y2 = x_pos + text_size[0] + padding, y_pos + padding

    cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
    cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), 2)
    cv2.putText(annotated, text, (x_pos, y_pos), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(f'output/img_out{num}.jpeg', annotated)

generate(1)
generate(2)
generate(3)