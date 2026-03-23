import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

img = cv2.imread('input/image3.jpeg')
results = model.predict(source=img, imgsz=640, conf=0.25)
r = results[0]

annotated = img.copy()
percentage_damage = 0.0
pothole_count = 0

if r.masks is not None:
    masks = r.masks.data.cpu().numpy()
    height, width = img.shape[:2]
    image_area = height * width
    pothole_count = len(masks)
    total_area = 0
    for mask in masks:
        # Resize mask to exactly match the image dimensions
        mask_resized = cv2.resize((mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        binary_mask = mask_resized * 255
        
        contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contour:
            continue
        cnt = max(contour, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        total_area += area
        
        # --- Individual Severity Logic ---
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Distance weight (bottom of screen = higher weight)
        distance_weight = 1.0 + (cY / height)
        # Trajectory danger index (center column)
        trajectory_weight = 1.5 if (width / 3 < cX < 2 * width / 3) else 1.0
        
        # Depth estimation (shadow intensity)
        mask_indices = binary_mask > 0
        if np.any(mask_indices):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_img[mask_indices])
            depth_weight = 1.5 if mean_intensity < 80 else 1.0
        else:
            depth_weight = 1.0

        severity_score = area * distance_weight * trajectory_weight * depth_weight
        relative_severity = (severity_score / image_area) * 100
        
        if relative_severity < 0.1:
            color = (0, 255, 0) # Green (Low)
            label = "LOW"
        elif relative_severity < 0.5:
            color = (0, 255, 255) # Yellow (Medium)
            label = "MEDIUM"
        else:
            color = (0, 0, 255) # Red (Critical)
            label = "CRITICAL"
        
        # Draw customized mask
        colored_mask = np.zeros_like(img)
        colored_mask[mask_indices] = color
        cv2.addWeighted(colored_mask, 0.4, annotated, 1.0, 0, annotated)
        
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
    percentage_damage = (total_area / image_area) * 100

# --- Dashboard UI / HUD Implementation ---
overlay = annotated.copy()
hud_height = int(annotated.shape[0] * 0.15) # 15% of image height
hud_height = max(130, hud_height) # At least 130 pixels
cv2.rectangle(overlay, (0, 0), (annotated.shape[1], hud_height), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

if percentage_damage < 2.0:
    severity_color = (0, 255, 0)
    severity_text = "STATUS: SAFE"
elif percentage_damage < 5.0:
    severity_color = (0, 255, 255)
    severity_text = "STATUS: MODERATE RISK"
else:
    severity_color = (0, 0, 255)
    severity_text = "STATUS: CRITICAL DAMAGE"

font = cv2.FONT_HERSHEY_SIMPLEX
right_x = int(annotated.shape[1] * 0.6)

cv2.putText(annotated, f'Road Damage: {percentage_damage:.2f}%', (40, 55), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(annotated, f'Detected Potholes: {pothole_count}', (40, 105), font, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
cv2.putText(annotated, severity_text, (right_x, 55), font, 1.0, severity_color, 2, cv2.LINE_AA)
cv2.line(annotated, (0, hud_height), (annotated.shape[1], hud_height), severity_color, 2)
# -----------------------------------------

cv2.imwrite('output/img_out3.jpeg', annotated)
print(f'Damage: {percentage_damage:.2f}%')