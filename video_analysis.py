import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

best_model = YOLO('best.pt')
video_path = 'input/sample3.mp4'

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_position = (40, 80)
font_color = (255, 255, 255)
background_color = (0, 0, 255)

damage_deque = deque(maxlen=20)

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output3.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = best_model.predict(source=frame, imgsz=640, conf=0.25)
        processed_frame = frame.copy()
        percentage_damage = 0.0 
        pothole_count = 0
        
        if results[0].masks is not None:
            total_area = 0
            masks = results[0].masks.data.cpu().numpy()
            height, width = frame.shape[:2]
            image_area = height * width
            pothole_count = len(masks)
            
            for mask in masks:
                # Resize mask to original frame dimensions to avoid mismatched shapes
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
                
                # Distance weight (bottom of screen = physically closer = higher weight)
                distance_weight = 1.0 + (cY / height)
                
                # Trajectory danger index (center column = driver's path = higher weight)
                trajectory_weight = 1.5 if (width / 3 < cX < 2 * width / 3) else 1.0
                
                # Depth estimation (shadow intensity inside the mask)
                mask_indices = binary_mask > 0
                if np.any(mask_indices):
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_intensity = np.mean(gray_frame[mask_indices])
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
                
                # Draw customized colored mask and bounding box
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_indices] = color
                cv2.addWeighted(colored_mask, 0.4, processed_frame, 1.0, 0, processed_frame)
                
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(processed_frame, label, (x, max(15, y - 5)), font, 0.6, color, 2, cv2.LINE_AA)
            
            percentage_damage = (total_area / image_area) * 100

        damage_deque.append(percentage_damage)
        smoothed_percentage_damage = float(sum(damage_deque)) / len(damage_deque)
            
        # --- Dashboard UI / HUD Implementation ---
        # 1. Create a semi-transparent black overlay at the top
        overlay = processed_frame.copy()
        hud_height = 130
        cv2.rectangle(overlay, (0, 0), (processed_frame.shape[1], hud_height), (0, 0, 0), -1)
        # Apply the overlay with 70% opacity
        cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
        
        # 2. Determine Severity Color and Status
        if smoothed_percentage_damage < 2.0:
            severity_color = (0, 255, 0) # Green (BGR)
            severity_text = "STATUS: SAFE"
        elif smoothed_percentage_damage < 5.0:
            severity_color = (0, 255, 255) # Yellow (BGR)
            severity_text = "STATUS: MODERATE RISK"
        else:
            severity_color = (0, 0, 255) # Red (BGR)
            severity_text = "STATUS: CRITICAL DAMAGE"

        # 3. Draw HUD text
        right_x = int(processed_frame.shape[1] * 0.6) # Right align starting at 60% of screen width
        
        # Main Damage Percentage
        cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', (40, 55), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        # Pothole Count
        cv2.putText(processed_frame, f'Detected Potholes: {pothole_count}', (40, 105), font, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        # Severity Alert
        cv2.putText(processed_frame, severity_text, (right_x, 55), font, 1.0, severity_color, 2, cv2.LINE_AA)
        
        # Stylized separator line at the bottom of the HUD
        cv2.line(processed_frame, (0, hud_height), (processed_frame.shape[1], hud_height), severity_color, 2)
        # -----------------------------------------

        out.write(processed_frame)

    else:
        break

cap.release()
out.release()
