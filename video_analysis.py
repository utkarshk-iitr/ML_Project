import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

def process_video(num):
    model = YOLO('best.pt')
    cap = cv2.VideoCapture(f'input/sample{num}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(f'output/output{num}.webm', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    damage_deque = deque(maxlen=20)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model.predict(source=frame, imgsz=640, conf=0.25)
        processed_frame = frame.copy()
        percentage_damage = 0.0
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            height, width = frame.shape[:2]
            image_area = height * width
            total_area = 0
            
            for mask in masks:
                mask_resized = cv2.resize((mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                binary_mask = mask_resized * 255
                contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contour:
                    continue
                
                cnt = max(contour, key=cv2.contourArea)
                total_area += cv2.contourArea(cnt)
                
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                
                distance_weight = 1.0 + (cY / height)
                trajectory_weight = 1.5 if (width / 3 < cX < 2 * width / 3) else 1.0
                
                mask_indices = binary_mask > 0
                if np.any(mask_indices):
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_intensity = np.mean(gray_frame[mask_indices])
                    depth_weight = 1.5 if mean_intensity < 80 else 1.0
                else:
                    depth_weight = 1.0

                severity_score = cv2.contourArea(cnt) * distance_weight * trajectory_weight * depth_weight
                relative_severity = (severity_score / image_area) * 100
                
                color = (0, 255, 0) if relative_severity < 0.1 else (0, 255, 255) if relative_severity < 0.5 else (0, 0, 255)
                
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_indices] = color
                cv2.addWeighted(colored_mask, 0.4, processed_frame, 1.0, 0, processed_frame)
                
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
            
            percentage_damage = (total_area / image_area) * 100
        
        damage_deque.append(percentage_damage)
        smoothed_damage = float(sum(damage_deque)) / len(damage_deque)
        
        
        text1 = f'Road Damage: {smoothed_damage:.2f}%'
        text_size1 = cv2.getTextSize(text1, font, 1.0, 2)[0]
        
        padding = 8
        x_pos1, y_pos1 = 15, 45
        box_x1 = x_pos1 - padding
        box_y1 = y_pos1 - text_size1[1] - padding
        box_x2 = x_pos1 + text_size1[0] + padding
        box_y2 = y_pos1 + padding
        
        cv2.rectangle(processed_frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        cv2.rectangle(processed_frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), 2)
        cv2.putText(processed_frame, text1, (x_pos1, y_pos1), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        out.write(processed_frame)
    
    cap.release()
    out.release()

process_video(1)
process_video(2)
