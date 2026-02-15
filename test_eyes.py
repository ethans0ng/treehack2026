import cv2
import numpy as np
import torch
import gc
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

gc.collect()
torch.cuda.empty_cache()

predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    image_encoder_engine="data/owl_image_encoder_patch32.engine"
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

def get_pupil_location(eye_img_bgr, is_left_eye):
    if eye_img_bgr.size == 0: return None
    h_orig, w_orig, _ = eye_img_bgr.shape
    
    # 1. Your Left/Right Shadow Dodge (Still critical)
    if is_left_eye:
        x_start, x_end = int(w_orig * 0.35), int(w_orig * 0.85) 
    else:
        x_start, x_end = int(w_orig * 0.15), int(w_orig * 0.65) 
        
    y_start, y_end = int(h_orig * 0.30), int(h_orig * 0.70) 
    crop = eye_img_bgr[y_start:y_end, x_start:x_end]
    
    if crop.size == 0: return None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. THE GLINT KILLER: This specifically erases the bright screen reflection
    # inside your dark pupil, making it a solid dark mass again.
    gray = cv2.medianBlur(gray, 7)
    
    # 3. Find the dark blob among the sclera
    # We calculate the average brightness of the eye box, and threshold 
    # anything darker than average to become our pure white tracking blob
    mean_val = np.mean(gray)
    _, thresh = cv2.threshold(gray, mean_val - 15, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Find the center of that blob
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    crop_area = crop.shape[0] * crop.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Ignore tiny camera noise AND ignore giant shadows that fill the box
        if area < 10 or area > (crop_area * 0.5): 
            continue 
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            px = int(M["m10"] / M["m00"])
            py = int(M["m01"] / M["m00"])
            return (px + x_start, py + y_start)
            
    return None

print("Scanning for a frame with BOTH eyes...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        small_frame = cv2.resize(frame, (640, 480))
        image_np = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).copy()
        image_pil_safe = Image.fromarray(image_np)
        
        output_eyes = predictor.predict(
            image=image_pil_safe, 
            text=["human eye"], 
            text_encodings=None, 
            threshold=0.15 
        )

        if len(output_eyes.boxes) >= 2:
            print(f"Found {len(output_eyes.boxes)} eyes. Drawing and saving...")
            
            # Sort boxes by X coordinate (left to right across the screen)
            # The first box [0] is the left side of the screen, the second [1] is the right
            boxes = sorted(output_eyes.boxes.cpu().numpy().astype(int), key=lambda b: b[0])
            
            for i, box in enumerate(boxes[:2]):
                x1, y1, x2, y2 = box
                h, w, _ = small_frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                eye_crop = small_frame[y1:y2, x1:x2]
                
                # Determine left/right based on index
                is_left_eye = (i == 0)
                pupil_rel = get_pupil_location(eye_crop, is_left_eye)
                
                if pupil_rel:
                    px, py = pupil_rel
                    gx, gy = x1 + px, y1 + py
                    cv2.circle(small_frame, (gx, gy), 5, (0, 0, 255), -1)
                    print(f"Pupil marked at X: {gx}, Y: {gy}")
                else:
                    print("Pupil rejected (blob too big or too small).")

            cv2.imwrite("debug.jpg", small_frame)
            print("SUCCESS: 'debug.jpg' saved. Stopping script.")
            break 

        del output_eyes
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("Script manually stopped.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()