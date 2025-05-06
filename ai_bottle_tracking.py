from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from ultralytics import YOLO
import math

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-k", "--known-height", type=float, default=24.0,
    help="known height of the bottle in centimeters")
ap.add_argument("-f", "--focal-length", type=float, default=None,
    help="focal length of camera (if known, otherwise calculated)")
args = vars(ap.parse_args())

MODEL_PATH = r"d:\projects\windows\multirobots\ball_tracking\models\yolov8n-seg.pt"

BOTTLE_CLASS_ID = 39

KNOWN_HEIGHT = args["known_height"]
FOCAL_LENGTH = args["focal_length"]

def calculate_distance(pixel_height, known_height, focal_length):
    return (known_height * focal_length) / pixel_height

try:
    with open("camera_calibration.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Focal Length:" in line:
                FOCAL_LENGTH = float(line.split(":")[1].strip().split()[0])
                print(f"Loaded focal length from file: {FOCAL_LENGTH}")
            if "Known Height:" in line:
                saved_height = float(line.split(":")[1].strip().split()[0])
                if saved_height != KNOWN_HEIGHT:
                    print(f"Note: Saved height ({saved_height}cm) differs from parameter ({KNOWN_HEIGHT}cm)")
except FileNotFoundError:
    print("No previous calibration file found. Will create one after calibration.")

model = YOLO(MODEL_PATH)

pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(1.0)


in_calibration_mode = False
calibration_distances = [30, 60, 90]
current_calibration_idx = 0
calibration_samples = []
calibration_countdown = 0

if FOCAL_LENGTH is None:
    print("Focal length not provided. Default will be used until calibration.")
    # Default assumption: 50cm distance for a bottle with height ~1/3 of screen height
    FOCAL_LENGTH = (600 / 3) * 50 / KNOWN_HEIGHT

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        break
        
    frame = imutils.resize(frame, width=600)
    
    output = frame.copy()
    
    results = model(frame, conf=0.25)  # Lower confidence threshold for better detection
    
    center = None
    bottle_found = False
    pixel_height = 0
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    if len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            
            for box_idx, box in enumerate(result.boxes):
                class_id = int(box.cls[0].item())
                
                if class_id == BOTTLE_CLASS_ID:
                    bottle_found = True
                    conf = float(box.conf[0].item())
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    pixel_height = y2 - y1
                    
                    distance = calculate_distance(pixel_height, KNOWN_HEIGHT, FOCAL_LENGTH)
                    
                    if hasattr(result, 'masks') and result.masks is not None and box_idx < len(result.masks):
                        mask = result.masks[box_idx].data[0].cpu().numpy()
                        
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # Convert to binary mask
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        
                        colored_mask = np.zeros_like(frame)
                        colored_mask[binary_mask > 0] = [0, 0, 255]
                        
                        alpha = 0.5
                        cv2.addWeighted(colored_mask, alpha, output, 1 - alpha, 0, output)

                        M = cv2.moments(binary_mask)
                        if M["m00"] > 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            center = (center_x, center_y)
                            
                            cv2.circle(output, center, 5, (0, 255, 0), -1)
                            
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)

                                rect = cv2.minAreaRect(largest_contour)
                                box_points = cv2.boxPoints(rect)
                                box_points = np.intp(box_points)
                                
                                # Draw rotated rectangle
                                cv2.drawContours(output, [box_points], 0, (0, 255, 255), 2)
                                
                                rect_width, rect_height = rect[1]
                                # Make sure height is the longer dimension for bottles
                                actual_height = max(rect_width, rect_height)
                                
                                # Refine distance estimate using actual height
                                refined_distance = calculate_distance(actual_height, KNOWN_HEIGHT, FOCAL_LENGTH)
                                
                                # Average with bounding box method for more stability
                                distance = (distance + refined_distance) / 2
                        else:
                            # Fallback to bounding box center
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            center = (center_x, center_y)
                            cv2.circle(output, center, 5, (0, 255, 0), -1)
                    else:
                        # If no mask, just use bounding box
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        center = (center_x, center_y)
                        cv2.circle(output, center, 5, (0, 255, 0), -1)
                    
                    rounded_distance = round(distance, 2)
                    
                    cv2.putText(output, f"Bottle: {conf:.2f}", (x1, y1 - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(output, f"Dist: {rounded_distance} cm", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if center is not None:
        pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
            
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        
        cv2.line(output, pts[i - 1], pts[i], (0, 255, 0), thickness)

    if in_calibration_mode:
        overlay = output.copy()
        cv2.rectangle(overlay, (50, 50), (550, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        if current_calibration_idx < len(calibration_distances):
            current_distance = calibration_distances[current_calibration_idx]
            
            instruction = f"Place bottle at {current_distance}cm and press SPACE"
            cv2.putText(output, instruction, (70, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if calibration_countdown > 0:
                cv2.putText(output, f"Capturing in: {calibration_countdown//10}", (70, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                calibration_countdown -= 1
                
                if calibration_countdown == 0 and bottle_found:
                    calibration_samples.append((current_distance, pixel_height))
                    print(f"Captured sample at {current_distance}cm: {pixel_height} pixels height")
                    current_calibration_idx += 1
                    

                    if current_calibration_idx >= len(calibration_distances):
                        focal_lengths = []
                        for distance, pixel_h in calibration_samples:
                            fl = (pixel_h * distance) / KNOWN_HEIGHT
                            focal_lengths.append(fl)
                        
                        FOCAL_LENGTH = sum(focal_lengths) / len(focal_lengths)
                        print(f"Multi-point calibration complete. Focal length set to {FOCAL_LENGTH:.2f} pixels")
                        
                        with open("camera_calibration.txt", "w") as f:
                            f.write(f"Focal Length: {FOCAL_LENGTH:.2f} pixels\n")
                            f.write(f"Known Height: {KNOWN_HEIGHT:.2f} cm\n")
                            f.write(f"Calibration Points: {calibration_samples}\n")
                        
                        print("Calibration data saved to camera_calibration.txt")
                        in_calibration_mode = False
                elif calibration_countdown == 0 and not bottle_found:
                    print("No bottle detected during calibration capture. Try again.")
                    calibration_countdown = 0
        else:
            in_calibration_mode = False
    else:
        cv2.putText(output, "C", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Becario multirobot", output)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c") and not in_calibration_mode:
        in_calibration_mode = True
        current_calibration_idx = 0
        calibration_samples = []
        print("Entering calibration mode...")
        print(f"You will be asked to place the bottle at {len(calibration_distances)} different distances")
        print(f"Distances: {calibration_distances} cm")

    elif key == ord(" ") and in_calibration_mode and current_calibration_idx < len(calibration_distances) and bottle_found and calibration_countdown == 0:
        calibration_countdown = 30

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
    
cv2.destroyAllWindows()