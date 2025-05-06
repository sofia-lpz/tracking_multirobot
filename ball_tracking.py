from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-k", "--known-diameter", type=float, default=6.0,
    help="known diameter of the ball in centimeters")
ap.add_argument("-f", "--focal-length", type=float, default=None,
    help="focal length of camera (if known, otherwise calculated)")
args = vars(ap.parse_args())

blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)

KNOWN_DIAMETER = args["known_diameter"]
FOCAL_LENGTH = args["focal_length"]

def calculate_distance(pixel_diameter, known_diameter, focal_length):
    return (known_diameter * focal_length) / pixel_diameter

def calculate_xyz_coordinates(x, y, frame_width, frame_height, distance):

    center_x = frame_width / 2
    center_y = frame_height / 2
    
    pixel_x = x - center_x
    pixel_y = center_y - y
    
    cm_per_pixel = KNOWN_DIAMETER / (2 * radius)

    x_cm = pixel_x * cm_per_pixel
    y_cm = pixel_y * cm_per_pixel
    
    return x_cm, y_cm, distance

try:
    with open("camera_calibration.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Focal Length:" in line:
                FOCAL_LENGTH = float(line.split(":")[1].strip().split()[0])
                print(f"Loaded focal length from file: {FOCAL_LENGTH}")
            if "Known Diameter:" in line:
                saved_diameter = float(line.split(":")[1].strip().split()[0])
                if saved_diameter != KNOWN_DIAMETER:
                    print(f"Note: Saved diameter ({saved_diameter}cm) differs from parameter ({KNOWN_DIAMETER}cm)")
except FileNotFoundError:
    print("No previous calibration file found. Will create one after calibration.")

if FOCAL_LENGTH is None:
    print("Focal length not provided. Default will be used until calibration.")
    # 50cm distance for a ball with diameter ~1/6 of screen width
    FOCAL_LENGTH = (600 / 6) * 50 / KNOWN_DIAMETER
    print(f"Using default focal length: {FOCAL_LENGTH:.2f}")

pts = deque(maxlen=args["buffer"])

in_calibration_mode = False
calibration_distances = [30, 60, 90]
current_calibration_idx = 0
calibration_samples = []
calibration_countdown = 0

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(1.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break
    
    frame = imutils.resize(frame, width=600)
    frame_height, frame_width = frame.shape[:2]
    output = frame.copy()
    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
   
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    center = None
    ball_found = False
    distance = 0
    radius = 0
    xyz = (0, 0, 0)
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            if radius > 10:
                ball_found = True

                pixel_diameter = 2 * radius

                distance = calculate_distance(pixel_diameter, KNOWN_DIAMETER, FOCAL_LENGTH)

                xyz = calculate_xyz_coordinates(x, y, frame_width, frame_height, distance)

                cv2.circle(output, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.circle(output, center, 5, (0, 0, 255), -1)
    
    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

        cv2.line(output, pts[i - 1], pts[i], (0, 0, 255), thickness)

    if in_calibration_mode:
        overlay = output.copy()
        cv2.rectangle(overlay, (50, 50), (550, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        if current_calibration_idx < len(calibration_distances):
            current_distance = calibration_distances[current_calibration_idx]
            
            instruction = f"Place ball at {current_distance}cm and press SPACE"
            cv2.putText(output, instruction, (70, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if calibration_countdown > 0:
                cv2.putText(output, f"Capturing in: {calibration_countdown//10}", (70, 130), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                calibration_countdown -= 1
                
                if calibration_countdown == 0 and ball_found:
                    # Record sample with the current distance and pixel diameter
                    pixel_diameter = 2 * radius
                    calibration_samples.append((current_distance, pixel_diameter))
                    print(f"Captured sample at {current_distance}cm: {pixel_diameter:.2f} pixels diameter")
                    current_calibration_idx += 1
                    
                    if current_calibration_idx >= len(calibration_distances):
                        # Calculate focal length using all samples
                        focal_lengths = []
                        for distance, pixel_d in calibration_samples:
                            fl = (pixel_d * distance) / KNOWN_DIAMETER
                            focal_lengths.append(fl)
                        
                        # Take average focal length
                        FOCAL_LENGTH = sum(focal_lengths) / len(focal_lengths)
                        print(f"Multi-point calibration complete. Focal length set to {FOCAL_LENGTH:.2f} pixels")
                        
                        with open("camera_calibration.txt", "w") as f:
                            f.write(f"Focal Length: {FOCAL_LENGTH:.2f} pixels\n")
                            f.write(f"Known Diameter: {KNOWN_DIAMETER:.2f} cm\n")
                            f.write(f"Calibration Points: {calibration_samples}\n")
                        
                        print("Calibration data saved to camera_calibration.txt")
                        in_calibration_mode = False
                elif calibration_countdown == 0 and not ball_found:
                    print("No ball detected during calibration capture. Try again.")
                    calibration_countdown = 0
        else:
            in_calibration_mode = False
    else:
        cv2.putText(output, "Press 'c' for calibration", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    legend_x = frame_width - 150
    legend_y = 30
    legend_spacing = 20

    if ball_found:
        x_cm, y_cm, z_cm = xyz
        cv2.putText(output, f"X: {x_cm:.2f} cm", (legend_x, legend_y + legend_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, f"Y: {y_cm:.2f} cm", (legend_x, legend_y + 2*legend_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, f"Z: {z_cm:.2f} cm", (legend_x, legend_y + 3*legend_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        cv2.putText(output, "No ball detected", (legend_x, legend_y + legend_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Ball XYZ Coordinate Tracking", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c") and not in_calibration_mode:
        in_calibration_mode = True
        current_calibration_idx = 0
        calibration_samples = []
        print("Entering calibration mode...")
        print(f"You will be asked to place the ball at {len(calibration_distances)} different distances")
        print(f"Distances: {calibration_distances} cm")
    elif key == ord(" ") and in_calibration_mode and current_calibration_idx < len(calibration_distances) and ball_found and calibration_countdown == 0:
        calibration_countdown = 30

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
    
cv2.destroyAllWindows()