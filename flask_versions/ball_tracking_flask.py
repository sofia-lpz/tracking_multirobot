from flask import Flask, jsonify, Response
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import threading

app = Flask(__name__)

blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)
BUFFER_SIZE = 64
KNOWN_DIAMETER = 6.0
FOCAL_LENGTH = None

ball_position = {"x": 0, "y": 0, "z": 0, "detected": False}
pts = deque(maxlen=BUFFER_SIZE)
video_stream = None
processing_thread = None
thread_running = False

def calculate_distance(pixel_diameter, known_diameter, focal_length):
    """Calculate distance (Z coordinate) based on apparent size"""
    return (known_diameter * focal_length) / pixel_diameter

def load_calibration():
    """Load camera calibration from file if available"""
    global FOCAL_LENGTH
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
        print("No previous calibration file found.")

    if FOCAL_LENGTH is None:
        print("Focal length not provided. Using default.")
        # 50cm distance for a ball with diameter ~1/6 of screen width
        FOCAL_LENGTH = (600 / 6) * 50 / KNOWN_DIAMETER
        print(f"Using default focal length: {FOCAL_LENGTH:.2f}")

def process_frame(frame):
    """Process a single video frame to detect the ball and get XYZ coordinates"""
    global ball_position

    output = frame.copy()

    frame = imutils.resize(frame, width=600)
    output = imutils.resize(output, width=600)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ball_position["detected"] = False
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        M = cv2.moments(c)
        if M["m00"] > 0 and radius > 10:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            pixel_diameter = 2 * radius
            distance = calculate_distance(pixel_diameter, KNOWN_DIAMETER, FOCAL_LENGTH)

            ball_position = {
                "x": center[0],          # X coordinate in pixels from left
                "y": center[1],          # Y coordinate in pixels from top
                "z": round(distance, 2), # Z coordinate in cm from camera
                "detected": True
            }

            pts.appendleft(center)

            cv2.circle(output, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(output, center, 5, (0, 0, 255), -1)
            
            rounded_distance = round(distance, 2)
            cv2.putText(output, f"Dist: {rounded_distance} cm", (int(x) - 60, int(y) - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            coord_text = f"X: {center[0]}, Y: {center[1]}, Z: {rounded_distance}cm"
            cv2.putText(output, coord_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
        cv2.line(output, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    return output

current_frame = None
frame_lock = threading.Lock()

def video_processing_loop():
    """Background thread for continuous video processing"""
    global thread_running, current_frame
    
    while thread_running:
        frame = video_stream.read()
        
        if frame is not None:
            processed_frame = process_frame(frame)
            
            with frame_lock:
                current_frame = processed_frame
        
        time.sleep(0.01)

@app.route('/coordinates', methods=['GET'])
def get_coordinates():
    """API endpoint to get the current XYZ coordinates"""
    return jsonify(ball_position)

@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint to check if the tracking system is running"""
    return jsonify({
        "running": thread_running,
        "calibration": {
            "focal_length": FOCAL_LENGTH,
            "known_diameter": KNOWN_DIAMETER
        }
    })

def generate_frames():
    """Generate frames for the video stream"""
    while True:
        with frame_lock:
            if current_frame is not None:
                _, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Route for streaming video feed"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def start_video_processing():
    """Initialize and start the video processing thread"""
    global video_stream, processing_thread, thread_running
    
    load_calibration()
    
    video_stream = VideoStream(src=0).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    thread_running = True
    processing_thread = threading.Thread(target=video_processing_loop)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("Video processing started")

def stop_video_processing():
    """Stop the video processing thread and release resources"""
    global thread_running, video_stream
    
    thread_running = False
    if processing_thread is not None:
        processing_thread.join(timeout=1.0)
    
    if video_stream is not None:
        video_stream.stop()
    
    print("Video processing stopped")

@app.route('/')
def index():
    """Simple HTML page with links to the available endpoints"""
    html = """
    <html>
      <head>
        <title>Ball Tracking Flask API</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          h1 { color: #333; }
          .container { display: flex; }
          .video { margin-right: 20px; }
          .endpoints { margin-left: 20px; }
          pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
        </style>
      </head>
      <body>
        <h1>Ball Tracking Flask API</h1>
        <div class="container">
          <div class="video">
            <h2>Live Video Feed</h2>
            <img src="/video_feed" width="600" height="450">
          </div>
          <div class="endpoints">
            <h2>API Endpoints</h2>
            <p><b>Get XYZ Coordinates:</b></p>
            <pre>GET /coordinates</pre>
            <p><b>Check System Status:</b></p>
            <pre>GET /status</pre>
            <p><b>JSON Example:</b></p>
            <pre>{
  "x": 320,
  "y": 240,
  "z": 50.25,
  "detected": true
}</pre>
          </div>
        </div>
      </body>
    </html>
    """
    return html

if __name__ == '__main__':
    try:
        start_video_processing()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        stop_video_processing()