import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
import threading
from collections import deque, defaultdict
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__, template_folder='templates')

# Verify and create template directory if needed
os.makedirs('templates', exist_ok=True)

# Email Configuration (REPLACE WITH YOUR ACTUAL CREDENTIALS)
SMTP_SERVER = 'smtp.gmail.com'  # Example for Gmail
SMTP_PORT = 587
EMAIL_ADDRESS = 'example@gmail.com'
EMAIL_PASSWORD = '---'  # Use app password for Gmail
RECIPIENT_EMAIL = 'recipient@gmail.com'

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Shared data with messaging
detection_data = {
    'frame': None,
    'detections': deque(maxlen=50),
    'recent_detections': deque(maxlen=10),  # Store last 10 detections for confirmation
    'confirmed_object': None,
    'stats': {
        'fps': 0,
        'total_detections': 0,
        'last_detection': None,
        'cpu_temp': 0,
        'memory_usage': 0,
        'last_detection_time': 0,
        'current_object_count': 0,
        'current_object': None
    },
    'settings': {
        'email_alerts': True,
        'min_confidence': 0.5,
        'detection_enabled': True,
        'detection_interval': 10,  # seconds between detections
        'confirm_count': 10  # number of consistent detections required
    },
    'messages': {
        'grain0': 'The feeder is empty, needs to be refilled ASAP!',
        'grain25': 'The feed is almost empty, refill is required promptly.',
        'grain50': 'The feed is at 50%, no refill needed but preparation is recommended',
        'grain75': 'The feed is still sufficient, refill nor preparation is yet required',
        'grain100': "The feeder has just been refilled, no need for refill or preparation"
    }
}

def get_system_stats():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
        
        with open("/proc/meminfo", "r") as f:
            meminfo = f.readlines()
            total = int(meminfo[0].split()[1])
            free = int(meminfo[1].split()[1])
            percent = (total - free) / total * 100
            
        return temp, percent
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return 0, 0

def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    return picam2

picam2 = init_camera()
model = YOLO("my_model.pt")

def generate_frames():
    while True:
        if detection_data['frame'] is not None:
            ret, buffer = cv2.imencode('.jpg', detection_data['frame'])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

def check_confirmed_object(current_obj):
    # Reset count if object changed
    if detection_data['stats']['current_object'] != current_obj:
        detection_data['stats']['current_object'] = current_obj
        detection_data['stats']['current_object_count'] = 0
        return False
    
    # Increment count for current object
    detection_data['stats']['current_object_count'] += 1
    
    # Check if reached confirmation threshold
    if detection_data['stats']['current_object_count'] >= detection_data['settings']['confirm_count']:
        return True
    return False

def detection_loop():
    last_confirmed_object = None  # Track the last confirmed object
    
    while True:
        try:
            start_time = time.time()
            temp, mem = get_system_stats()
            detection_data['stats']['cpu_temp'] = temp
            detection_data['stats']['memory_usage'] = mem
            
            if not detection_data['settings']['detection_enabled']:
                time.sleep(0.5)
                continue
                
            # Check if enough time has passed since last detection
            current_time = time.time()
            if current_time - detection_data['stats']['last_detection_time'] < detection_data['settings']['detection_interval']:
                time.sleep(0.1)
                continue
                
            frame = picam2.capture_array()
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot() if len(results[0].boxes) > 0 else frame.copy()
            
            current_detections = []
            highest_conf_detection = None
            highest_conf = 0
            
            # Find detection with highest confidence above threshold
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= detection_data['settings']['min_confidence'] and conf > highest_conf:
                    cls_id = int(box.cls)
                    class_name = results[0].names[cls_id]
                    message = detection_data['messages'].get(class_name, "No status message available")
                    
                    highest_conf_detection = {
                        'class': class_name,
                        'confidence': conf,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': message
                    }
                    highest_conf = conf
            
            # Process the highest confidence detection
            if highest_conf_detection:
                detection_data['recent_detections'].append(highest_conf_detection)
                detection_data['stats']['last_detection_time'] = current_time
                detection_data['stats']['total_detections'] += 1
                
                # Check for confirmed object
                if check_confirmed_object(highest_conf_detection['class']):
                    # Only update confirmed object if it's different
                    if detection_data['confirmed_object'] != highest_conf_detection['class']:
                        detection_data['confirmed_object'] = highest_conf_detection['class']
                        
                        # Send email ONLY when confirmed object changes
                        if (detection_data['settings']['email_alerts'] and 
                            last_confirmed_object != highest_conf_detection['class']):
                            
                            send_email(
                                subject=f"CONFIRMED Feed Level: {highest_conf_detection['class']}",
                                body=f"Confirmed Detection:\n\n"
                                     f"Object: {highest_conf_detection['class']}\n"
                                     f"Confidence: {highest_conf_detection['confidence']:.2%}\n"
                                     f"Time: {highest_conf_detection['time']}\n\n"
                                     f"Status: {highest_conf_detection['message']}\n\n"
                                     f"After {detection_data['settings']['confirm_count']} consistent detections."
                            )
                            last_confirmed_object = highest_conf_detection['class']
                
                current_detections.append(highest_conf_detection)
                detection_data['stats']['last_detection'] = highest_conf_detection
            
            detection_data['frame'] = annotated_frame
            detection_data['detections'].extend(current_detections)
            detection_data['stats']['fps'] = 1 / max(0.001, time.time() - start_time)
            
            elapsed = time.time() - start_time
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
                
        except Exception as e:
            print(f"Detection error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    return jsonify({
        'detections': list(detection_data['detections']),
        'stats': detection_data['stats'],
        'settings': detection_data['settings'],
        'confirmed_object': detection_data['confirmed_object'],
        'messages': detection_data['messages']
    })

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    for key in ['email_alerts', 'min_confidence', 'detection_enabled', 'detection_interval', 'confirm_count']:
        if key in data:
            # Special handling for min_confidence to ensure it's a float
            if key == 'min_confidence':
                try:
                    detection_data['settings'][key] = float(data[key])
                except (ValueError, TypeError):
                    continue
            else:
                detection_data['settings'][key] = data[key]
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Start detection thread with lower priority
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    
    try:
        import os
        os.nice(10)
    except:
        pass
    
    detection_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)