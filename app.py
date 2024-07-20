from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import os
import time

# Initialize Flask app
app = Flask(__name__)

# Create a directory to save detected faces if it doesn't exist
save_directory = 'saved_faces'
os.makedirs(save_directory, exist_ok=True)

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

video_capture = None  # Initialize video capture variable

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        return jsonify({'status': 'success', 'message': 'Camera started'}), 200
    else:
        return jsonify({'status': 'already_running', 'message': 'Camera is already running'}), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        cv2.destroyAllWindows()
        video_capture = None
        return jsonify({'status': 'success', 'message': 'Camera stopped'}), 200
    else:
        return jsonify({'status': 'not_running', 'message': 'Camera is not running'}), 200

@app.route('/start_detection', methods=['POST'])
def start_detection():
    if video_capture is None:
        return jsonify({'status': 'failed', 'message': 'Camera is not running'}), 500

    # Flag to check if the image has been captured
    captured = False
    timestamp = int(time.time())

    while not captured:
        ret, frame = video_capture.read()
        if not ret:
            return jsonify({'status': 'failed', 'message': 'Failed to capture image'}), 500

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the detected face image
                face_image = frame[y:y + h, x:x + w]
                file_path = os.path.join(save_directory, f'face_{timestamp}.jpg')
                cv2.imwrite(file_path, face_image)

                # Set flag to indicate that an image has been captured
                captured = True
                break  # Exit the loop after capturing the image

    if captured:
        return jsonify({'status': 'success', 'file_path': file_path}), 200
    else:
        return jsonify({'status': 'failed', 'message': 'No face detected'}), 500

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while video_capture is not None:
            success, frame = video_capture.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# CRUD Routes
@app.route('/images')
def images():
    image_files = [f for f in os.listdir(save_directory) if f.endswith('.jpg')]
    return render_template('images.html', images=image_files)

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(save_directory, filename)

@app.route('/delete_image/<filename>', methods=['POST'])
def delete_image(filename):
    file_path = os.path.join(save_directory, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'status': 'success', 'message': 'Image deleted'}), 200
    else:
        return jsonify({'status': 'failed', 'message': 'Image not found'}), 404
    
@app.route('/manage_images')
def manage_images():
    images = {}
    for filename in os.listdir(save_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Example details; replace this with actual data fetching logic
            details = {
                'name': 'John Doe',
                'enrollment_number': '123456',
                'gender': 'Male',
                'year': '2nd',
                'department': 'Computer Science'
            }
            images[filename] = details
    return render_template('manage_images.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
