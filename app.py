import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, render_template, flash, redirect, url_for, Response, jsonify
import warnings
import uuid
import math
import base64
from io import BytesIO
import time
import logging

print("Starting Flask app...")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Hook:
    def __init__(self):
        self.forward_out = None
        self.backward_out = None

    def register_hook(self, module):
        self.hook_f = module.register_forward_hook(self.forward_hook)
        self.hook_b = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.forward_out = output

    def backward_hook(self, module, grad_in, grad_out):
        self.backward_out = grad_out[0]

    def unregister_hook(self):
        self.hook_f.remove()
        self.hook_b.remove()

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se = SEBlock(256)
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.se(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

def calculate_emotion_weight(emotion_angles):
    emotion_weight = {}
    for i, angle in emotion_angles.items():
        rad = math.radians(angle)
        weight = math.cos(rad)
        if i == 6:  # Neutral
            weight = 0.0
        elif i == 1:  # Surprise
            weight = 0.5
        emotion_weight[i] = weight
    return emotion_weight

def get_satisfaction_category(si):
    if si > 0.75:
        return "Satisfactory"
    elif si >= 0.5 and si <= 0.75:
        return "Neutral"
    else:
        return "Dissatisfactory"

emotion_angles = {
    0: 7.8,    # Happy
    1: 48.6,   # Surprise
    2: 207.5,  # Sad
    3: 120.0,  # Anger
    4: 240.0,  # Disgust
    5: 150.0,  # Fear
    6: 90.0    # Neutral
}

emotion_weight = calculate_emotion_weight(emotion_angles)
w_min = min(emotion_weight.values())
w_max = max(emotion_weight.values())

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

class_labels = ['Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Neutral']

model = ResEmoteNet(num_classes=7).to(device)
model_path = "./models/best_model_resemotenet_80.pth"
if not os.path.exists(model_path):
    model_path = "./models/final_model_resemotenet_80.pth"
    print("Best model not found, using final model.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

final_layer = model.conv3
hook = Hook()
hook.register_hook(final_layer)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (154, 1, 254)
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4

def detect_emotion(pil_crop_img):
    try:
        if not isinstance(pil_crop_img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(pil_crop_img)}")
        vid_fr_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
        logits = model(vid_fr_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        predicted_class_idx = predicted_class.item()
        confidence = probabilities[0][predicted_class_idx].item() * 100

        one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_().to(device)
        one_hot_output[0][predicted_class_idx] = 1
        logits.backward(gradient=one_hot_output, retain_graph=True)

        gradients = hook.backward_out
        feature_maps = hook.forward_out

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = cam.clamp(min=0).squeeze()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().detach().numpy()

        scores = probabilities.cpu().detach().numpy().flatten()
        rounded_scores = [round(float(score), 2) for score in scores]
        return rounded_scores, cam, confidence
    except Exception as e:
        logger.error(f"Error in detect_emotion: {e}")
        return None, None, None

def plot_heatmap(x, y, w, h, cam, pil_crop_img, image):
    try:
        pass
    except Exception as e:
        logger.error(f"Error in plot_heatmap: {e}")

def update_max_emotion(rounded_scores):
    max_index = np.argmax(rounded_scores)
    return class_labels[max_index]

def print_max_emotion(x, y, max_emotion, image, confidence=None, si=None, category=None):
    org = (x, y - 15)
    text = f"{max_emotion}: {confidence:.2f}%" if confidence is not None else max_emotion
    cv2.putText(image, text, org, font, font_scale, font_color, thickness, line_type)
    if si is not None and category is not None:
        org2 = (x, y - 35)
        text2 = f"SI: {si:.2f} ({category})"
        cv2.putText(image, text2, org2, font, font_scale, font_color, thickness, line_type)

def print_all_emotion(x, y, w, rounded_scores, image):
    org = (x + w + 10, y)
    for index, value in enumerate(class_labels):
        emotion_str = f'{value}: {rounded_scores[index]:.2f}'
        y_offset = org[1] + (index * 30)
        cv2.putText(image, emotion_str, (org[0], y_offset), font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(image, use_mediapipe=True):
    faces = {}
    try:
        if use_mediapipe:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image)
                if not results.detections:
                    return faces, {}
                emotion_scores_dict = {}
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    faces[f'face_{i}'] = {'facial_area': [x1, y1, x2, y2]}
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pil_crop_img = Image.fromarray(image[y1:y2, x1:x2])
                    rounded_scores, cam, confidence = detect_emotion(pil_crop_img)
                    if rounded_scores is None or cam is None or confidence is None:
                        continue
                    si = sum(float(rounded_scores[j]) * emotion_weight[j] for j in range(7))
                    si_normalized = (si - w_min) / (w_max - w_min)
                    si_normalized = max(0, min(1, si_normalized))
                    category = get_satisfaction_category(si_normalized)
                    emotion_scores_dict[f'face_{i}'] = {
                        'scores': [float(score) for score in rounded_scores],
                        'confidence': float(round(confidence, 2)),
                        'satisfaction_index': float(round(si_normalized, 2)),
                        'satisfaction_category': category
                    }
                    max_emotion = update_max_emotion(rounded_scores)
                    plot_heatmap(x1, y1, x2 - x1, y2 - y1, cam, pil_crop_img, image)
                    print_max_emotion(x1, y1, max_emotion, image, confidence, si_normalized, category)
        return faces, emotion_scores_dict
    except Exception as e:
        logger.error(f"Error in detect_bounding_box: {e}")
        return {}, {}

def process_frame(frame, frame_id, results=None):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, emotion_scores_dict = detect_bounding_box(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results is not None:
            frame_results = {'frame_id': frame_id, 'faces': []}
            for face_id, data in faces.items():
                if face_id in emotion_scores_dict:
                    frame_results['faces'].append({
                        'face_id': face_id,
                        'emotion_scores': {k: float(v) for k, v in zip(class_labels, emotion_scores_dict[face_id]['scores'])},
                        'max_emotion': update_max_emotion(emotion_scores_dict[face_id]['scores']),
                        'confidence': float(emotion_scores_dict[face_id]['confidence']),
                        'satisfaction_index': float(emotion_scores_dict[face_id]['satisfaction_index']),
                        'satisfaction_category': emotion_scores_dict[face_id]['satisfaction_category']
                    })
            if frame_results['faces']:
                results.append(frame_results)
        return frame_bgr
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return frame

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

@app.route('/process_camera_frame', methods=['POST'])
def process_camera_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("No image data provided")
            return jsonify({'error': 'No image data provided'}), 400
        
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400
        
        frame_id = int(time.time() * 1000)
        results = []
        processed_frame = process_frame(frame, frame_id, results)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_img_data = base64.b64encode(buffer).decode('utf-8')
        
        serialized_results = []
        if results:
            for result in results[0]['faces']:
                serialized_result = {
                    'face_id': result['face_id'],
                    'emotion_scores': {k: float(v) for k, v in result['emotion_scores'].items()},
                    'max_emotion': result['max_emotion'],
                    'confidence': float(result['confidence']),
                    'satisfaction_index': float(result['satisfaction_index']),
                    'satisfaction_category': result['satisfaction_category']
                }
                serialized_results.append(serialized_result)
        
        logger.info("Frame processed successfully")
        return jsonify({
            'results': serialized_results if serialized_results else [],
            'image': f'data:image/jpeg;base64,{processed_img_data}'
        })
    except Exception as e:
        logger.error(f"Error in process_camera_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_face_crops', methods=['POST'])
def process_face_crops():
    try:
        data = request.get_json()
        if not data or 'faces' not in data:
            logger.error("No face data provided")
            return jsonify({'error': 'No face data provided'}), 400
        
        results = []
        for face_data in data['faces']:
            img_data = face_data['image'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pil_face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            rounded_scores, _, confidence = detect_emotion(pil_face_img)
            if rounded_scores is None or confidence is None:
                continue
            si = sum(float(rounded_scores[j]) * emotion_weight[j] for j in range(7))
            si_normalized = (si - w_min) / (w_max - w_min)
            si_normalized = max(0, min(1, si_normalized))
            category = get_satisfaction_category(si_normalized)
            max_emotion = update_max_emotion(rounded_scores)
            results.append({
                'face_id': f"face_{face_data['id']}",
                'emotion_scores': {k: float(v) for k, v in zip(class_labels, rounded_scores)},
                'max_emotion': max_emotion,
                'confidence': float(round(confidence, 2)),
                'satisfaction_index': float(round(si_normalized, 2)),
                'satisfaction_category': category
            })
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error in process_face_crops: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            if image is None:
                flash('Invalid image file')
                return redirect(request.url)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, emotion_scores_dict = detect_bounding_box(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            output_filename = f'processed_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, image_bgr)
            results = []
            for face_id, data in faces.items():
                if face_id in emotion_scores_dict:
                    emotion_scores = {k: float(v) for k, v in zip(class_labels, emotion_scores_dict[face_id]['scores'])}
                    max_emotion = update_max_emotion(emotion_scores_dict[face_id]['scores'])
                    results.append({
                        'face_id': face_id,
                        'emotion_scores': emotion_scores,
                        'max_emotion': max_emotion,
                        'confidence': float(emotion_scores_dict[face_id]['confidence']),
                        'satisfaction_index': float(emotion_scores_dict[face_id]['satisfaction_index']),
                        'satisfaction_category': emotion_scores_dict[face_id]['satisfaction_category']
                    })
            hook.unregister_hook()
            return render_template('result.html', output_image=output_filename, results=results)
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.mp4'
            filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(filepath)
            logger.debug(f"Saved video to: {filepath}")
            if not os.path.exists(filepath):
                flash('Failed to save video file')
                return redirect(request.url)
            return render_template('video_stream.html', video_path=filepath)
    return render_template('video_upload.html')

def generate_video_feed(video_path):
    logger.debug(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return
    try:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, frame_id)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_id += 1
    except Exception as e:
        logger.error(f"Error in generate_video_feed: {e}")
    finally:
        cap.release()

@app.route('/video_stream_upload/<path:video_path>')
def video_stream_upload(video_path):
    return Response(generate_video_feed(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera_feed():
    return render_template('camera.html')

def generate_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return
    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, frame_id)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_id += 1
    finally:
        cap.release()

@app.route('/video_stream')
def video_stream():
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)