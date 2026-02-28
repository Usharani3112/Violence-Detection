"""
Violence Detection Flask Application
Features: Live Webcam Detection + Video Upload + Alarm System
"""

from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import tempfile
import json
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
model = None
IMG_SIZE = 224  # Updated to match training
camera = None
detection_active = False
violence_detected_flag = False

# Load configuration
CONFIG_PATH = 'config.json'
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
else:
    # Default configuration
    CONFIG = {
        'webcam': {'threshold': 0.80, 'detection_interval': 15, 'smoothing_window': 5},
        'video': {'threshold': 0.75, 'num_frames': 16, 'min_violence_frame_ratio': 0.4}
    }

# CBAM Architecture Functions
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    return layers.Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature, ratio=8):
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def build_model(img_size=224):
    """Build CNN + CBAM model"""
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)
    
    # Classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def load_model():
    """Load trained model from Colab"""
    global model, IMG_SIZE
    
    model_dir = 'model'
    info_path = os.path.join(model_dir, 'model_info.json')
    
    # Load model info
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            IMG_SIZE = info.get('img_size', 224)
    
    # Try to load complete model first (Keras 3.x format)
    model_path = os.path.join(model_dir, 'complete_model.keras')
    
    if os.path.exists(model_path):
        print(f'Loading complete model from {model_path}...')
        try:
            # Load with custom objects for Lambda layers
            custom_objects = {
                'reduce_mean': lambda x, axis=-1, keepdims=True: tf.reduce_mean(x, axis=axis, keepdims=keepdims),
                'reduce_max': lambda x, axis=-1, keepdims=True: tf.reduce_max(x, axis=axis, keepdims=keepdims)
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            
            # Recompile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            accuracy = info.get('accuracy', 0) * 100 if os.path.exists(info_path) else 0
            print(f'✓ Model loaded successfully: {model.count_params():,} parameters')
            if accuracy > 0:
                print(f'✓ Model accuracy: {accuracy:.2f}%')
            
            return model
        except Exception as e:
            print(f'⚠️ Could not load complete model: {e}')
            print('Falling back to building architecture and loading weights...')
    
    # Fallback: Build architecture and load weights
    print(f'Building model architecture (IMG_SIZE={IMG_SIZE})...')
    model = build_model(IMG_SIZE)
    
    # Try multiple weight files
    weight_files = [
        'best_checkpoint.weights.h5',
        'final_model.weights.h5',
        'model.weights.h5'
    ]
    
    weights_path = None
    for wf in weight_files:
        path = os.path.join(model_dir, wf)
        if os.path.exists(path):
            weights_path = path
            break
    
    if not weights_path:
        raise FileNotFoundError(
            'No model file found. Please:\n'
            '1. Download complete_model.keras from Colab\n'
            '2. Place it in the model/ folder\n'
            '3. Run: upgrade_to_keras3.bat\n'
            '4. Run: python app.py'
        )
    
    print(f'Loading weights from {weights_path}...')
    model.load_weights(weights_path)
    
    accuracy = info.get('accuracy', 0) * 100 if os.path.exists(info_path) else 0
    print(f'✓ Model loaded successfully: {model.count_params():,} parameters')
    if accuracy > 0:
        print(f'✓ Model accuracy: {accuracy:.2f}%')
    
    return model

def preprocess_frame(frame):
    """Preprocess frame for prediction"""
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return frame

def predict_frame(frame):
    """Predict violence in single frame"""
    frame_processed = preprocess_frame(frame)
    frame_batch = np.expand_dims(frame_processed, axis=0)
    prediction = model.predict(frame_batch, verbose=0)
    
    # Get both class probabilities
    nonviolence_prob = float(prediction[0][0])
    violence_prob = float(prediction[0][1])
    
    # Debug: Print to console occasionally
    if np.random.random() < 0.1:
        print(f'[DEBUG] NonViolence: {nonviolence_prob:.3f}, Violence: {violence_prob:.3f}')
    
    return violence_prob

def predict_video(video_path, threshold=None, num_frames=None):
    """Predict violence in video file"""
    if threshold is None:
        threshold = CONFIG['video']['threshold']
    if num_frames is None:
        num_frames = CONFIG['video']['num_frames']
    
    min_violence_ratio = CONFIG['video']['min_violence_frame_ratio']
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return {'error': 'Cannot read video'}
    
    # Sample frames evenly
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = preprocess_frame(frame)
            frames.append(frame_processed)
    
    cap.release()
    
    if len(frames) == 0:
        return {'error': 'No frames extracted'}
    
    # Predict all frames
    frames_array = np.array(frames)
    predictions = model.predict(frames_array, verbose=0)
    
    # Get violence probabilities
    violence_probs = predictions[:, 1]
    
    # Calculate statistics
    avg_violence_prob = float(np.mean(violence_probs))
    max_violence_prob = float(np.max(violence_probs))
    
    # Count frames exceeding threshold
    violence_frame_count = int(np.sum(violence_probs > threshold))
    violence_frame_ratio = float(violence_frame_count / len(violence_probs))
    
    # Simple logic: frame ratio AND average
    is_violence = bool((violence_frame_ratio > min_violence_ratio) and (avg_violence_prob > threshold))
    
    return {
        'result': 'VIOLENCE DETECTED' if is_violence else 'NO VIOLENCE',
        'is_violence': is_violence,
        'violence_probability': avg_violence_prob,
        'nonviolence_probability': float(1 - avg_violence_prob),
        'max_probability': max_violence_prob,
        'violence_frames': violence_frame_count,
        'total_frames': int(len(violence_probs)),
        'violence_frame_ratio': violence_frame_ratio,
        'confidence': float(max(avg_violence_prob, 1 - avg_violence_prob)),
        'confidence_percent': f'{max(avg_violence_prob, 1 - avg_violence_prob) * 100:.2f}%',
        'frames_analyzed': len(frames),
        'threshold_used': float(threshold)
    }

def generate_webcam_frames():
    """Generate webcam frames with violence detection"""
    global camera, violence_detected_flag
    
    camera = cv2.VideoCapture(0)
    frame_count = 0
    
    # Get config values
    detection_interval = CONFIG['webcam']['detection_interval']
    violence_threshold = CONFIG['webcam']['threshold']
    smoothing_window = CONFIG['webcam']['smoothing_window']
    
    # Smoothing: track last N predictions
    recent_predictions = []
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect violence every N frames
        if frame_count % detection_interval == 0:
            violence_prob = predict_frame(frame)
            
            # Add to recent predictions for smoothing
            recent_predictions.append(violence_prob)
            if len(recent_predictions) > smoothing_window:
                recent_predictions.pop(0)
            
            # Use average of recent predictions for stability
            avg_recent_prob = float(np.mean(recent_predictions))
            
            # Require sustained high probability to trigger
            violence_detected_flag = avg_recent_prob > violence_threshold
            
            # Print to console for debugging
            print(f'[WEBCAM] Violence: {avg_recent_prob:.2%} | Threshold: {violence_threshold:.2%} | Alert: {violence_detected_flag}')
            
            # Draw result on frame with both probabilities
            if violence_detected_flag:
                color = (0, 0, 255)  # Red
                text = f'VIOLENCE: {avg_recent_prob*100:.1f}%'
            else:
                color = (0, 255, 0)  # Green
                text = f'SAFE: {(1-avg_recent_prob)*100:.1f}%'
            
            # Show both probabilities for debugging
            debug_text = f'V:{avg_recent_prob*100:.0f}% NV:{(1-avg_recent_prob)*100:.0f}% | Thresh:{violence_threshold*100:.0f}%'
            
            cv2.rectangle(frame, (10, 10), (450, 80), color, -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, debug_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        frame_count += 1
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if camera:
        camera.release()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam detection"""
    global detection_active
    detection_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam detection"""
    global detection_active, camera
    detection_active = False
    if camera:
        camera.release()
    return jsonify({'status': 'stopped'})

@app.route('/webcam_feed')
def webcam_feed():
    """Video streaming route"""
    return Response(generate_webcam_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_violence')
def check_violence():
    """Check if violence is detected"""
    return jsonify({'violence_detected': violence_detected_flag})

@app.route('/predict_video', methods=['POST'])
def predict_video_route():
    """Handle video upload and prediction"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        result = predict_video(filepath)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/test_prediction', methods=['POST'])
def test_prediction():
    """Test endpoint to see raw predictions"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Read image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Predict
    frame_processed = preprocess_frame(frame)
    frame_batch = np.expand_dims(frame_processed, axis=0)
    prediction = model.predict(frame_batch, verbose=0)
    
    return jsonify({
        'nonviolence_probability': float(prediction[0][0]),
        'violence_probability': float(prediction[0][1]),
        'predicted_class': 'Violence' if prediction[0][1] > prediction[0][0] else 'NonViolence',
        'confidence': float(max(prediction[0]))
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print('='*60)
    print('🎯 Violence Detection System - Flask Server')
    print('='*60)
    
    try:
        load_model()
        print('✓ Model ready')
        
        print('\n🚀 Server starting...')
        print('   URL: http://127.0.0.1:5000')
        print('   Features:')
        print('   • Live Webcam Detection')
        print('   • Video Upload Detection')
        print('   • Alarm System')
        print('\n   Press Ctrl+C to stop')
        print('='*60)
        
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        print('\nMake sure you have:')
        print('1. model/model.weights.h5 file')
        print('2. model/model_info.json file')
        input('\nPress Enter to exit...')
