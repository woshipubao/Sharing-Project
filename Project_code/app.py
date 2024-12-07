# Application Test
from flask import Flask, render_template, jsonify
import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import threading
import queue
import time
import os

app = Flask(__name__)

# 전역 변수로 최신 예측 결과 저장
latest_predictions = {
    'predictions': [],
    'timestamp': None
}

# 오디오 처리를 위한 클래스
class AudioClassifier:
    def __init__(self):
        self.CHUNK = 1024 * 16
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3
        self.CONFIDENCE_THRESHOLD = 70.0

        # 모델 로드
        print("Loading models...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        current_dir = os.getcwd()
        model_path = os.path.join(current_dir, 'audio_classification_model')
        self.model = tf.keras.modelsload_model(model_path)
        
        self.classes = ['car_driving', 'car_horn', 'human_laugh', 'human_talk', 'cat', 'dog', 'construction_site', 'siren']
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
    def process_audio(self, audio_data):
        try:
            audio_data = librosa.effects.preemphasis(audio_data, coef=0.97)
            audio_data = librosa.util.normalize(audio_data)
            
            target_length = 48000
            if len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            else:
                audio_data = audio_data[:target_length]

            waveform = tf.cast(audio_data, dtype=tf.float32)
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            features = tf.reduce_mean(embeddings, axis=0)
            
            features_np = features.numpy().reshape(1, -1)
            prediction = self.model.predict(features_np, verbose=0)
            
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.classes[idx],
                    'confidence': float(prediction[0][idx] * 100)
                }
                for idx in top_3_indices
            ]

            return top_3_predictions

        except Exception as e:
            print(f"Error in processing audio: {str(e)}")
            return []

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            predictions = self.process_audio(audio_data)
            
            global latest_predictions
            latest_predictions['predictions'] = predictions
            latest_predictions['timestamp'] = time.time()
            
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        if not self.is_recording:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

    def __del__(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

# 전역 인스턴스 생성
classifier = AudioClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    classifier.start_recording()
    return jsonify({'status': 'success', 'message': 'Recording started'})

@app.route('/stop_recording')
def stop_recording():
    classifier.stop_recording()
    return jsonify({'status': 'success', 'message': 'Recording stopped'})

@app.route('/get_predictions')
def get_predictions():
    global latest_predictions
    
    if not latest_predictions['timestamp'] or \
       time.time() - latest_predictions['timestamp'] > 5:
        return jsonify({
            'status': 'no_data',
            'predictions': []
        })

    predictions = latest_predictions['predictions']
    if not predictions or predictions[0]['confidence'] < classifier.CONFIDENCE_THRESHOLD:
        return jsonify({
            'status': 'below_threshold',
            'top_confidence': predictions[0]['confidence'] if predictions else 0
        })

    return jsonify({
        'status': 'success',
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(debug=True)
