# model test code
import os
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
from keras.models import load_model

# YAMNet 모델 로드
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 저장된 모델 로드
model = load_model('audio_classification_model')

# 클래스 레이블 정의
classes = ['car', 'human', 'cat', 'dog']

# 오디오 파일 처리 및 특징 추출
def process_wav_file(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    waveform = librosa.util.normalize(waveform)
    return waveform, sample_rate

def extract_features(waveform):
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    
    # 패딩이나 자르기를 통해 모든 오디오를 3초(48000 샘플)로 맞춤
    target_length = 48000
    if len(waveform) < target_length:
        padding = target_length - len(waveform)
        waveform = tf.pad(waveform, [[0, padding]])
    else:
        waveform = waveform[:target_length]

    scores, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0)

# 예측 함수
def predict(file_path):
    try:
        waveform, _ = process_wav_file(file_path)
        feature = extract_features(waveform)
        feature = feature.numpy().reshape(1, -1)  # 모델 입력 형식에 맞게 차원 변경
        prediction = model.predict(feature)
        predicted_class = classes[np.argmax(prediction)]
        print(f"Predicted class for '{os.path.basename(file_path)}': {predicted_class}")
    except Exception as e:
        print(f"Error predicting file {file_path}: {str(e)}")

# 테스트할 오디오 파일 경로
test_file_path = 'D:\Embedd Practice\sample_data\car\car_horn3.wav'

# 예측 수행
predict(test_file_path)
