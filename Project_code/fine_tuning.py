import os
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv
import keras
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# 데이터셋의 경로와 클래스 레이블을 CSV에 매핑
data_dir = 'D:\Embedd Practice\sample_data'
classes = ['car', 'human', 'cat']  # 레이블을 정의 및 클래스 추가

# YAMNet 모델 로드
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# wav 파일 로드 및 라벨링
def load_data(data_dir, classes):
    wav_files = []
    labels = []
    for idx, label in enumerate(classes):
        class_dir = os.path.join(data_dir, label)
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(class_dir, file)
                wav_files.append(file_path)
                labels.append(idx)
    return wav_files, labels

wav_files, labels = load_data(data_dir, classes)

# Train/Test Split
train_files, test_files, train_labels, test_labels = train_test_split(wav_files, labels, test_size=0.2)

# 오디오 로드 및 처리
def process_wav_file(file_path):
    # librosa를 사용하여 오디오 파일 로드
    waveform, sample_rate = librosa.load(file_path, sr=16000)  # 직접 16kHz로 로드
    
    # 정규화
    waveform = librosa.util.normalize(waveform)
    
    return waveform, sample_rate

# YAMNet을 사용하여 특징 추출
def extract_features(waveform, sample_rate):
    # 입력 데이터를 float32로 변환
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    
    # 패딩이나 자르기를 통해 모든 오디오를 3초(48000 샘플)로 맞춤
    target_length = 48000
    if len(waveform) < target_length:
        # 짧은 경우 패딩
        padding = target_length - len(waveform)
        waveform = tf.pad(waveform, [[0, padding]])
    else:
        # 긴 경우 자르기
        waveform = waveform[:target_length]
    
    # YAMNet 특징 추출
    scores, embeddings, _ = yamnet_model(waveform)
    # 임베딩의 평균을 특징 벡터로 사용
    return tf.reduce_mean(embeddings, axis=0)

# 데이터셋 준비
def prepare_dataset(files):
    features = []
    for file in files:
        try:
            print(f"Processing file: {file}")  # 진행 상황 출력
            waveform, sample_rate = process_wav_file(file)
            feature = extract_features(waveform, sample_rate)
            features.append(feature.numpy())
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    return np.array(features)

# 특징 추출
print("특징 추출 시작...")
X_train = prepare_dataset(train_files)
X_test = prepare_dataset(test_files)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 데이터가 비어있지 않은지 확인
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("No features were extracted successfully")

print(f"훈련 데이터 형태: {X_train.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")

# 분류 모델 정의
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(classes), activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
print("모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 모델 저장
model.save('audio_classification_model')