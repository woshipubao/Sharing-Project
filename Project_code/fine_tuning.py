import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# 데이터셋의 경로와 클래스 레이블을 CSV에 매핑
data_dir = 'D:\Embedd Practice\sample_data'
classes = ['car']  # 레이블을 정의

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
                labels.append(idx)  # 클래스 인덱스를 라벨로 사용
    return wav_files, labels

wav_files, labels = load_data(data_dir, classes)

# Train/Test Split
train_files, test_files, train_labels, test_labels = train_test_split(wav_files, labels, test_size=0.2)

# 오디오 로드 및 처리
def process_wav_file(file_path):
    sample_rate, wav_data = wavfile.read(file_path)
    waveform = wav_data / tf.int16.max  # [-1.0, 1.0] 값으로 정규화
    return waveform, sample_rate

# 데이터 전처리
train_waveforms = [process_wav_file(f)[0] for f in train_files]
test_waveforms = [process_wav_file(f)[0] for f in test_files]
