# YAMnet Test Code
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import scipy.signal

# 모델 불러오기
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# 가장 높은 점수를 받은 클래스 찾기
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# 로드된 오디오가 적절한 sample_rate(16K)인지 확인하고 변환하는 메서드
def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

# wav 파일 불러오기'
current_dir = os.getcwd()
wav_file_path = os.path.join(current_dir, 'sample_rate', 'car_horn', 'car_horn7.wav')

sample_rate, wav_data = wavfile.read(wav_file_path, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)


# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Listening to the wav file.
Audio(wav_data, rate=sample_rate)

waveform = wav_data / tf.int16.max # wav_data를 [-1.0, 1.0]의 값으로 정규화

# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')
