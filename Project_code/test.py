import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile
import scipy.signal

# 모델 불러오기
model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                 original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def process_audio(wav_file_name):
    try:
        # WAV 파일 불러오기
        sample_rate, wav_data = wavfile.read(wav_file_name)
        
        # 스테레오를 모노로 변환 (만약 스테레오 파일이라면)
        if len(wav_data.shape) > 1:
            wav_data = np.mean(wav_data, axis=1)
        
        # 샘플 레이트 확인 및 조정
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        
        # 오디오 정보 출력
        duration = len(wav_data)/sample_rate
        print(f'Sample rate: {sample_rate} Hz')
        print(f'Total duration: {duration:.2f}s')
        print(f'Size of the input: {len(wav_data)}')
        
        # 데이터 정규화 (-1 to 1 범위로)
        waveform = wav_data.astype(np.float32) / 32768.0
        
        # 모델 실행
        scores, embeddings, spectrogram = model(waveform)
        
        # 결과 처리
        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)
        
        scores_np = scores.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        print(f'The main sound is: {infered_class}')
        
        # Top 5 소리 출력
        top_5_indices = scores_np.mean(axis=0).argsort()[-5:][::-1]
        print("\nTop 5 sounds detected:")
        for idx in top_5_indices:
            score = scores_np.mean(axis=0)[idx]
            print(f"{class_names[idx]}: {score:.3f}")
        
        return scores_np, embeddings, spectrogram, class_names
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        raise

if __name__ == "__main__":
    wav_file_name = 'D:\\pratice\\Embedd Practice\\sample_data\\car_driving1.wav'
    scores, embeddings, spectrogram, class_names = process_audio(wav_file_name)