import os
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
from keras.models import load_model

# YAMNet 모델 로드
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 저장된 모델 로드
current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'audio_classification_model')
model = load_model(model_path)

# 클래스 레이블 정의
classes = ['car_driving', 'car_horn', 'human_laugh', 'human_talk', 'cat', 'dog', 'construction_site']

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
        prediction = model.predict(feature)[0]

        top_3_indices = np.argsort(prediction)[-3:][::-1]
        top_3_classes = [classes[i] for i in top_3_indices]
        top_3_confidence = [prediction[i] * 100 for i in top_3_indices]

        return list(zip(top_3_classes, top_3_confidence)), None
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_directory(directory_path):
    # 결과를 저장할 딕셔너리 초기화
    results = {cls: {'correct': 0, 'total': 0} for cls in classes}
    
    # 디렉토리 내의 모든 파일 처리
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):  # WAV 파일만 처리
                file_path = os.path.join(root, file)
                true_class = os.path.basename(os.path.dirname(file_path)).lower()
                
                # 예측 수행
                predictions, error = predict(file_path)  # predict 함수의 반환값을 올바르게 받음
                
                if predictions is not None:
                    # 결과 출력
                    print(f"\nFile: {file}")
                    print(f"True class: {true_class}")
                    print("Top 3 predictions:")
                    for i, (pred_class, confidence) in enumerate(predictions, 1):
                        print(f"{i}. {pred_class}: {confidence:.2f}%")
                    
                    # 통계 업데이트 (true_class가 classes 안에 있는 경우에만)
                    if true_class in results:
                        results[true_class]['total'] += 1
                        if predictions[0][0] == true_class:  # 1순위 예측이 정답인 경우
                            results[true_class]['correct'] += 1
                else:
                    print(f"\nFile: {file}")
                    print(f"Prediction failed: {error}")
    
    # 최종 통계 출력
    print("\n=== Final Results ===")
    total_correct = 0
    total_files = 0
    
    for cls in classes:
        if results[cls]['total'] > 0:
            accuracy = (results[cls]['correct'] / results[cls]['total']) * 100
            print(f"\n{cls.capitalize()}:")
            print(f"Correct predictions: {results[cls]['correct']}/{results[cls]['total']}")
            print(f"Accuracy: {accuracy:.2f}%")
            
            total_correct += results[cls]['correct']
            total_files += results[cls]['total']
    
    if total_files > 0:
        overall_accuracy = (total_correct / total_files) * 100
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

# 테스트 실행
test_directory = os.path.join(current_dir, 'real_data')
process_directory(test_directory)
