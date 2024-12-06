# Real Test
import os
import pyaudio
import wave
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import time

class RealtimeAudioClassifier:
    def __init__(self):
        # 오디오 설정
        self.CHUNK = 1024 * 16
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3
        self.CONFIDENCE_THRESHOLD = 70.0  # 70% 임계값 설정

        # YAMNet 모델 로드
        print("Loading YAMNet model...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # 분류 모델 로드
        print("Loading classification model...")
        self.model = load_model(r'D:\Embedded Project\audio_classification_model')
        self.classes = ['car_driving', 'car_horn', 'human_laugh', 'human_talk', 'cat', 'dog', 'construction_site']

        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        print("Recording started... Press Ctrl+C to stop")
        
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        try:
            while True:
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))

                audio_data = np.concatenate(frames)
                
                # 분석 및 예측
                top_predictions = self.process_audio(audio_data)
                
                # 결과 출력
                print("\n" + "="*50)
                if top_predictions[0][1] >= self.CONFIDENCE_THRESHOLD:
                    print("Top 3 Predictions:")
                    for i, (class_name, confidence) in enumerate(top_predictions[:3], 1):
                        print(f"{i}. {class_name}: {confidence:.2f}%")
                else:
                    print("Predicted: no data")
                    print("(Top confidence was: {:.2f}%)".format(top_predictions[0][1]))
                print("="*50)

        except KeyboardInterrupt:
            print("\nRecording stopped.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

    def process_audio(self, audio_data):
        try:
            # 오디오 정규화
            audio_data = librosa.util.normalize(audio_data)
            
            # 3초(48000 샘플)로 길이 조정
            target_length = 48000
            if len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            else:
                audio_data = audio_data[:target_length]

            # tf.float32로 변환
            waveform = tf.cast(audio_data, dtype=tf.float32)
            
            # YAMNet 특징 추출
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            features = tf.reduce_mean(embeddings, axis=0)

            # 예측
            features_np = features.numpy().reshape(1, -1)
            prediction = self.model.predict(features_np, verbose=0)
            
            # 상위 3개 예측 결과 추출
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_predictions = [
                (self.classes[idx], float(prediction[0][idx] * 100))
                for idx in top_3_indices
            ]

            return top_3_predictions

        except Exception as e:
            print(f"Error in processing audio: {str(e)}")
            return [("Error", 0.0), ("Error", 0.0), ("Error", 0.0)]

def main():
    try:
        classifier = RealtimeAudioClassifier()
        classifier.start_recording()
    except Exception as e:
        print(f"Error initializing classifier: {str(e)}")

if __name__ == "__main__":
    main()