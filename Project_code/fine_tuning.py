import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv
import keras
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 캐시 디렉토리 설정 및 정리, 모델 재학습 시 충돌 방지
cache_dir = os.path.join(os.path.expanduser('~'), 'tfhub_modules')
temp_cache_dir = os.path.join(os.environ.get('TEMP', '/tmp'), 'tfhub_modules')

# 기존 캐시 삭제
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
if os.path.exists(temp_cache_dir):
    shutil.rmtree(temp_cache_dir)

# 새 캐시 디렉토리 생성
os.makedirs(cache_dir, exist_ok=True)

# 환경 변수 설정
os.environ['TFHUB_CACHE_DIR'] = cache_dir

# YAMNet 모델 로드 - 대체 방법 사용
try:
    print("Attempting to load YAMNet model...")
    yamnet_model = hub.KerasLayer('https://tfhub.dev/google/yamnet/1')
    print("Model loaded successfully!")
except Exception as e:
    print(f"First attempt failed: {str(e)}")
    try:
        print("Trying alternative loading method...")
        model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(model_handle)
        print("Model loaded successfully with alternative method!")
    except Exception as e:
        print(f"Second attempt failed: {str(e)}")
        raise

# Class weight 계산 함수 추가
def compute_class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes = unique_classes,
        y = labels
    )
    return dict(zip(unique_classes, class_weights))

# 오디오 데이터 증강 함수 추가
def augment_audio(waveform):
    augmented = []
    # 기존 증강에 추가:
    # 볼륨 변경
    augmented.append(waveform * np.random.uniform(0.8, 1.2))
    # 시간 역전
    augmented.append(waveform[::-1])
    # 다양한 노이즈 레벨
    for noise_level in [0.003, 0.005, 0.007]:
        noise = np.random.normal(0, noise_level, len(waveform))
        augmented.append(waveform + noise)
    return augmented

# 데이터셋의 경로와 클래스 레이블을 CSV에 매핑
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'sample_data')  # 상대 경로로 변경
classes = ['car_driving', 'car_horn', 'human_laugh', 'human_talk', 'cat', 'dog', 'construction_site', 'siren']  # 레이블을 정의 및 클래스 추가

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
train_files, test_files, train_labels, test_labels = train_test_split(
    wav_files, labels, test_size=0.25, stratify=labels, random_state=42
)

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
def prepare_dataset(files, labels, augment=False):
    features = []
    final_labels = []

    for file, label in zip(files, labels):
        try:
            print(f"Processing file: {file}")  # 진행 상황 출력
            waveform, sample_rate = process_wav_file(file)

            if augment and len(files) < 50: # 50개 미만인 클래스만 증강
                augment_waveforms = augment_audio(waveform)
                for aug_waveform in augment_waveforms:
                    feature = extract_features(aug_waveform, sample_rate)
                    features.append(feature.numpy())
                    final_labels.append(label)
            else:
                feature = extract_features(waveform, sample_rate)
                features.append(feature.numpy())
                final_labels.append(label)

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    return np.array(features), np.array(final_labels)

# 특징 추출
print("특징 추출 시작...")
X_train, y_train = prepare_dataset(train_files, train_labels, augment=True)
X_test, y_test = prepare_dataset(test_files, test_labels, augment=False)

# 클래스 가중치 계산
class_weights = compute_class_weights(y_train)
print("Class weights:", class_weights)

# 데이터가 비어있지 않은지 확인
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("No features were extracted successfully")

print(f"훈련 데이터 형태: {X_train.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")

# 분류 모델 정의 및 정규화 추가
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(1024,),
                      kernel_regularizer=tf.keras.regularizers.l2(0.005)),  # L2 규제 완화
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    keras.layers.Dense(len(classes), activation='softmax')
])

# 2. 데이터 증강은 유지하되 노이즈 수준을 조절
def augment_audio(waveform):
    augmented = []
    augmented.append(waveform)
    
    # 노이즈 레벨 낮춤
    noise = np.random.normal(0, 0.003, len(waveform))
    augmented.append(waveform + noise)
    
    # 시간 스트레칭 범위 조절
    stretched = librosa.effects.time_stretch(waveform, rate=0.95)
    augmented.append(stretched[:len(waveform)])
    
    # 피치 시프트 범위 조절
    pitched = librosa.effects.pitch_shift(waveform, sr=16000, n_steps=1)
    augmented.append(pitched)
    
    return augmented

# 3. 학습 파라미터 최적화
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=0.0001,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,  # 더 길게 학습할 기회 제공
    restore_best_weights=True,
    verbose=1
)

# 4. 데이터셋 분할 비율 조정
train_files, test_files, train_labels, test_labels = train_test_split(
    wav_files, labels, test_size=0.25, stratify=labels, random_state=42
)

# 학습률 스케줄러 추가
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=0.0001,
    verbose=1
)

# Early stopping 추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
print("모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,  # 배치 크기 줄임
    class_weight=class_weights,  # 클래스 가중치 적용
    callbacks=[lr_schedule, early_stopping]
)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 학습 과정 시각화 함수
def plot_training_history(history, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(current_dir, 'Project_code')  # 상대 경로로 변경
    # Accuracy 그래프
    plt.figure(figsize=(12, 4))
    
    # subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 그래프 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"학습 히스토리 그래프가 저장되었습니다: {os.path.join(save_dir, 'training_history.png')}")

# 학습 히스토리 시각화 및 저장
plot_training_history(history)

# 모델 저장
model_save_dir = os.path.join(current_dir, 'audio_classification_model')  # 상대 경로로 변경
model.save(model_save_dir)
print(f"모델이 저장되었습니다: {model_save_dir}")
