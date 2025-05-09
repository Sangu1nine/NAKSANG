# TensorFlow import를 가장 먼저
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 나머지 라이브러리
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
base_path = "."
models_dir = os.path.join(base_path, 'models')
data_path = os.path.join(base_path, 'data')
os.makedirs(models_dir, exist_ok=True)

print("데이터 로딩 중...")
X_train_full = np.load(os.path.join(data_path, 'X_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_train_full = np.load(os.path.join(data_path, 'y_train.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))
print("데이터 로딩 완료!")

# 디렉토리 생성
results_dir = os.path.join(models_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# 훈련 세트와 검증 세트 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# 클래스 분포 확인
train_class_counts = np.bincount(y_train.astype(int))
val_class_counts = np.bincount(y_val.astype(int))
test_class_counts = np.bincount(y_test.astype(int))

# 비율 계산 함수
def get_ratio(counts):
    total = sum(counts)
    return [f"{count}개 ({count / total * 100:.1f}%)" for count in counts]

# 출력
print("클래스 분포:")
print(f"훈련 세트: 비낙상 {get_ratio(train_class_counts)[0]}, 낙상 {get_ratio(train_class_counts)[1]}")
print(f"검증 세트: 비낙상 {get_ratio(val_class_counts)[0]}, 낙상 {get_ratio(val_class_counts)[1]}")
print(f"테스트 세트: 비낙상 {get_ratio(test_class_counts)[0]}, 낙상 {get_ratio(test_class_counts)[1]}")

model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LayerNormalization(),
    Dropout(0.3),

    LSTM(16),
    LayerNormalization(),
    Dropout(0.3),

    Dense(8, activation='relu'),
    LayerNormalization(),

    Dense(1, activation='sigmoid')  # 이진 분류
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 예측 및 성능 지표 계산
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("\n분류 보고서:")
print(classification_report(y_test, y_pred_classes, target_names=['비낙상', '낙상']))

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['비낙상', '낙상'],
            yticklabels=['비낙상', '낙상'])
plt.title('혼동 행렬')
plt.ylabel('실제')
plt.xlabel('예측')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# 학습 곡선 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train_Loss')
plt.plot(history.history['val_loss'], label='Validation_Loss')
plt.title('Model_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_Accuracy')
plt.title('Model_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'learning_curves.png'))
plt.close()

# 모델 저장 (.keras 형식으로 변경)
keras_model_path = os.path.join(results_dir, 'fall_detection_model.keras')
model.save(keras_model_path)
print(f"\n모델 저장 완료: {keras_model_path}")

# 이전 버전과의 호환성을 위해 h5 형식으로도 저장
h5_model_path = os.path.join(results_dir, 'fall_detection_model.h5')
model.save(h5_model_path, save_format='h5')
print(f"호환성을 위한 h5 모델 저장 완료: {h5_model_path}")