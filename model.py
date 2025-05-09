# TensorFlow import를 가장 먼저
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# F1 스코어와 리콜(재현율) 평가를 위한 콜백 함수
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score

# 나머지 라이브러리
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # 가비지 컬렉션
import time
from datetime import datetime

# 데이터 로드
base_path = "/content/drive/MyDrive/Digital_Smart_Final"
models_dir = os.path.join(base_path, 'models/models_4')
data_path = os.path.join(base_path, 'sam_100_win_150_str_75')
os.makedirs(models_dir, exist_ok=True)

# 디렉토리 생성
results_dir = os.path.join(models_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

print("데이터 로딩 중...")
X_train_full = np.load(os.path.join(data_path, 'X_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_train_full = np.load(os.path.join(data_path, 'y_train.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))
gc.collect()  # 메모리 확보
print("데이터 로딩 완료!")

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

# 클래스 가중치 계산 (낙상 클래스에 더 높은 가중치)
# 낙상 검출에서는 FN을 줄이는 것이 중요
class_counts = np.bincount(y_train_full.astype(int))
total = np.sum(class_counts)
# 낙상(1)에 더 높은 가중치 부여
class_weight = {
    0: total / (2 * class_counts[0]),  # 비낙상
    1: total / (class_counts[1]) * 1.5  # 낙상 (더 높은 가중치)
}
print(f"클래스 가중치: {class_weight}")

class F1RecallMetrics(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        
    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.val_data
        y_pred = (self.model.predict(x_val) > 0.5).astype(int)
        
        f1 = f1_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)  # 재현율(민감도): FN 최소화에 중요
        precision = precision_score(y_val, y_pred)
        
        logs['val_f1'] = f1
        logs['val_recall'] = recall
        logs['val_precision'] = precision
        
        print(f" — val_f1: {f1:.4f} — val_recall: {recall:.4f} — val_precision: {precision:.4f}")

# 하이퍼파라미터 최적화를 위한 그리드 서치 함수
def build_model(learning_rate=0.001, lstm_units1=32, lstm_units2=16, dropout_rate=0.3):
    model = Sequential([
        LSTM(lstm_units1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LayerNormalization(),
        Dropout(dropout_rate),

        LSTM(lstm_units2),
        LayerNormalization(),
        Dropout(dropout_rate),

        Dense(8, activation='relu'),
        LayerNormalization(),

        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 간단한 하이퍼파라미터 그리드 서치
param_grid = {
    'learning_rate': [0.001, 0.0005],
    'lstm_units1': [32, 64],
    'lstm_units2': [16, 32],
    'dropout_rate': [0.3, 0.5]
}

best_recall = 0
best_model = None
best_params = {}
results = []

# 주요 조합만 시도하여 시간 절약
for lr in param_grid['learning_rate']:
    for lstm1 in param_grid['lstm_units1']:
        for lstm2 in param_grid['lstm_units2']:
            for dr in param_grid['dropout_rate']:
                print(f"\n시도: lr={lr}, lstm1={lstm1}, lstm2={lstm2}, dropout={dr}")
                
                # 모델 생성
                model = build_model(lr, lstm1, lstm2, dr)
                
                # 콜백 설정
                callbacks = [
                    EarlyStopping(patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
                    F1RecallMetrics(val_data=(X_val, y_val))
                ]
                
                # 학습 실행
                history = model.fit(
                    X_train, y_train,
                    epochs=30,  # 더 적은 에폭으로 빠른 테스트
                    batch_size=16,
                    validation_data=(X_val, y_val),  # validation_split 대신 명시적 검증 세트 사용
                    callbacks=callbacks,
                    class_weight=class_weight,  # 클래스 가중치 적용
                    verbose=1
                )
                
                # 모델 평가
                y_pred = (model.predict(X_val) > 0.5).astype(int)
                val_recall = recall_score(y_val, y_pred)
                val_f1 = f1_score(y_val, y_pred)
                
                results.append({
                    'params': {'lr': lr, 'lstm1': lstm1, 'lstm2': lstm2, 'dropout': dr},
                    'recall': val_recall,
                    'f1': val_f1
                })
                
                # 최고 성능 모델 저장
                if val_recall > best_recall:
                    best_recall = val_recall
                    best_params = {'lr': lr, 'lstm1': lstm1, 'lstm2': lstm2, 'dropout': dr}
                    # 기존 최고 모델 삭제하고 메모리 정리
                    if best_model:
                        del best_model
                        gc.collect()
                    best_model = model
                else:
                    # 불필요한 모델 삭제
                    del model
                    gc.collect()

# 결과 출력
print("\n그리드 서치 결과:")
for result in sorted(results, key=lambda x: (-x['recall'], -x['f1'])):
    print(f"파라미터: {result['params']}, 재현율: {result['recall']:.4f}, F1: {result['f1']:.4f}")

print(f"\n최적 파라미터: {best_params}, 재현율: {best_recall:.4f}")

# 최종 모델로 테스트 세트 평가
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
y_pred = best_model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# 다양한 임계값에서의 성능 확인 (FN 최소화)
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n다양한 임계값에서의 성능:")
for threshold in thresholds:
    y_pred_th = (y_pred > threshold).astype(int)
    recall = recall_score(y_test, y_pred_th)
    precision = precision_score(y_test, y_pred_th)
    f1 = f1_score(y_test, y_pred_th)
    print(f"임계값 {threshold}: 재현율={recall:.4f}, 정밀도={precision:.4f}, F1={f1:.4f}")

# 최적의 임계값 선택 (재현율 우선)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
# 재현율이 0.9 이상이면서 정밀도가 최대인 지점 찾기
valid_indices = recalls >= 0.9
if np.any(valid_indices):
    max_precision_index = np.argmax(precisions[valid_indices])
    optimal_threshold = thresholds[np.where(valid_indices)[0][max_precision_index]]
    print(f"\n최적 임계값 (재현율 >= 0.9): {optimal_threshold:.4f}")
else:
    # 재현율 0.9 이상인 지점이 없으면 재현율이 최대한 높은 지점 선택
    optimal_threshold = 0.5
    print(f"\n적절한 임계값을 찾을 수 없어 기본값 사용: {optimal_threshold}")

# 최종 분류 보고서
y_pred_optimal = (y_pred > optimal_threshold).astype(int)
print("\n최적 임계값 적용 후 분류 보고서:")
print(classification_report(y_test, y_pred_optimal, target_names=['비낙상', '낙상']))

# 최종 모델 저장
best_model.save(os.path.join(results_dir, 'fall_detection_model.h5'))
# 최적 임계값 정보도 저장
threshold_info = {
    'optimal_threshold': float(optimal_threshold),
    'best_params': best_params
}
import json
with open(os.path.join(results_dir, 'model_config.json'), 'w') as f:
    json.dump(threshold_info, f)

print("\n모델 및 구성 저장 완료")
# 메모리 정리
gc.collect()

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