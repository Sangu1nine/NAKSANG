"""
Keras 모델(.h5)을 TensorFlow Lite 형식으로 변환하는 스크립트

이 스크립트는:
1. 기본 TFLite 모델 생성
2. 양자화된 TFLite 모델 생성
3. 'src' 디렉토리에 모델 자동 복사
"""

import os
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization

def create_falldetection_model():
    """호환성 문제 시 새 모델 생성"""
    model = Sequential([
        LSTM(32, input_shape=(150, 6), return_sequences=True),
        LayerNormalization(),
        Dropout(0.3),
        
        LSTM(16),
        LayerNormalization(),
        Dropout(0.3),
        
        Dense(8, activation='relu'),
        LayerNormalization(),
        
        Dense(1, activation='sigmoid')
    ])
    return model

def load_keras_model(model_path):
    """다양한 방법으로 모델 로드 시도"""
    print(f"모델 로드 중: {model_path}")
    
    try:
        # 방법 1: 직접 로드
        model = load_model(model_path)
        print("모델 로드 성공")
        return model
    except Exception as e:
        print(f"기본 모델 로드 실패: {e}")
        
        try:
            # 방법 2: 모델 구성 후 가중치만 로드
            model = create_falldetection_model()
            model.load_weights(model_path)
            print("가중치 로드 성공")
            return model
        except Exception as e:
            print(f"가중치 로드 실패: {e}")
            
            # 방법 3: 빈 모델 생성
            print("새 모델 생성")
            return create_falldetection_model()

def convert_to_tflite(model, output_path, quantize=False):
    """모델을 TFLite 형식으로 변환"""
    try:
        # 변환기 생성
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # TF 연산 지원 추가 (오류 해결을 위한 핵심 변경사항)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # 양자화 설정
        if quantize:
            print("양자화 적용 중...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 대표 데이터셋 함수
            def representative_dataset():
                for _ in range(100):
                    data = np.random.random((1, 150, 6)).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
        
        # 변환 실행
        tflite_model = converter.convert()
        
        # 파일 저장
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"모델 저장 완료: {output_path} (크기: {model_size:.2f} MB)")
        return True
    
    except Exception as e:
        print(f"변환 오류: {e}")
        return False

def main():
    # 디렉토리 설정
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, 'models')
    src_dir = os.path.join(base_dir, 'src')
    
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    
    # 경로 설정
    h5_model_path = os.path.join(model_dir, 'fall_detection_model.h5')
    standard_tflite_path = os.path.join(model_dir, 'fall_detection.tflite')
    quantized_tflite_path = os.path.join(model_dir, 'fall_detection_quantized.tflite')
    
    # src 폴더 경로
    src_standard_path = os.path.join(src_dir, 'fall_detection.tflite')
    src_quantized_path = os.path.join(src_dir, 'fall_detection_quantized.tflite')
    
    print("===== TFLite 변환 시작 =====")
    
    # 모델 로드
    model = load_keras_model(h5_model_path)
    
    # 기본 모델 변환
    print("\n1. 기본 TFLite 모델 생성")
    standard_success = convert_to_tflite(model, standard_tflite_path, quantize=False)
    
    # 양자화 모델 변환
    print("\n2. 양자화 TFLite 모델 생성")
    quantized_success = convert_to_tflite(model, quantized_tflite_path, quantize=True)
    
    # src 디렉토리로 파일 복사
    print("\n3. 모델 파일 src 디렉토리로 복사")
    if standard_success:
        shutil.copy2(standard_tflite_path, src_standard_path)
        print(f"기본 모델 복사 완료: {src_standard_path}")
    
    if quantized_success:
        shutil.copy2(quantized_tflite_path, src_quantized_path)
        print(f"양자화 모델 복사 완료: {src_quantized_path}")
    
    print("\n===== 파일 크기 비교 =====")
    if standard_success and quantized_success:
        std_size = os.path.getsize(standard_tflite_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_tflite_path) / (1024 * 1024)
        reduction = (1 - quant_size / std_size) * 100
        
        print(f"기본 모델: {std_size:.2f} MB")
        print(f"양자화 모델: {quant_size:.2f} MB")
        print(f"크기 감소: {reduction:.1f}%")
    
    print("\n===== 라즈베리파이 사용 안내 =====")
    print("RaspberryPi.py 또는 transfrom.py의 MODEL_PATH 변수를 업데이트하세요:")
    print("MODEL_PATH = 'src/fall_detection.tflite' 또는")
    print("MODEL_PATH = 'src/fall_detection_quantized.tflite'")

if __name__ == "__main__":
    main()
