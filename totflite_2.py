"""
TensorFlow 모델을 TensorFlow Lite 형식으로 변환하는 스크립트

이 스크립트는 다음 기능을 수행합니다:
1. 저장된 Keras 모델(.h5) 로드
2. TFLite 형식으로 변환 (기본 변환 및 양자화 옵션 제공)
3. 변환된 모델 저장
4. 간단한 추론 테스트를 통한 검증

사용법:
- 기본 변환: python totflite_2.py
- 양자화 적용: python totflite_2.py --quantize
"""

import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model

def convert_to_tflite(model_path, output_dir, quantize=False):
    """
    Keras 모델을 TFLite 형식으로 변환
    
    Args:
        model_path: Keras 모델 파일 경로(.h5)
        output_dir: 출력 디렉토리
        quantize: 양자화 적용 여부
    
    Returns:
        변환된 TFLite 모델 파일 경로
    """
    print(f"모델 로드 중: {model_path}")
    model = load_model(model_path)
    
    # TFLite 변환 객체 생성
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 변환 옵션 설정
    if quantize:
        print("양자화 적용 중 (int8 변환)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # 대표 데이터셋 생성 (양자화를 위한 캘리브레이션 데이터)
        def representative_dataset():
            # 임의의 입력 데이터 생성 (실제 애플리케이션에서는 실제 데이터 사용 권장)
            # 모델 입력 형태: (배치, 시퀀스 길이, 특성 수)
            for _ in range(100):
                data = np.random.random((1, 150, 6)).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
    
    # 모델 변환
    print("TFLite 모델로 변환 중...")
    tflite_model = converter.convert()
    
    # 출력 파일 경로 설정
    if quantize:
        tflite_filename = "fall_detection_quantized.tflite"
    else:
        tflite_filename = "fall_detection.tflite"
    
    tflite_model_path = os.path.join(output_dir, tflite_filename)
    
    # 모델 저장
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite 모델 저장 완료: {tflite_model_path}")
    model_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
    print(f"모델 크기: {model_size_mb:.2f} MB")
    
    return tflite_model_path

def test_tflite_model(tflite_model_path, test_data_path):
    """
    변환된 TFLite 모델의 추론 테스트
    
    Args:
        tflite_model_path: TFLite 모델 파일 경로
        test_data_path: 테스트 데이터 디렉토리
    """
    print("\n모델 추론 테스트 중...")
    
    # 테스트 데이터 로드
    try:
        X_test = np.load(os.path.join(test_data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(test_data_path, 'y_test.npy'))
        
        # 테스트용으로 일부 데이터만 사용
        test_samples = min(10, len(X_test))
        X_test_samples = X_test[:test_samples]
        y_test_samples = y_test[:test_samples]
        
        # TFLite 인터프리터 로드
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # 입력 및 출력 텐서 정보 가져오기
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"입력 텐서 정보: {input_details}")
        print(f"출력 텐서 정보: {output_details}")
        
        # 추론 실행
        correct_count = 0
        for i in range(test_samples):
            # 입력 데이터 준비
            input_data = X_test_samples[i:i+1].astype(np.float32)
            
            # 입력 텐서에 데이터 설정
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # 추론 실행
            interpreter.invoke()
            
            # 출력 텐서에서 결과 가져오기
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # 예측 클래스
            predicted_class = 1 if output_data[0][0] > 0.5 else 0
            actual_class = y_test_samples[i]
            
            if predicted_class == actual_class:
                correct_count += 1
                
            print(f"샘플 {i+1}: 실제={actual_class}, 예측={predicted_class}, 확률={output_data[0][0]:.4f}")
        
        accuracy = correct_count / test_samples
        print(f"\n테스트 정확도: {accuracy:.2%} ({correct_count}/{test_samples})")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        print("테스트 데이터를 찾을 수 없거나 모델 호환성 문제가 있을 수 있습니다.")

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Keras 모델을 TFLite로 변환')
    parser.add_argument('--quantize', action='store_true', help='int8 양자화 적용')
    parser.add_argument('--model_path', type=str, help='Keras 모델 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--data_path', type=str, help='테스트 데이터 경로')
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    base_path = args.model_path or "/content/drive/MyDrive/Digital_Smart_Final"
    models_dir = os.path.join(base_path, 'models/models_4')
    results_dir = os.path.join(models_dir, 'results')
    
    # 모델 및 데이터 경로 설정
    model_path = args.model_path or os.path.join(results_dir, 'fall_detection_model.h5')
    output_dir = args.output_dir or os.path.join(results_dir, 'tflite_models')
    data_path = args.data_path or os.path.join(base_path, 'sam_100_win_150_str_75')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 변환
    tflite_model_path = convert_to_tflite(model_path, output_dir, quantize=args.quantize)
    
    # 변환된 모델 테스트
    test_tflite_model(tflite_model_path, data_path)
    
    # 라즈베리파이용 복사 안내
    print("\n=== 라즈베리파이 사용 안내 ===")
    print(f"1. 변환된 TFLite 모델({os.path.basename(tflite_model_path)})을 라즈베리파이의 'src/' 디렉토리로 복사하세요.")
    print("2. RaspberryPi.py 또는 transfrom.py의 MODEL_PATH 변수를 업데이트하세요.")
    print("   MODEL_PATH = 'src/fall_detection.tflite' 또는")
    print("   MODEL_PATH = 'src/fall_detection_quantized.tflite'")

if __name__ == "__main__":
    main() 