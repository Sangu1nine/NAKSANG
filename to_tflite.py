"""
모델 변환 및 테스트 스크립트:
1. models 폴더의 .keras 모델을 TFLite로 변환
2. 변환된 모델을 models 폴더에 저장
3. test_data의 테스트셋으로 정확도 평가
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def convert_to_tflite(model_path, output_path):
    """모델을 TFLite 형식으로 변환"""
    print(f"모델 로드 중: {model_path}")
    model = load_model(model_path)
    
    print("TFLite 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 기본 TFLite 연산만 사용하도록 설정 (호환성 향상)
    # 주석 해제시 SELECT_TF_OPS 사용
    """
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    """
    
    # 변환 실행
    tflite_model = converter.convert()
    
    # 파일 저장
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"모델 저장 완료: {output_path} (크기: {model_size:.2f} MB)")
    
    return output_path

def evaluate_tflite_model(tflite_path, test_data_path):
    """TFLite 모델 정확도 평가"""
    print(f"\n테스트 데이터 로드 중: {test_data_path}")
    
    # 테스트 데이터 로드
    X_test = np.load(os.path.join(test_data_path, "X_test.npy"))
    y_test = np.load(os.path.join(test_data_path, "y_test.npy"))
    
    print(f"테스트 데이터 크기: {X_test.shape}")
    
    # TFLite 모델 로드
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 입출력 텐서 정보
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"입력 텐서 형태: {input_details[0]['shape']}")
    print(f"출력 텐서 형태: {output_details[0]['shape']}")
    
    # 예측 및 정확도 계산
    correct = 0
    total = len(X_test)
    
    for i in range(total):
        # 입력 데이터 설정
        test_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_data)
        
        # 추론 실행
        interpreter.invoke()
        
        # 결과 가져오기
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = 1 if output[0][0] > 0.5 else 0
        
        if pred == y_test[i]:
            correct += 1
        
        # 진행 상황 표시 (10% 단위)
        if (i+1) % (total // 10) == 0 or i == total - 1:
            print(f"진행률: {(i+1)/total*100:.1f}% ({i+1}/{total})")
    
    accuracy = correct / total
    print(f"\n테스트 정확도: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

def main():
    # 모델 경로
    keras_model_path = "models/results/fall_detection_model.keras"
    tflite_model_path = "models/fall_detection.tflite"
    flex_tflite_path = "models/fall_detection_method1.tflite"  # 호환성을 위한 이름
    test_data_dir = "data"
    
    # 모델 변환
    print("===== TFLite 모델 변환 =====")
    
    # TFLite 모델 변환 (Flex 델리게이트 포함)
    print("\n1. Flex 델리게이트 포함 TFLite 모델 생성")
    tflite_path = convert_to_tflite(keras_model_path, flex_tflite_path)
    
    # 모델 평가
    print("\n===== TFLite 모델 평가 =====")
    evaluate_tflite_model(tflite_path, test_data_dir)

if __name__ == "__main__":
    main() 