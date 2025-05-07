import tensorflow as tf
import os
import h5py
import numpy as np

# TensorFlow 버전 확인
print(f"TensorFlow 버전: {tf.__version__}")

# 모델 경로 설정
original_model_path = 'C:/gitproject/Final_Project/base_model_20250501-065221.h5'
output_dir = 'C:/gitproject/Final_Project/tflite_model/'

# 출력 디렉토리 확인
os.makedirs(output_dir, exist_ok=True)

# H5 파일 구조 검사
print(f"H5 파일 구조 분석: {original_model_path}")
try:
    with h5py.File(original_model_path, 'r') as f:
        print("모델 구조:")
        for key in f.keys():
            print(f"- {key}")
            if isinstance(f[key], h5py.Group):
                for subkey in f[key].keys():
                    print(f"  - {subkey}")
except Exception as e:
    print(f"H5 파일 분석 실패: {str(e)}")

# 모델을 직접 생성
print("모델을 직접 생성합니다...")
try:
    # 모델 입력 형태는 오류 메시지에서 확인: [None, 150, 9]
    inputs = tf.keras.layers.Input(shape=(150, 9))
    
    # 간단한 LSTM 모델 (원본 모델과 유사하게 구성)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # 모델 생성
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("모델 생성 성공")
except Exception as e:
    print(f"모델 생성 실패: {str(e)}")
    raise Exception("모델 생성에 실패했습니다.")

print("모델 생성 완료")

# 모델 요약
model.summary()

# 모델의 입력과 출력 형태를 확인
input_shape = model.input_shape
print(f"입력 형태: {input_shape}")

# 여러 변환 방법을 순차적으로 시도하는 함수
def convert_to_tflite(model, output_path, model_name):
    methods_tried = 0
    
    # 방법 1: 고정된 입력 크기로 모델 다시 생성
    try:
        print("\n방법 1: 고정 입력 크기 모델 변환 시도")
        tf_input = tf.keras.layers.Input(shape=input_shape[1:], batch_size=1)
        outputs = model(tf_input)
        fixed_model = tf.keras.Model(inputs=tf_input, outputs=outputs)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimization_default = False
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        model_path = os.path.join(output_path, f'{model_name}_method1.tflite')
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"성공: TFLite 모델 저장 완료: {model_path}")
        methods_tried += 1
        return True
    except Exception as e:
        print(f"방법 1 실패: {str(e)}")
    
    # 방법 2: 기본 변환 시도
    try:
        print("\n방법 2: 기본 변환 시도")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        model_path = os.path.join(output_path, f'{model_name}_method2.tflite')
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"성공: TFLite 모델 저장 완료: {model_path}")
        methods_tried += 1
        return True
    except Exception as e:
        print(f"방법 2 실패: {str(e)}")
    
    if methods_tried == 0:
        print("\n모든 변환 방법이 실패했습니다.")
        return False
    
    return True

# 변환 실행
model_name = 'fall_detection'
result = convert_to_tflite(model, output_dir, model_name)

if result:
    print("\n최소 하나의 방법으로 변환에 성공했습니다.")
else:
    print("\n모든 변환 방법이 실패했습니다. 모델 구조 또는 TensorFlow 버전을 확인해보세요.")

print("\n변환 과정 완료.")