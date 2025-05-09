"""
주의사항:
1. scikit-learn(sklearn) 라이브러리가 설치되어 있어야 합니다.
   - 라즈베리파이에 설치: 'pip install scikit-learn'
2. TFLite 모델 파일('models/fall_detection_method1.tflite')이 올바른 경로에 있어야 합니다.
3. 스케일러 파일들이 'scalers' 디렉토리에 존재해야 합니다.
   - 각 센서 특성(AccX, AccY, AccZ, GyrX, GyrY, GyrZ)마다 standard_scaler.pkl과 minmax_scaler.pkl 파일 필요
4. scipy 라이브러리가 설치되어 있어야 합니다.
   - 라즈베리파이에 설치: 'pip install scipy'

코드 개요:
이 코드는 라즈베리파이에서 MPU6050 센서를 이용한 실시간 낙상 감지 시스템을 구현합니다.
- 20Hz로 센서 데이터를 수집하고 푸리에 변환(FFT)을 통해 100Hz 데이터로 업샘플링합니다.
- 업샘플링된 데이터를 사용하여 100Hz에서 학습된 모델과 호환되게 낙상 감지를 수행합니다.
- 동작 흐름:
  1. 센서에서 가속도/자이로 데이터 읽기 (20Hz)
  2. 푸리에 변환을 통한 데이터 업샘플링 (20Hz → 100Hz)
  3. 데이터 정규화 (MinMax → Standard 스케일링)
  4. 150 프레임 윈도우로 데이터 수집
  5. 75 프레임마다 낙상 예측 수행
  6. 낙상 감지시 "NAKSANG" 경보 발생
"""

import time
import numpy as np
# tflite_runtime.interpreter 대신 tensorflow를 직접 사용
import tensorflow as tf
from collections import deque
import signal
import sys
import pickle
import os
from scipy import signal as scipy_signal  # 푸리에 변환을 위한 scipy.signal

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    print("Could not import SMBus2 library. Run 'pip install smbus2'.")
    SENSOR_AVAILABLE = False

# MPU6050 I2C settings
DEV_ADDR = 0x68
PWR_MGMT_1 = 0x6B

# Gyroscope register addresses
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0  # ±250°/s range: 131 LSB/°/s

# Accelerometer register addresses
register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0  # ±2g range: 16384 LSB/g

# Sampling rate settings
INPUT_SAMPLING_RATE = 20   # Hz - 데이터 수집 레이트
OUTPUT_SAMPLING_RATE = 100  # Hz - 모델 입력 레이트(업샘플링 후)

# Model settings
MODEL_PATH = 'models/fall_detection.tflite'
SEQ_LENGTH = 150  # Sequence length 
STRIDE = 75      # Prediction interval (predict every 75 data points)
N_FEATURES = 6   # Number of features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)

# Scalers directory
SCALERS_DIR = 'scalers'

# 푸리에 변환을 위한 버퍼 크기 
# 3초는 절대적인 것이 아니며 실험과 테스트를 통해 최적의 버퍼 크기를 조정하면 됩니다.
# 20Hz로 3초 데이터 = 60 샘플
FFT_BUFFER_SIZE = 60

# Load scaler functions
def load_scalers():
    """
    성능 참고사항:
    - 이 코드(transfrom.py)는 20Hz로 데이터를 수집한 후 FFT 변환으로 100Hz로 업샘플링합니다.
    - 100Hz 직접 샘플링(RaspberryPi.py)보다 I/O 부하가 80% 감소되어 라즈베리파이에 더 적합합니다.
    - FFT 연산은 계산 집약적이지만 3초마다 한번만 수행되므로 지속적인 I/O 부하보다 효율적입니다.
    - 초기 3초 지연은 발생하지만 그 이후 실시간 감지에는 문제가 없습니다.
    
    Load all standard and minmax scalers from pickle files
    """
    scalers = {}
    features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    
    for feature in features:
        std_scaler_path = os.path.join(SCALERS_DIR, f"{feature}_standard_scaler.pkl")
        minmax_scaler_path = os.path.join(SCALERS_DIR, f"{feature}_minmax_scaler.pkl")
        
        try:
            with open(std_scaler_path, 'rb') as f:
                scalers[f"{feature}_standard"] = pickle.load(f)
            with open(minmax_scaler_path, 'rb') as f:
                scalers[f"{feature}_minmax"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading scaler for {feature}: {e}")
    
    return scalers

# MPU6050 sensor class
class MPU6050Sensor:
    def __init__(self, scalers=None):
        """Initialize IMU sensor (MPU6050) and I2C settings"""
        if not SENSOR_AVAILABLE:
            raise ImportError("smbus2 library is not installed.")
        
        self.bus = SMBus(1)  # Use I2C bus 1
        self.setup_mpu6050()
        self.frame_counter = 0
        self.scalers = scalers
        print("MPU6050 sensor initialization complete")
    
    def setup_mpu6050(self):
        """MPU6050 sensor initial setup"""
        # Power management setting - disable sleep mode
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)  # Stabilization time
    
    def read_word(self, reg):
        """Read 16-bit word (2 bytes)"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        value = (high << 8) + low
        return value
    
    def read_word_2c(self, reg):
        """Convert to 2's complement value"""
        val = self.read_word(reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def normalize_data(self, data, feature_names):
        """Standard scale and normalize the sensor data"""
        if self.scalers is None:
            return data  # Return original data if no scalers are provided
        
        normalized_data = []
        for i, feature in enumerate(feature_names):
            # Get value
            value = data[i]
            
            # 1. 먼저 MinMax 스케일링 적용
            if f"{feature}_minmax" in self.scalers:
                scaler = self.scalers[f"{feature}_minmax"]
                value = value * scaler.scale_[0] + scaler.min_[0]
            
            # 2. 그 다음 Standard 스케일링 적용
            if f"{feature}_standard" in self.scalers:
                scaler = self.scalers[f"{feature}_standard"]
                value = (value - scaler.mean_[0]) / scaler.scale_[0]
            
            normalized_data.append(value)
        
        return np.array(normalized_data)
    
    def get_data(self):
        """Read IMU sensor data - all axes of accelerometer and gyroscope"""
        # Accelerometer data (converted to g units)
        accel_x = self.read_word_2c(register_accel_xout_h) / sensitive_accel
        accel_y = self.read_word_2c(register_accel_yout_h) / sensitive_accel
        accel_z = self.read_word_2c(register_accel_zout_h) / sensitive_accel
        
        # Gyroscope data (converted to °/s units)
        gyro_x = self.read_word_2c(register_gyro_xout_h) / sensitive_gyro
        gyro_y = self.read_word_2c(register_gyro_yout_h) / sensitive_gyro
        gyro_z = self.read_word_2c(register_gyro_zout_h) / sensitive_gyro
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Collect raw data
        raw_data = np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        # Normalize data if scalers are provided
        if self.scalers:
            feature_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            return self.normalize_data(raw_data, feature_names)
        
        # Return raw data
        return raw_data

# FFT 기반 리샘플링 클래스
class FFTResampler:
    def __init__(self, input_rate=20, output_rate=100, buffer_size=60):
        """
        FFT 기반 리샘플링 초기화
        
        Args:
            input_rate: 입력 샘플링 레이트 (Hz)
            output_rate: 출력 샘플링 레이트 (Hz)
            buffer_size: FFT 적용을 위한 버퍼 크기
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.buffer_size = buffer_size
        
        # 각 센서 축별 데이터 버퍼
        self.buffers = {
            'AccX': deque(maxlen=buffer_size),
            'AccY': deque(maxlen=buffer_size),
            'AccZ': deque(maxlen=buffer_size),
            'GyrX': deque(maxlen=buffer_size),
            'GyrY': deque(maxlen=buffer_size),
            'GyrZ': deque(maxlen=buffer_size)
        }
        
        # 리샘플링 결과 저장 큐 (output_rate/input_rate 배수만큼 데이터 생성)
        self.resampling_factor = output_rate / input_rate
        self.resampled_queue = deque(maxlen=int(buffer_size * self.resampling_factor))
        
        print(f"FFT Resampler initialized: {input_rate}Hz → {output_rate}Hz")
    
    def add_sample(self, data):
        """
        새로운 샘플 추가 (6축 센서 데이터)
        
        Args:
            data: [AccX, AccY, AccZ, GyrX, GyrY, GyrZ] 값이 담긴 배열
        """
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        # 각 센서 축별로 버퍼에 데이터 추가
        for i, feature in enumerate(features):
            self.buffers[feature].append(data[i])
        
        # 버퍼가 가득 찼을 때 리샘플링 수행
        if len(self.buffers['AccX']) == self.buffer_size:
            self._perform_resampling()
    
    def _perform_resampling(self):
        """
        FFT 기반 리샘플링 수행
        """
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        output_size = int(self.buffer_size * self.resampling_factor)
        resampled_data = []
        
        for feature in features:
            # 버퍼 데이터를 배열로 변환
            buffer_data = np.array(list(self.buffers[feature]))
            
            # scipy.signal.resample 사용하여 FFT 기반 리샘플링
            resampled = scipy_signal.resample(buffer_data, output_size)
            
            # 리샘플링된 데이터 저장
            resampled_data.append(resampled)
        
        # 리샘플링된 모든 축의 데이터를 시간 순서대로 큐에 저장
        # 출력: [(t1에서의 6축 데이터), (t2에서의 6축 데이터), ...] 형태
        for i in range(output_size):
            self.resampled_queue.append(np.array([resampled_data[j][i] for j in range(len(features))]))
    
    def get_resampled_data(self):
        """
        리샘플링된 데이터 반환
        
        Returns:
            리샘플링된 데이터 큐에서 가장 오래된 샘플 하나를 반환
            큐가 비어있으면 None 반환
        """
        if len(self.resampled_queue) > 0:
            return self.resampled_queue.popleft()
        return None
    
    def has_resampled_data(self):
        """리샘플링된 데이터가 있는지 확인"""
        return len(self.resampled_queue) > 0

# Fall detector class
class FallDetector:
    def __init__(self, model_path, seq_length=150, stride=75, n_features=6):
        """Initialize fall detection model"""
        self.seq_length = seq_length
        self.stride = stride
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        self.alarm_active = False
        self.data_counter = 0  # Data counter
        
        # Load TFLite model
        self.interpreter = self.load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Model loading complete")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def load_model(self, model_path):
        """Load TFLite model"""
        try:
            # TensorFlow를 직접 사용하여 모델 로드
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def add_data_point(self, data_array):
        """Add new data point to the data buffer"""
        self.data_buffer.append(data_array)
        self.data_counter += 1
    
    def should_predict(self):
        """Check if prediction should be performed (based on stride interval)"""
        # Only predict when buffer is full and data counter is a multiple of stride
        return len(self.data_buffer) == self.seq_length and self.data_counter % self.stride == 0
    
    def predict(self):
        """Perform fall prediction"""
        try:
            if len(self.data_buffer) < self.seq_length:
                return None  # Not enough data
            
            # Extract data from buffer and convert to array
            data = np.array(list(self.data_buffer))
            
            # Adjust data shape (add batch dimension)
            input_data = np.expand_dims(data, axis=0).astype(np.float32)
            
            # Set model input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get results
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process output based on shape
            if output_data.size == 1:
                # Single value output
                fall_prob = float(output_data.flatten()[0])
            else:
                # Multi-dimensional output
                fall_prob = float(output_data[0][0])
            
            # Prediction result (0: normal, 1: fall)
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {
                'prediction': int(prediction),
                'fall_probability': float(fall_prob)
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def trigger_alarm(self):
        """Display NAKSANG when fall is detected"""
        if not self.alarm_active:
            self.alarm_active = True
            print("\n" + "-" * 30)
            print("!!!!!!! NAKSANG !!!!!!!")
            print("-" * 30 + "\n")
    
    def stop_alarm(self):
        """Stop alarm"""
        if self.alarm_active:
            self.alarm_active = False
            print("Alarm stopped")

def main():
    """Main function"""
    print("Fall detection system starting (20Hz with FFT upsampling)")
    
    try:
        # Load scalers
        print("Loading scalers...")
        scalers = load_scalers()
        print(f"Loaded {len(scalers)} scalers")
        
        # Initialize sensor
        try:
            sensor = MPU6050Sensor(scalers=scalers)
        except Exception as e:
            print(f"Sensor initialization failed: {e}")
            print("Terminating program.")
            return
        
        # Initialize fall detector
        detector = FallDetector(
            model_path=MODEL_PATH,
            seq_length=SEQ_LENGTH,
            stride=STRIDE,
            n_features=N_FEATURES
        )
        
        # Initialize FFT resampler
        resampler = FFTResampler(
            input_rate=INPUT_SAMPLING_RATE,
            output_rate=OUTPUT_SAMPLING_RATE,
            buffer_size=FFT_BUFFER_SIZE
        )
        
        # Ctrl+C signal handler
        def signal_handler(sig, frame):
            print("\nProgram terminated")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Fall detection loop
        print("Collecting sensor data...")
        
        # 초기 데이터 수집 (FFT 버퍼 채우기)
        print(f"Filling initial FFT buffer ({FFT_BUFFER_SIZE} samples at {INPUT_SAMPLING_RATE}Hz)...")
        for _ in range(FFT_BUFFER_SIZE):
            data = sensor.get_data()
            resampler.add_sample(data)
            time.sleep(1.0 / INPUT_SAMPLING_RATE)  # 20Hz 샘플링
        
        # 리샘플링된 데이터로 초기 검출 버퍼 채우기
        print(f"Filling initial detection buffer ({SEQ_LENGTH} samples)...")
        buffer_count = 0
        while buffer_count < SEQ_LENGTH:
            # FFT 리샘플링된 데이터 처리
            if resampler.has_resampled_data():
                resampled_data = resampler.get_resampled_data()
                detector.add_data_point(resampled_data)
                buffer_count += 1
            
            # FFT 버퍼가 비어있으면 더 많은 데이터 수집
            if not resampler.has_resampled_data():
                data = sensor.get_data()
                resampler.add_sample(data)
                time.sleep(1.0 / INPUT_SAMPLING_RATE)  # 20Hz 샘플링
        
        print("Real-time fall detection started")
        
        # Main detection loop
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # 20Hz로 센서 데이터 읽기
            data = sensor.get_data()
            
            # 리샘플러에 데이터 추가
            resampler.add_sample(data)
            
            # Debug output (once per second)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Acceleration(g): X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                print(f"Gyroscope(°/s): X={data[3]:.2f}, Y={data[4]:.2f}, Z={data[5]:.2f}")
                last_time = current_time
            
            # 리샘플링된 데이터 처리 (100Hz)
            while resampler.has_resampled_data():
                resampled_data = resampler.get_resampled_data()
                detector.add_data_point(resampled_data)
                
                # 낙상 예측
                if detector.should_predict():
                    result = detector.predict()
                    
                    # 낙상 감지시 경보
                    if result and result['prediction'] == 1:
                        print(f"Fall detected! Probability: {result['fall_probability']:.2%}")
                        detector.trigger_alarm()
                        alarm_start_time = current_time
            
            # 3초 후 경보 해제
            if detector.alarm_active and (current_time - alarm_start_time >= 3.0):
                detector.stop_alarm()
            
            # 20Hz 샘플링 유지
            sleep_time = 1.0 / INPUT_SAMPLING_RATE - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
