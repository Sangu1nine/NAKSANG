import time
import numpy as np
import tensorflow as tf  # TensorFlow 전체 import
from collections import deque
import signal
import sys
import pickle
import os

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

# 오프셋 보정을 위한 변수
CALIBRATION_SAMPLES = 100  # 정지 상태에서 수집할 샘플 수
ENABLE_OFFSET_CALIBRATION = True  # 오프셋 보정 활성화 여부

# 이동 평균 필터 설정
MOVING_AVG_WINDOW = 5  # 이동 평균 윈도우 크기
ENABLE_MOVING_AVG = True  # 이동 평균 필터 활성화 여부

# 상보 필터 설정
COMP_FILTER_ALPHA = 0.98  # 상보 필터 알파값 (가속도계:자이로스코프 비율 설정)
ENABLE_COMP_FILTER = False  # 상보 필터 활성화 여부

# 칼만 필터 설정
ENABLE_KALMAN_FILTER = False  # 칼만 필터 활성화 여부

# Model settings
MODEL_PATH = 'models/fall_detection.tflite'
SEQ_LENGTH = 150  # Sequence length 
STRIDE = 75      # Prediction interval (predict every 10 data points)
N_FEATURES = 6   # Number of features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
SAMPLING_RATE = 100  # Hz - sampling rate is set to 100Hz

# Scalers directory
SCALERS_DIR = 'scalers'

# Load scaler functions
def load_scalers():
    """Load all standard and minmax scalers from pickle files"""
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

# 칼만 필터 클래스 구현
class KalmanFilter:
    def __init__(self):
        """칼만 필터 초기화"""
        self.Q_angle = 0.001  # 프로세스 노이즈 공분산
        self.Q_bias = 0.003   # 프로세스 노이즈 공분산
        self.R_measure = 0.03  # 측정 노이즈 공분산
        
        self.angle = 0.0      # 각도
        self.bias = 0.0       # 자이로스코프 바이어스
        
        self.P = np.array([[0.0, 0.0], [0.0, 0.0]])  # 오차 공분산 행렬
        self.P[0][0] = 0.001
        self.P[0][1] = 0.0
        self.P[1][0] = 0.0
        self.P[1][1] = 0.001
    
    def update(self, angle_measured, rate, dt):
        """칼만 필터 업데이트"""
        # 시간 업데이트 방정식
        rate_unbiased = rate - self.bias
        self.angle += dt * rate_unbiased
        
        # 공분산 행렬 업데이트
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt
        
        # 칼만 이득 계산
        S = self.P[0][0] + self.R_measure
        K = np.zeros(2)
        K[0] = self.P[0][0] / S
        K[1] = self.P[1][0] / S
        
        # 측정값으로 상태 업데이트
        y = angle_measured - self.angle
        self.angle += K[0] * y
        self.bias += K[1] * y
        
        # 공분산 행렬 업데이트
        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]
        
        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp
        
        return self.angle

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
        
        # 이동 평균 필터를 위한 데이터 큐 초기화
        self.accel_x_queue = deque(maxlen=MOVING_AVG_WINDOW)
        self.accel_y_queue = deque(maxlen=MOVING_AVG_WINDOW)
        self.accel_z_queue = deque(maxlen=MOVING_AVG_WINDOW)
        self.gyro_x_queue = deque(maxlen=MOVING_AVG_WINDOW)
        self.gyro_y_queue = deque(maxlen=MOVING_AVG_WINDOW)
        self.gyro_z_queue = deque(maxlen=MOVING_AVG_WINDOW)
        
        # 칼만 필터 초기화
        self.kalman_accel_x = KalmanFilter()
        self.kalman_accel_y = KalmanFilter()
        self.kalman_accel_z = KalmanFilter()
        self.kalman_gyro_x = KalmanFilter()
        self.kalman_gyro_y = KalmanFilter()
        self.kalman_gyro_z = KalmanFilter()
        
        # 상보 필터를 위한 변수 초기화
        self.last_time = time.time()
        self.complementary_angle_x = 0
        self.complementary_angle_y = 0
        
        # 오프셋 보정을 위한 변수 초기화
        self.accel_x_offset = 0
        self.accel_y_offset = 0
        self.accel_z_offset = 0
        self.gyro_x_offset = 0
        self.gyro_y_offset = 0
        self.gyro_z_offset = 0
        
        # 초기화 옵션에 따라 오프셋 보정 수행
        if ENABLE_OFFSET_CALIBRATION:
            self.calibrate_offsets()
        
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
            
            # Apply standard scaling (z-score normalization)
            # z = (x - mean) / std
            if f"{feature}_standard" in self.scalers:
                scaler = self.scalers[f"{feature}_standard"]
                value = (value - scaler.mean_[0]) / scaler.scale_[0]
            
            # Apply min-max scaling to [0, 1] range
            # x_norm = (x - min) / (max - min)
            if f"{feature}_minmax" in self.scalers:
                scaler = self.scalers[f"{feature}_minmax"]
                value = value * scaler.scale_[0] + scaler.min_[0]
            
            normalized_data.append(value)
        
        return np.array(normalized_data)
    
    def calibrate_offsets(self):
        """정지 상태에서 샘플을 수집하여 오프셋 보정값 계산"""
        print(f"센서 오프셋 보정 중... ({CALIBRATION_SAMPLES} 샘플 수집)")
        
        accel_x_samples = []
        accel_y_samples = []
        accel_z_samples = []
        gyro_x_samples = []
        gyro_y_samples = []
        gyro_z_samples = []
        
        # 샘플 수집
        for _ in range(CALIBRATION_SAMPLES):
            # 가속도계 데이터
            accel_x = self.read_word_2c(register_accel_xout_h) / sensitive_accel
            accel_y = self.read_word_2c(register_accel_yout_h) / sensitive_accel
            accel_z = self.read_word_2c(register_accel_zout_h) / sensitive_accel
            
            # 자이로스코프 데이터
            gyro_x = self.read_word_2c(register_gyro_xout_h) / sensitive_gyro
            gyro_y = self.read_word_2c(register_gyro_yout_h) / sensitive_gyro
            gyro_z = self.read_word_2c(register_gyro_zout_h) / sensitive_gyro
            
            accel_x_samples.append(accel_x)
            accel_y_samples.append(accel_y)
            accel_z_samples.append(accel_z - 1.0)  # 정지 상태에서 z축은 중력 때문에 약 1g
            gyro_x_samples.append(gyro_x)
            gyro_y_samples.append(gyro_y)
            gyro_z_samples.append(gyro_z)
            
            time.sleep(0.01)  # 샘플링 간격
        
        # 오프셋 계산 (평균값)
        self.accel_x_offset = np.mean(accel_x_samples)
        self.accel_y_offset = np.mean(accel_y_samples)
        self.accel_z_offset = np.mean(accel_z_samples)
        self.gyro_x_offset = np.mean(gyro_x_samples)
        self.gyro_y_offset = np.mean(gyro_y_samples)
        self.gyro_z_offset = np.mean(gyro_z_samples)
        
        print("오프셋 보정값:")
        print(f"가속도: X={self.accel_x_offset:.6f}, Y={self.accel_y_offset:.6f}, Z={self.accel_z_offset:.6f}")
        print(f"자이로: X={self.gyro_x_offset:.6f}, Y={self.gyro_y_offset:.6f}, Z={self.gyro_z_offset:.6f}")
    
    def apply_moving_average(self, value, queue):
        """이동 평균 필터 적용"""
        queue.append(value)
        return np.mean(queue)
    
    def apply_complementary_filter(self, accel_angle, gyro_rate):
        """상보 필터 적용"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 상보 필터 공식: angle = α × (angle + gyro_rate × dt) + (1 - α) × accel_angle
        # α는 일반적으로 0.98 정도로 설정
        angle = COMP_FILTER_ALPHA * (self.complementary_angle_x + gyro_rate * dt) + (1 - COMP_FILTER_ALPHA) * accel_angle
        return angle
    
    def apply_kalman_filter(self, measured_value, rate, kalman_filter):
        """칼만 필터 적용"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        return kalman_filter.update(measured_value, rate, dt)
    
    def get_data(self):
        """Read IMU sensor data - all axes of accelerometer and gyroscope (converted to physical units)"""
        # Raw accelerometer data
        accel_x = self.read_word_2c(register_accel_xout_h)
        accel_y = self.read_word_2c(register_accel_yout_h)
        accel_z = self.read_word_2c(register_accel_zout_h)
        
        # Convert accelerometer data to g units
        accel_x = accel_x / sensitive_accel
        accel_y = accel_y / sensitive_accel
        accel_z = accel_z / sensitive_accel
        
        # Raw gyroscope data
        gyro_x = self.read_word_2c(register_gyro_xout_h)
        gyro_y = self.read_word_2c(register_gyro_yout_h)
        gyro_z = self.read_word_2c(register_gyro_zout_h)
        
        # Convert gyroscope data to degrees per second
        gyro_x = gyro_x / sensitive_gyro
        gyro_y = gyro_y / sensitive_gyro
        gyro_z = gyro_z / sensitive_gyro
        
        # 오프셋 보정 적용
        if ENABLE_OFFSET_CALIBRATION:
            accel_x -= self.accel_x_offset
            accel_y -= self.accel_y_offset
            accel_z -= self.accel_z_offset
            gyro_x -= self.gyro_x_offset
            gyro_y -= self.gyro_y_offset
            gyro_z -= self.gyro_z_offset
        
        # 이동 평균 필터 적용
        if ENABLE_MOVING_AVG:
            accel_x = self.apply_moving_average(accel_x, self.accel_x_queue)
            accel_y = self.apply_moving_average(accel_y, self.accel_y_queue)
            accel_z = self.apply_moving_average(accel_z, self.accel_z_queue)
            gyro_x = self.apply_moving_average(gyro_x, self.gyro_x_queue)
            gyro_y = self.apply_moving_average(gyro_y, self.gyro_y_queue)
            gyro_z = self.apply_moving_average(gyro_z, self.gyro_z_queue)
        
        # 상보 필터 적용 (필요한 경우)
        if ENABLE_COMP_FILTER:
            # 가속도계를 사용한 각도 계산
            accel_angle_x = np.arctan2(accel_y, accel_z) * 180 / np.pi
            accel_angle_y = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
            
            # 상보 필터 적용
            self.complementary_angle_x = self.apply_complementary_filter(accel_angle_x, gyro_x)
            self.complementary_angle_y = self.apply_complementary_filter(accel_angle_y, gyro_y)
            
            # 필터된 값 적용
            gyro_x = self.complementary_angle_x
            gyro_y = self.complementary_angle_y
        
        # 칼만 필터 적용 (필요한 경우)
        if ENABLE_KALMAN_FILTER:
            # 가속도계를 사용한 각도 계산
            accel_angle_x = np.arctan2(accel_y, accel_z) * 180 / np.pi
            accel_angle_y = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
            
            # 칼만 필터 적용
            kalman_angle_x = self.apply_kalman_filter(accel_angle_x, gyro_x, self.kalman_accel_x)
            kalman_angle_y = self.apply_kalman_filter(accel_angle_y, gyro_y, self.kalman_accel_y)
            
            # 필터된 값 적용
            gyro_x = kalman_angle_x
            gyro_y = kalman_angle_y
            
            accel_x = self.apply_kalman_filter(accel_x, 0, self.kalman_accel_x)
            accel_y = self.apply_kalman_filter(accel_y, 0, self.kalman_accel_y)
            accel_z = self.apply_kalman_filter(accel_z, 0, self.kalman_accel_z)
            gyro_z = self.apply_kalman_filter(gyro_z, 0, self.kalman_gyro_z)
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Collect converted data
        converted_data = np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        # Normalize data if scalers are provided
        if self.scalers:
            feature_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            return self.normalize_data(converted_data, feature_names)
        
        # Return converted data
        return converted_data

# Fall detector class
class FallDetector:
    def __init__(self, model_path, seq_length=50, stride=10, n_features=6):
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
    print("Fall detection system starting")
    
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
        
        # Ctrl+C signal handler
        def signal_handler(sig, frame):
            print("\nProgram terminated")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Fall detection loop
        print("Collecting sensor data...")
        
        # Fill initial data buffer
        print(f"Filling initial data buffer ({SEQ_LENGTH} samples)...")
        for _ in range(SEQ_LENGTH):
            data = sensor.get_data()
            detector.add_data_point(data)
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz sampling
        
        print("Real-time fall detection started")
        
        # Main detection loop
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # Read sensor data
            data = sensor.get_data()
            
            # Debug output (once per second)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Acceleration(g): X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                print(f"Gyroscope(°/s): X={data[3]:.2f}, Y={data[4]:.2f}, Z={data[5]:.2f}")
                last_time = current_time
            
            # Add to data buffer
            detector.add_data_point(data)
            
            # Perform prediction based on stride interval
            if detector.should_predict():
                # Fall prediction
                result = detector.predict()
                
                # If result exists and fall is predicted
                if result and result['prediction'] == 1:
                    print(f"Fall detected! Probability: {result['fall_probability']:.2%}")
                    detector.trigger_alarm()
                    alarm_start_time = current_time
            
            # Automatically turn off alarm after 3 seconds
            if detector.alarm_active and (current_time - alarm_start_time >= 3.0):
                detector.stop_alarm()
            
            # Maintain sampling rate
            sleep_time = 1.0 / SAMPLING_RATE - (time.time() - current_time)
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
