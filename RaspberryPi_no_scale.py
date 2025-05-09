import pickle
import os
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque
import signal
import sys
import time

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

# Model settings
MODEL_PATH = 'fall_detection_method1.tflite'
SEQ_LENGTH = 150  # Sequence length 
STRIDE = 10      # Prediction interval (predict every 10 data points)
N_FEATURES = 9   # Number of features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
SAMPLING_RATE = 100  # Hz - sampling rate is set to 100Hz

# MPU6050 sensor class
class MPU6050Sensor:
    def __init__(self):
        """Initialize IMU sensor (MPU6050) and I2C settings"""
        if not SENSOR_AVAILABLE:
            raise ImportError("smbus2 library is not installed.")
        
        self.bus = SMBus(1)  # Use I2C bus 1
        self.setup_mpu6050()
        self.frame_counter = 0
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
        
        # Euler angle calculation (simplified method)
        accel_xangle = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * 180 / np.pi
        accel_yangle = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
        accel_zangle = np.arctan2(accel_z, np.sqrt(accel_x**2 + accel_y**2)) * 180 / np.pi
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Return all data
        return np.array([
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z,
            accel_xangle, accel_yangle, accel_zangle
        ])

# Fall detector class
class FallDetector:
    def __init__(self, model_path, scalers_path, seq_length=150, stride=10, n_features=9):
        """Initialize fall detection model"""
        self.seq_length = seq_length
        self.stride = stride
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        self.alarm_active = False
        self.data_counter = 0  # Data counter
        
        # 센서 특성 정의 (첫 번째 코드와 일치해야 함)
        self.sensor_features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        # 스케일러 로드
        self.minmax_scalers = {}
        self.standard_scalers = {}
        self.load_scalers(scalers_path)
        
        # Load TFLite model
        self.interpreter = self.load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Model loading complete")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def load_scalers(self, scalers_path):
        """스케일러 로드 함수"""
        try:
            # MinMaxScaler 로드
            for feature in self.sensor_features:
                minmax_path = os.path.join(scalers_path, f"{feature}_minmax_scaler.pkl")
                standard_path = os.path.join(scalers_path, f"{feature}_standard_scaler.pkl")
                
                with open(minmax_path, 'rb') as f:
                    self.minmax_scalers[feature] = pickle.load(f)
                
                with open(standard_path, 'rb') as f:
                    self.standard_scalers[feature] = pickle.load(f)
            
            print("Scalers loaded successfully")
        except Exception as e:
            print(f"Error loading scalers: {str(e)}")
            raise
    
    def apply_scaling(self, data):
        """데이터에 스케일링 적용"""
        # 원본 데이터는 [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, accel_xangle, accel_yangle, accel_zangle]
        # 스케일러는 처음 6개 특성(AccX, AccY, AccZ, GyrX, GyrY, GyrZ)에만 적용
        
        scaled_data = data.copy()
        
        # 처음 6개 특성에 대해 스케일링 적용
        for i, feature in enumerate(self.sensor_features):
            # 데이터가 1D 배열이면 스칼라 값을 2D로 변환
            value = data[i].reshape(-1, 1)
            
            # MinMax 스케일링 적용
            minmax_scaled = self.minmax_scalers[feature].transform(value)
            
            # Standard 스케일링 적용
            standard_scaled = self.standard_scalers[feature].transform(minmax_scaled)
            
            # 결과를 원래 형태로 변환
            scaled_data[i] = standard_scaled.flatten()[0]
        
        return scaled_data
    
    def add_data_point(self, data_array):
        """스케일링을 적용한 데이터 추가"""
        # 스케일링 적용
        scaled_data = self.apply_scaling(data_array)
        
        # 버퍼에 추가
        self.data_buffer.append(scaled_data)
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
    
    # 모델 및 스케일러 경로 설정
    MODEL_PATH = 'fall_detection_method1.tflite'
    SCALERS_PATH = 'preprocessed_data_stride50_0508/scalers'  # 첫 번째 코드에서 저장한, 스케일러 경로
    
    try:
        # Initialize sensor
        try:
            sensor = MPU6050Sensor()
        except Exception as e:
            print(f"Sensor initialization failed: {e}")
            print("Terminating program.")
            return
        
        # Initialize fall detector with scalers path
        detector = FallDetector(
            model_path=MODEL_PATH,
            scalers_path=SCALERS_PATH,
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