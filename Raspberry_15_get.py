"""
=============================================================================
파일명: Raspberry_with_realtime_transmission.py
설명: MPU6050 센서를 이용한 실시간 낙상 감지 및 WiFi 데이터 전송 시스템

이 시스템은 라즈베리파이에서 MPU6050 센서의 가속도계와 자이로스코프 데이터를
실시간으로 수집하여 낙상을 감지하고, 감지된 데이터를 WiFi를 통해 서버로 전송합니다.

주요 기능:
- MPU6050 센서 데이터 실시간 수집 (100Hz)
- 데이터 정규화 및 전처리
- TensorFlow Lite 모델을 사용한 낙상 감지
- WiFi를 통한 실시간 센서 데이터 전송
- 낙상 감지 시 알람 및 이벤트 전송

개발자: NAKSANG 프로젝트팀
버전: 1.0
=============================================================================
"""

import time
import numpy as np
import tensorflow as tf  # TensorFlow 전체 import
from collections import deque
import signal
import sys
import pickle
import os
import socket
import json
import threading

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
MODEL_PATH = 'models/fall_detection.tflite'
SEQ_LENGTH = 150  # Sequence length 
STRIDE = 15      # Prediction interval (predict every 25 data points)
N_FEATURES = 6   # Number of features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
SAMPLING_RATE = 100  # Hz - sampling rate is set to 100Hz

# WiFi 통신 설정
WIFI_SERVER_IP = '192.168.0.177'  # 로컬 PC의 IP 주소 (변경 필요)
WIFI_SERVER_PORT = 8000  # 통신 포트

# Scalers directory
SCALERS_DIR = 'scalers'

# 데이터 전송 관련 변수
wifi_client = None
wifi_connected = False
send_data_queue = []

# WiFi 연결 함수
def connect_wifi():
    """WiFi 서버에 연결"""
    global wifi_client, wifi_connected
    try:
        wifi_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        wifi_client.connect((WIFI_SERVER_IP, WIFI_SERVER_PORT))
        wifi_connected = True
        print(f"WiFi connection successful: {WIFI_SERVER_IP}:{WIFI_SERVER_PORT}")
        return True
    except Exception as e:
        print(f"WiFi connection failed: {str(e)}")
        wifi_connected = False
        return False

# 데이터 전송 스레드 함수
def send_data_thread():
    """데이터 전송 스레드"""
    global send_data_queue, wifi_client, wifi_connected
    
    while wifi_connected:
        if len(send_data_queue) > 0:
            try:
                # 큐에서 데이터 가져오기
                sensor_data = send_data_queue.pop(0)
                # JSON 형식으로 변환하여 전송
                data_json = json.dumps(sensor_data)
                wifi_client.sendall((data_json + '\n').encode('utf-8'))
            except Exception as e:
                print(f"Data transmission error: {str(e)}")
                wifi_connected = False
                break
        else:
            time.sleep(0.001)  # 큐가 비어있을 때 CPU 사용량 줄이기

# WiFi 연결 종료 함수
def close_wifi():
    """WiFi 연결 종료"""
    global wifi_client, wifi_connected
    if wifi_client:
        try:
            wifi_client.close()
            print("WiFi connection closed")
        except:
            pass
    wifi_connected = False

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
        print("MPU6050 sensor initialized")
    
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
        print("Model loading completed")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def load_model(self, model_path):
        """Load TFLite model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Model loading error: {str(e)}")
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
            print(f"Prediction error: {str(e)}")
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
    print("Fall detection system started")
    
    try:
        # Load scalers
        print("Loading scalers...")
        scalers = load_scalers()
        print(f"{len(scalers)} scalers loaded")
        
        # Initialize sensor
        try:
            sensor = MPU6050Sensor(scalers=scalers)
        except Exception as e:
            print(f"Sensor initialization failed: {e}")
            print("Program ended.")
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
            print("\nProgram ended")
            close_wifi()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # WiFi connection attempt
        wifi_thread = None
        if connect_wifi():
            # Start data transmission thread
            wifi_thread = threading.Thread(target=send_data_thread)
            wifi_thread.daemon = True
            wifi_thread.start()
            print("Started WiFi data transmission thread")
        
        # Fall detection loop
        print("Collecting sensor data...")
        
        # Fill initial data buffer
        print(f"Filling initial data buffer ({SEQ_LENGTH} samples)...")
        for _ in range(SEQ_LENGTH):
            data = sensor.get_data()
            detector.add_data_point(data)
            
            # Send data to WiFi (if connected)
            if wifi_connected:
                sensor_data = {
                    'timestamp': time.time(),
                    'accel': {'x': float(data[0]), 'y': float(data[1]), 'z': float(data[2])},
                    'gyro': {'x': float(data[3]), 'y': float(data[4]), 'z': float(data[5])}
                }
                send_data_queue.append(sensor_data)
            
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz sampling
        
        print("Fall detection started")
        
        # Main detection loop
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # Read sensor data
            data = sensor.get_data()
            
            # Send data to WiFi (if connected)
            if wifi_connected:
                sensor_data = {
                    'timestamp': time.time(),
                    'accel': {'x': float(data[0]), 'y': float(data[1]), 'z': float(data[2])},
                    'gyro': {'x': float(data[3]), 'y': float(data[4]), 'z': float(data[5])}
                }
                send_data_queue.append(sensor_data)
            
            # Debug output (once per second)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Acceleration (g): X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                print(f"Gyroscope (°/s): X={data[3]:.2f}, Y={data[4]:.2f}, Z={data[5]:.2f}")
                if wifi_connected:
                    print(f"WiFi transmission status: Connected (queue length: {len(send_data_queue)})")
                else:
                    print("WiFi transmission status: Not connected")
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
                    
                    # Send fall detection information
                    if wifi_connected:
                        fall_event = {
                            'event': 'fall_detected',
                            'timestamp': current_time,
                            'probability': result['fall_probability']
                        }
                        send_data_queue.append(fall_event)
            
            # Automatically turn off alarm after 3 seconds
            if detector.alarm_active and (current_time - alarm_start_time >= 3.0):
                detector.stop_alarm()
            
            # Maintain sampling rate
            sleep_time = 1.0 / SAMPLING_RATE - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nProgram ended")
        close_wifi()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        close_wifi()

if __name__ == "__main__":
    main() 