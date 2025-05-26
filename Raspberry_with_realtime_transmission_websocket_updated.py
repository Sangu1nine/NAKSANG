"""
=============================================================================
파일명: Raspberry_with_realtime_transmission_websocket_updated.py
설명: MPU6050 센서를 이용한 실시간 낙상 감지 및 WebSocket 데이터 전송 시스템
     (TIMESTAMPTZ 및 Asia/Seoul 시간대 지원)

이 시스템은 라즈베리파이에서 MPU6050 센서의 가속도계와 자이로스코프 데이터를
실시간으로 수집하여 낙상을 감지하고, 감지된 데이터를 WebSocket을 통해 서버로 전송합니다.

주요 기능:
- MPU6050 센서 데이터 실시간 수집 (100Hz)
- 데이터 정규화 및 전처리
- TensorFlow Lite 모델을 사용한 낙상 감지
- WebSocket을 통한 실시간 센서 데이터 전송
- 낙상 감지 시 알람 및 이벤트 전송
- TIMESTAMPTZ (Asia/Seoul) 시간대 지원

개발자: NAKSANG 프로젝트팀
버전: 2.1 (TIMESTAMPTZ + Asia/Seoul 지원)
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
import json
import threading
import asyncio
import websockets
from datetime import datetime, timezone, timedelta
import uuid

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
STRIDE = 5      # Prediction interval (predict every 25 data points)
N_FEATURES = 6   # Number of features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
SAMPLING_RATE = 100  # Hz - sampling rate is set to 100Hz

# WebSocket 통신 설정
WEBSOCKET_SERVER_IP = '192.168.0.177'  # 로컬 PC의 IP 주소 (변경 필요)
WEBSOCKET_SERVER_PORT = 8000  # 통신 포트
USER_ID = "raspberry_pi_01"  # 라즈베리파이 고유 사용자 ID (변경 가능)

# Scalers directory
SCALERS_DIR = 'scalers'

# 시간대 설정 (Asia/Seoul)
KST = timezone(timedelta(hours=9))  # Korea Standard Time

# 데이터 전송 관련 변수
websocket_client = None
websocket_connected = False
send_data_queue = []
data_queue_lock = threading.Lock()

def get_current_timestamp():
    """현재 시간을 Asia/Seoul 시간대의 ISO 8601 형식으로 반환"""
    return datetime.now(KST).isoformat()

def unix_to_kst_iso(unix_timestamp):
    """Unix timestamp를 Asia/Seoul 시간대의 ISO 8601 형식으로 변환"""
    dt = datetime.fromtimestamp(unix_timestamp, tz=KST)
    return dt.isoformat()

# WebSocket 연결 URL 생성
def get_websocket_url():
    """WebSocket 연결 URL 생성"""
    return f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"

# WebSocket 연결 및 데이터 전송 (비동기) - 자동 재연결 기능 추가
async def websocket_handler():
    """WebSocket 연결 및 데이터 전송 처리 (자동 재연결 지원)"""
    global websocket_client, websocket_connected, send_data_queue
    
    ws_url = get_websocket_url()
    reconnect_delay = 1  # 재연결 대기 시간 (초)
    max_reconnect_delay = 30  # 최대 재연결 대기 시간
    
    while True:  # 무한 재연결 루프
        try:
            print(f"Attempting WebSocket connection: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                websocket_connected = True
                websocket_client = websocket
                reconnect_delay = 1  # 연결 성공 시 재연결 대기 시간 초기화
                print(f"✅ WebSocket connection successful: {ws_url}")
                
                # 데이터 전송 루프
                while websocket_connected:
                    with data_queue_lock:
                        if len(send_data_queue) > 0:
                            # 🔧 핵심 수정: 낙상 데이터 우선 처리
                            fall_data_index = None
                            for i, item in enumerate(send_data_queue):
                                if item.get('type') == 'fall_detection':
                                    fall_data_index = i
                                    break
                            
                            if fall_data_index is not None:
                                data_package = send_data_queue.pop(fall_data_index)
                                print(f"🚨 낙상 데이터 우선 전송 시작!")
                            else:
                                data_package = send_data_queue.pop(0)
                            
                            try:
                                # 낙상 데이터인 경우 전송 전 상세 로깅
                                if data_package.get('type') == 'fall_detection':
                                    print(f"🚨 낙상 데이터 전송 시작:")
                                    print(f"🚨 - 사용자 ID: {data_package['data'].get('user_id')}")
                                    print(f"🚨 - 신뢰도: {data_package['data'].get('confidence_score', 0):.2%}")
                                    print(f"🚨 - 데이터 크기: {len(str(data_package))} bytes")
                                
                                # JSON 형식으로 변환하여 전송
                                data_json = json.dumps(data_package, ensure_ascii=False)
                                await websocket.send(data_json)
                                
                                # 낙상 데이터인 경우 특별 로깅
                                if data_package.get('type') == 'fall_detection':
                                    print(f"🚨 Fall data transmission successful! Confidence: {data_package['data'].get('confidence_score', 0):.2%}")
                                    print(f"🚨 전송 완료 시간: {get_current_timestamp()}")
                                    
                            except Exception as e:
                                print(f"❌ Data transmission error: {str(e)}")
                                print(f"❌ 에러 타입: {type(e).__name__}")
                                
                                # 낙상 데이터 전송 실패 시 특별 처리
                                if data_package.get('type') == 'fall_detection':
                                    print(f"🚨 낙상 데이터 전송 실패! 재시도 큐에 추가")
                                    print(f"🚨 실패한 데이터: {data_package['data'].get('user_id')} - {data_package['data'].get('confidence_score', 0):.2%}")
                                
                                # 전송 실패한 데이터를 다시 큐에 추가 (우선순위)
                                with data_queue_lock:
                                    send_data_queue.insert(0, data_package)
                                break
                    
                    await asyncio.sleep(0.001)  # 짧은 대기
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"⚠️ WebSocket connection closed: {e}")
        except websockets.exceptions.InvalidURI as e:
            print(f"❌ Invalid WebSocket URI: {e}")
            break  # URI 오류는 재연결해도 해결되지 않음
        except Exception as e:
            print(f"❌ WebSocket connection failed: {str(e)}")
        finally:
            websocket_connected = False
            websocket_client = None
        
        # 재연결 대기
        print(f"🔄 Reconnection attempt in {reconnect_delay} seconds...")
        await asyncio.sleep(reconnect_delay)
        
        # 지수 백오프: 재연결 대기 시간을 점진적으로 증가
        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# WebSocket 클라이언트 시작 (별도 스레드에서 실행) - 개선된 버전
def start_websocket_client():
    """WebSocket 클라이언트를 새 이벤트 루프에서 시작 (자동 재연결 지원)"""
    try:
        # 새 이벤트 루프 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("🌐 WebSocket client started (auto reconnection enabled)")
        
        # WebSocket 연결 시도 (무한 재연결)
        loop.run_until_complete(websocket_handler())
    except Exception as e:
        print(f"❌ WebSocket thread critical error: {str(e)}")
    finally:
        try:
            loop.close()
        except:
            pass

# 데이터 큐에 추가하는 함수 - 개선된 버전
def add_data_to_queue(data_package):
    """데이터를 전송 큐에 안전하게 추가 (낙상 데이터는 연결 상태 무관)"""
    global send_data_queue
    
    with data_queue_lock:
        # 🔧 핵심 수정: 낙상 데이터는 연결 상태와 관계없이 항상 큐에 추가
        if data_package.get('type') == 'fall_detection':
            send_data_queue.insert(0, data_package)
            print(f"🚨 낙상 데이터 큐 추가 완료! (연결상태무관) 대기열: {len(send_data_queue)}개")
        elif websocket_connected:  # IMU 데이터는 연결된 경우에만
            send_data_queue.append(data_package)
        # else: 연결이 끊어진 경우 IMU 데이터는 추가하지 않음
        
        # 큐 크기 제한 (메모리 보호) - 낙상 데이터는 보호
        while len(send_data_queue) > 1000:
            # 가장 오래된 IMU 데이터부터 제거 (낙상 데이터는 보호)
            for i in range(len(send_data_queue) - 1, -1, -1):
                if send_data_queue[i].get('type') != 'fall_detection':
                    send_data_queue.pop(i)
                    break
            else:
                # 모든 데이터가 낙상 데이터인 경우 (매우 드문 경우)
                break

# IMU 센서 데이터 패키징 함수
def create_imu_data_package(sensor_data, user_id):
    """IMU 센서 데이터를 데이터베이스 스키마에 맞게 패키징"""
    return {
        'type': 'imu_data',
        'data': {
            'user_id': user_id,
            'timestamp': get_current_timestamp(),
            'acc_x': float(sensor_data[0]),
            'acc_y': float(sensor_data[1]),
            'acc_z': float(sensor_data[2]),
            'gyr_x': float(sensor_data[3]),
            'gyr_y': float(sensor_data[4]),
            'gyr_z': float(sensor_data[5])
        }
    }

# 낙상 감지 데이터 패키징 함수
def create_fall_data_package(user_id, fall_probability, sensor_data_snapshot):
    """낙상 감지 데이터를 데이터베이스 스키마에 맞게 패키징"""
    return {
        'type': 'fall_detection',
        'data': {
            'user_id': user_id,
            'timestamp': get_current_timestamp(),
            'fall_detected': True,
            'confidence_score': float(fall_probability),
            'sensor_data': {
                'acceleration': {
                    'x': float(sensor_data_snapshot[0]),
                    'y': float(sensor_data_snapshot[1]),
                    'z': float(sensor_data_snapshot[2])
                },
                'gyroscope': {
                    'x': float(sensor_data_snapshot[3]),
                    'y': float(sensor_data_snapshot[4]),
                    'z': float(sensor_data_snapshot[5])
                },
                'timestamp': get_current_timestamp()
            }
        }
    }

# WebSocket 연결 종료 함수
def close_websocket():
    """WebSocket 연결 종료"""
    global websocket_connected
    websocket_connected = False
    print("WebSocket connection closed")

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
        """IMU 센서(MPU6050) 및 I2C 설정 초기화"""
        if not SENSOR_AVAILABLE:
            raise ImportError("smbus2 library is not installed.")
        
        self.bus = SMBus(1)  # I2C 버스 1 사용
        self.setup_mpu6050()
        self.frame_counter = 0
        self.scalers = scalers
        print("MPU6050 sensor initialized")
    
    def setup_mpu6050(self):
        """MPU6050 센서 초기 설정"""
        # 전원 관리 설정 - 슬립 모드 비활성화
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)  # 안정화 시간
    
    def read_word(self, reg):
        """16비트 워드(2바이트) 읽기"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        value = (high << 8) + low
        return value
    
    def read_word_2c(self, reg):
        """2의 보수 값으로 변환"""
        val = self.read_word(reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def normalize_data(self, data, feature_names):
        """센서 데이터 표준화 및 정규화"""
        if self.scalers is None:
            return data  # 스케일러가 없으면 원본 데이터 반환
        
        normalized_data = []
        for i, feature in enumerate(feature_names):
            # 값 가져오기
            value = data[i]
            
            # 표준 스케일링 적용 (z-score 정규화)
            # z = (x - mean) / std
            if f"{feature}_standard" in self.scalers:
                scaler = self.scalers[f"{feature}_standard"]
                value = (value - scaler.mean_[0]) / scaler.scale_[0]
            
            # 최소-최대 스케일링을 [0, 1] 범위로 적용
            # x_norm = (x - min) / (max - min)
            if f"{feature}_minmax" in self.scalers:
                scaler = self.scalers[f"{feature}_minmax"]
                value = value * scaler.scale_[0] + scaler.min_[0]
            
            normalized_data.append(value)
        
        return np.array(normalized_data)
    
    def get_data(self):
        """IMU 센서 데이터 읽기 - 가속도계와 자이로스코프의 모든 축 (물리 단위로 변환)"""
        
        # 원시 가속도계 데이터
        accel_x = self.read_word_2c(register_accel_xout_h)
        accel_y = self.read_word_2c(register_accel_yout_h)
        accel_z = self.read_word_2c(register_accel_zout_h)
        
        # 가속도계 데이터를 g 단위로 변환
        accel_x = accel_x / sensitive_accel
        accel_y = accel_y / sensitive_accel
        accel_z = accel_z / sensitive_accel
        
        # 원시 자이로스코프 데이터
        gyro_x = self.read_word_2c(register_gyro_xout_h)
        gyro_y = self.read_word_2c(register_gyro_yout_h)
        gyro_z = self.read_word_2c(register_gyro_zout_h)
        
        # 자이로스코프 데이터를 도/초 단위로 변환
        gyro_x = gyro_x / sensitive_gyro
        gyro_y = gyro_y / sensitive_gyro
        gyro_z = gyro_z / sensitive_gyro
        
        # 프레임 카운터 증가
        self.frame_counter += 1
        
        # 변환된 데이터 수집
        converted_data = np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        # 스케일러가 제공된 경우 데이터 정규화
        if self.scalers:
            feature_names = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            return self.normalize_data(converted_data, feature_names)
        
        # 변환된 데이터 반환
        return converted_data

# Fall detector class
class FallDetector:
    def __init__(self, model_path, seq_length=50, stride=10, n_features=6):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.stride = stride
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        self.alarm_active = False
        self.data_counter = 0  # 데이터 카운터
        
        # TFLite 모델 로드
        self.interpreter = self.load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Model loading completed")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            raise
    
    def add_data_point(self, data_array):
        """데이터 버퍼에 새로운 데이터 포인트 추가"""
        self.data_buffer.append(data_array)
        self.data_counter += 1
    
    def should_predict(self):
        """예측을 수행해야 하는지 확인 (스트라이드 간격 기반)"""
        # 버퍼가 가득 차고 데이터 카운터가 스트라이드의 배수일 때만 예측
        return len(self.data_buffer) == self.seq_length and self.data_counter % self.stride == 0
    
    def predict(self):
        """낙상 예측 수행"""
        try:
            if len(self.data_buffer) < self.seq_length:
                return None  # 충분한 데이터 없음
            
            # 버퍼에서 데이터 추출하고 배열로 변환
            data = np.array(list(self.data_buffer))
            
            # 데이터 형태 조정 (배치 차원 추가)
            input_data = np.expand_dims(data, axis=0).astype(np.float32)
            
            # 모델 입력 설정
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 추론 실행
            self.interpreter.invoke()
            
            # 결과 가져오기
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 출력 형태에 따른 처리
            if output_data.size == 1:
                # 단일 값 출력
                fall_prob = float(output_data.flatten()[0])
            else:
                # 다차원 출력
                fall_prob = float(output_data[0][0])
            
            # 예측 결과 (0: 정상, 1: 낙상)
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {
                'prediction': int(prediction),
                'fall_probability': float(fall_prob)
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def trigger_alarm(self):
        """낙상 감지 시 NAKSANG 표시"""
        if not self.alarm_active:
            self.alarm_active = True
            print("\n" + "-" * 30)
            print("!!!!!!! NAKSANG !!!!!!!")
            print("-" * 30 + "\n")
    
    def stop_alarm(self):
        """알람 중지"""
        if self.alarm_active:
            self.alarm_active = False
            print("Alarm stopped")

def main():
    """메인 함수"""
    print("Fall detection system started (TIMESTAMPTZ + Asia/Seoul version)")
    print(f"Current time (KST): {get_current_timestamp()}")
    
    try:
        # 스케일러 로드
        print("Loading scalers...")
        scalers = load_scalers()
        print(f"{len(scalers)} scalers loaded")
        
        # 센서 초기화
        try:
            sensor = MPU6050Sensor(scalers=scalers)
        except Exception as e:
            print(f"Sensor initialization failed: {e}")
            print("Program terminated.")
            return
        
        # 낙상 감지기 초기화
        detector = FallDetector(
            model_path=MODEL_PATH,
            seq_length=SEQ_LENGTH,
            stride=STRIDE,
            n_features=N_FEATURES
        )
        
        # Ctrl+C 시그널 핸들러 (개선된 버전)
        def signal_handler(sig, frame):
            print("\nTerminating program...")
            
            # WebSocket 큐에 남은 데이터 전송 대기
            if websocket_connected:
                print("Transmitting remaining data...")
                max_wait_time = 5  # 최대 5초 대기
                wait_start = time.time()
                
                while time.time() - wait_start < max_wait_time:
                    with data_queue_lock:
                        queue_length = len(send_data_queue)
                    
                    if queue_length == 0:
                        print("All data transmission completed")
                        break
                    
                    print(f"Waiting... (remaining data: {queue_length} items)")
                    time.sleep(0.5)
                
                if queue_length > 0:
                    print(f"Warning: {queue_length} data items were not transmitted.")
            
            print("Closing WebSocket connection...")
            close_websocket()
            time.sleep(1)  # 연결 종료 대기
            print("Program terminated")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # WebSocket 클라이언트 시작 (별도 스레드)
        websocket_thread = threading.Thread(target=start_websocket_client)
        websocket_thread.daemon = True
        websocket_thread.start()
        print("WebSocket client thread started")
        
        # 연결 대기
        time.sleep(2)
        
        # 낙상 감지 루프
        print("Collecting sensor data...")
        
        # 초기 데이터 버퍼 채우기
        print(f"Filling initial data buffer ({SEQ_LENGTH} samples)...")
        for _ in range(SEQ_LENGTH):
            data = sensor.get_data()
            detector.add_data_point(data)
            
            # WebSocket으로 IMU 데이터 전송 (연결된 경우)
            if websocket_connected:
                imu_package = create_imu_data_package(data, USER_ID)
                add_data_to_queue(imu_package)
            
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz 샘플링
        
        print("Fall detection started")
        
        # 메인 감지 루프
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # 센서 데이터 읽기
            data = sensor.get_data()
            
            # WebSocket으로 IMU 데이터 전송 (연결된 경우)
            if websocket_connected:
                imu_package = create_imu_data_package(data, USER_ID)
                add_data_to_queue(imu_package)
            
            # 디버그 출력 (1초마다)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Acceleration (g): X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                print(f"Gyroscope (°/s): X={data[3]:.2f}, Y={data[4]:.2f}, Z={data[5]:.2f}")
                print(f"Current KST time: {get_current_timestamp()}")
                if websocket_connected:
                    with data_queue_lock:
                        queue_length = len(send_data_queue)
                    print(f"WebSocket status: Connected (queue length: {queue_length})")
                else:
                    print("WebSocket status: Not connected")
                last_time = current_time
            
            # 데이터 버퍼에 추가
            detector.add_data_point(data)
            
            # 스트라이드 간격에 따른 예측 수행
            if detector.should_predict():
                # 낙상 예측
                result = detector.predict()
                
                # 결과가 존재하고 낙상이 예측된 경우
                if result and result['prediction'] == 1:
                    print(f"🚨 FALL DETECTED! Probability: {result['fall_probability']:.2%}")
                    print(f"🕐 Detection time (KST): {get_current_timestamp()}")
                    detector.trigger_alarm()
                    alarm_start_time = current_time
                    
                    # 🔧 추가: WebSocket 연결 상태 먼저 확인
                    print(f"📡 WebSocket 연결 상태: {websocket_connected}")
                    print(f"📡 현재 전송 큐 길이: {len(send_data_queue)}")
                    
                    # 낙상 감지 데이터 패키징
                    fall_package = create_fall_data_package(
                        USER_ID, 
                        result['fall_probability'], 
                        data
                    )
                    
                    # 🔍 낙상 데이터 상세 로깅
                    print(f"🔍 낙상 데이터 패키지 생성:")
                    print(f"🔍 - 타입: {fall_package.get('type')}")
                    print(f"🔍 - 사용자 ID: {fall_package['data'].get('user_id')}")
                    print(f"🔍 - 신뢰도: {fall_package['data'].get('confidence_score'):.2%}")
                    print(f"🔍 - 타임스탬프: {fall_package['data'].get('timestamp')}")
                    print(f"🔍 - 전체 데이터 크기: {len(str(fall_package))} bytes")
                    
                    # 🔧 추가: 패키지 내용 전체 출력
                    print(f"🔍 - 완전한 패키지 내용:")
                    print(json.dumps(fall_package, ensure_ascii=False, indent=2))
                    
                    # 낙상 데이터 전송 (연결 상태와 관계없이 큐에 추가)
                    print(f"📤 큐에 낙상 데이터 추가 시작...")
                    add_data_to_queue(fall_package)
                    print(f"🚨 Fall detection data added to queue (confidence: {result['fall_probability']:.2%})")
                    
                    # 🔧 추가: 큐 추가 후 상태 확인
                    with data_queue_lock:
                        queue_length_after = len(send_data_queue)
                        fall_data_count = sum(1 for item in send_data_queue if item.get('type') == 'fall_detection')
                    print(f"📤 큐 추가 후 길이: {queue_length_after}")
                    print(f"📤 큐 내 낙상 데이터 개수: {fall_data_count}")
                    
                    # 연결 상태 확인 및 즉시 전송 시도
                    if websocket_connected:
                        print("✅ WebSocket connected - transmission scheduled")
                        print(f"📊 현재 큐 상태:")
                        print(f"📊 - 전체 큐 길이: {queue_length_after}")
                        print(f"📊 - 낙상 데이터 개수: {fall_data_count}")
                        
                        # 🔧 추가: 강제로 즉시 전송 시도
                        print("🔥 강제 즉시 전송 대기 (3초)...")
                        time.sleep(3.0)  # 전송 충분히 대기
                        
                        with data_queue_lock:
                            remaining_queue = len(send_data_queue)
                            remaining_falls = sum(1 for item in send_data_queue if item.get('type') == 'fall_detection')
                        print(f"📊 3초 후 큐 상태:")
                        print(f"📊 - 남은 큐 길이: {remaining_queue}")
                        print(f"📊 - 남은 낙상 데이터: {remaining_falls}")
                        
                        if remaining_falls == 0:
                            print("🎉 낙상 데이터 전송 성공!")
                        else:
                            print("❌ 낙상 데이터가 아직 큐에 남아있음!")
                    else:
                        print("⚠️ WebSocket disconnected - will transmit when reconnected")
                        print("🔄 재연결 시도 중...")
                    
                    print("✅ 낙상 데이터 처리 완료")
            
            # 3초 후 자동으로 알람 끄기
            if detector.alarm_active and (current_time - alarm_start_time >= 3.0):
                detector.stop_alarm()
            
            # 샘플링 레이트 유지
            sleep_time = 1.0 / SAMPLING_RATE - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nProgram terminated")
        close_websocket()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        close_websocket()

if __name__ == "__main__":
    main()