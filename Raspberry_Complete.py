"""
Optimized Raspberry Pi Fall Detection System
- 라즈베리파이 최적화: 메모리 사용량 감소, CPU 부하 최소화
- 불필요한 디버그 출력 제거, 간소화된 보행 감지
- 핵심 기능만 유지: 낙상 감지 + 상태 기반 데이터 전송
"""

import time
import numpy as np
import tensorflow as tf
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
import queue
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    print("SMBus2 library missing. Install: pip install smbus2")
    SENSOR_AVAILABLE = False

# === 설정 ===
DEV_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_REGISTERS = [0x3B, 0x3D, 0x3F]
GYRO_REGISTERS = [0x43, 0x45, 0x47]
SENSITIVE_ACCEL = 16384.0
SENSITIVE_GYRO = 131.0

MODEL_PATH = 'models/fall_detection.tflite'
SCALERS_DIR = 'scalers'
SEQ_LENGTH = 150
STRIDE = 5
SAMPLING_RATE = 100
SEND_RATE = 10

WEBSOCKET_SERVER_IP = '192.168.0.177'
WEBSOCKET_SERVER_PORT = 8000
USER_ID = "raspberry_pi_01"
KST = timezone(timedelta(hours=9))

class UserState(Enum):
    DAILY = "Idle"
    WALKING = "Walking"
    FALL = "Fall"

class SimpleWalkingDetector:
    """간소화된 보행 감지기 - 라즈베리파이 최적화"""
    def __init__(self):
        # 메모리 최적화: 1초 버퍼만 사용 (100샘플)
        self.buffer_size = 100
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.is_walking = False
        self.walking_count = 0
        self.idle_count = 0
        
        # 간소화된 임계값
        self.acc_std_threshold = 0.15
        self.walking_threshold = 5  # 5회 연속 감지
        self.idle_threshold = 10    # 10회 연속 미감지

    def add_data(self, acc_x, acc_y, acc_z):
        """간소화된 보행 감지"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        self.acc_buffer.append(acc_magnitude)
        
        if len(self.acc_buffer) >= self.buffer_size:
            # 간단한 표준편차 기반 감지
            acc_std = np.std(self.acc_buffer)
            
            if acc_std > self.acc_std_threshold:
                self.walking_count += 1
                self.idle_count = 0
            else:
                self.idle_count += 1
                self.walking_count = 0
            
            # 상태 변경
            if not self.is_walking and self.walking_count >= self.walking_threshold:
                self.is_walking = True
                print("Walking started")
            elif self.is_walking and self.idle_count >= self.idle_threshold:
                self.is_walking = False
                print("Walking stopped")
        
        return self.is_walking

class SimpleStateManager:
    """간소화된 상태 관리자"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = 10.0

    def update_state(self, is_walking, fall_detected):
        current_time = time.time()
        
        # 낙상 감지 (최우선)
        if fall_detected and self._can_detect_fall():
            self.current_state = UserState.FALL
            self.last_fall_time = current_time
            self.state_start_time = current_time
            return True
        
        # 보행 상태 전환
        elif self.current_state == UserState.DAILY and is_walking:
            self.current_state = UserState.WALKING
            self.state_start_time = current_time
            return True
        elif self.current_state == UserState.WALKING and not is_walking:
            if current_time - self.state_start_time > 5.0:  # 5초 후 복귀
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                return True
        
        # 낙상 후 자동 복구
        elif self.current_state == UserState.FALL:
            if current_time - self.state_start_time > 3.0:
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                return True
        
        return False

    def _can_detect_fall(self):
        if self.last_fall_time is None:
            return True
        return time.time() - self.last_fall_time > self.fall_cooldown

    def should_send_data(self):
        return self.current_state != UserState.DAILY

class OptimizedDataSender:
    """최적화된 데이터 전송"""
    def __init__(self):
        self.imu_queue = queue.Queue(maxsize=20)  # 큐 크기 감소
        self.fall_queue = queue.Queue(maxsize=50)
        self.websocket = None
        self.connected = False

    def add_imu_data(self, data):
        try:
            self.imu_queue.put_nowait(data)
        except queue.Full:
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(data)
            except queue.Empty:
                pass

    def add_fall_data(self, data):
        try:
            self.fall_queue.put_nowait(data)
        except queue.Full:
            pass

    async def send_loop(self):
        while True:
            try:
                # 낙상 데이터 우선
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data)
                # IMU 데이터
                elif self.connected and not self.imu_queue.empty():
                    imu_data = self.imu_queue.get_nowait()
                    await self._send_data(imu_data)
                
                await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(1)

    async def _send_data(self, data):
        if not self.websocket:
            return
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            await self.websocket.send(json_data)
        except Exception:
            pass

class OptimizedSensor:
    """최적화된 센서 클래스"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("Sensor library missing.")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        self.scalers = self._load_scalers()

    def _load_scalers(self):
        scalers = {}
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        for feature in features:
            try:
                std_path = os.path.join(SCALERS_DIR, f"{feature}_standard_scaler.pkl")
                minmax_path = os.path.join(SCALERS_DIR, f"{feature}_minmax_scaler.pkl")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(std_path, 'rb') as f:
                        scalers[f"{feature}_standard"] = pickle.load(f)
                    with open(minmax_path, 'rb') as f:
                        scalers[f"{feature}_minmax"] = pickle.load(f)
            except Exception:
                pass
        
        return scalers

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val

    def get_data(self):
        raw_data = []
        for reg in ACCEL_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_ACCEL)
        for reg in GYRO_REGISTERS:
            raw_data.append(self._read_word_2c(reg) / SENSITIVE_GYRO)

        if self.scalers:
            features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
            normalized = []
            for i, feature in enumerate(features):
                val = raw_data[i]
                
                if f"{feature}_standard" in self.scalers:
                    scaler = self.scalers[f"{feature}_standard"]
                    val = (val - scaler.mean_[0]) / scaler.scale_[0]
                
                if f"{feature}_minmax" in self.scalers:
                    scaler = self.scalers[f"{feature}_minmax"]
                    val = val * scaler.scale_[0] + scaler.min_[0]
                
                normalized.append(val)
            return np.array(normalized)
        
        return np.array(raw_data)

class OptimizedFallDetector:
    """최적화된 낙상 감지기"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def add_data(self, data):
        self.buffer.append(data)
        self.counter += 1

    def should_predict(self):
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0

    def predict(self):
        if len(self.buffer) < SEQ_LENGTH:
            return None

        try:
            data = np.expand_dims(np.array(list(self.buffer)), axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            fall_prob = float(output.flatten()[0])
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {'prediction': prediction, 'probability': fall_prob}
            
        except Exception:
            return None

def create_imu_package(data, user_id):
    """간소화된 IMU 패키지"""
    return {
        'type': 'imu_data',
        'data': {
            'user_id': user_id,
            'timestamp': datetime.now(KST).isoformat(),
            'acc_x': float(data[0]),
            'acc_y': float(data[1]),
            'acc_z': float(data[2]),
            'gyr_x': float(data[3]),
            'gyr_y': float(data[4]),
            'gyr_z': float(data[5])
        }
    }

def create_fall_package(user_id, probability, sensor_data):
    """간소화된 낙상 패키지"""
    return {
        'type': 'fall_detection',
        'data': {
            'user_id': user_id,
            'timestamp': datetime.now(KST).isoformat(),
            'fall_detected': True,
            'confidence_score': float(probability),
            'sensor_data': {
                'acceleration': {'x': float(sensor_data[0]), 'y': float(sensor_data[1]), 'z': float(sensor_data[2])},
                'gyroscope': {'x': float(sensor_data[3]), 'y': float(sensor_data[4]), 'z': float(sensor_data[5])}
            }
        }
    }

async def websocket_handler(data_sender):
    """WebSocket 연결 핸들러"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    retry_delay = 1
    
    while True:
        try:
            async with websockets.connect(url) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                retry_delay = 1
                print("WebSocket connected")
                
                await data_sender.send_loop()
                
        except Exception:
            pass
        finally:
            data_sender.websocket = None
            data_sender.connected = False
        
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)

def main():
    """메인 함수"""
    print("Optimized Fall Detection System Started")
    
    # 초기화
    try:
        sensor = OptimizedSensor()
        fall_detector = OptimizedFallDetector()
        walking_detector = SimpleWalkingDetector()
        state_manager = SimpleStateManager()
        data_sender = OptimizedDataSender()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 종료 핸들러
    def signal_handler(sig, frame):
        print("Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # WebSocket 스레드 시작
    def start_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(data_sender))
    
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    
    # 초기 버퍼 채우기
    for _ in range(SEQ_LENGTH):
        data = sensor.get_data()
        fall_detector.add_data(data)
        time.sleep(1.0 / SAMPLING_RATE)
    
    print("System ready")
    
    # 메인 루프
    last_print = time.time()
    imu_send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # 보행 감지
            is_walking = walking_detector.add_data(data[0], data[1], data[2])
            
            # 낙상 감지
            fall_detector.add_data(data)
            fall_result = None
            if fall_detector.should_predict():
                fall_result = fall_detector.predict()
            
            fall_detected = fall_result and fall_result['prediction'] == 1
            
            # 상태 업데이트
            state_manager.update_state(is_walking, fall_detected)
            current_state = state_manager.current_state
            
            # 데이터 전송
            if fall_detected:
                print(f"FALL DETECTED! Confidence: {fall_result['probability']:.2%}")
                fall_package = create_fall_package(USER_ID, fall_result['probability'], data)
                data_sender.add_fall_data(fall_package)
            
            elif current_state == UserState.WALKING:
                imu_send_counter += 1
                if imu_send_counter >= (SAMPLING_RATE // SEND_RATE):
                    imu_package = create_imu_package(data, USER_ID)
                    data_sender.add_imu_data(imu_package)
                    imu_send_counter = 0
            
            # 간소화된 상태 출력 (10초마다)
            if current_time - last_print >= 10.0:
                print(f"State: {current_state.value}, Walking: {is_walking}, Connected: {data_sender.connected}")
                last_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main() 