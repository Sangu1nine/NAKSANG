"""
Compact ROC-based Walking Detection for Raspberry Pi
- 과학적 ROC 분석 기반 (F1 Score: 0.641)
- KFall 데이터셋 32명, 21,696윈도우 분석 결과
- 핵심 기능만 간결하게 구현 (300줄 이하)
- 🔧 수정: 센서 정규화, 낙상 감지 타이밍, 상태 관리 버그 수정
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
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False

# === 설정 ===
DEV_ADDR, PWR_MGMT_1 = 0x68, 0x6B
ACCEL_REGS = [0x3B, 0x3D, 0x3F]
GYRO_REGS = [0x43, 0x45, 0x47]
SENS_ACCEL, SENS_GYRO = 16384.0, 131.0
MODEL_PATH = 'models/fall_detection.tflite'
SCALERS_DIR = 'scalers'
SEQ_LENGTH, STRIDE = 150, 5
USER_ID = "raspberry_pi_01"
WS_SERVER = "ws://192.168.0.177:8000"
KST = timezone(timedelta(hours=9))

class UserState(Enum):
    DAILY = "Idle"
    WALKING = "Walking" 
    FALL = "Fall"

class ROCWalkingDetector:
    """과학적 ROC 분석 기반 보행 감지 (F1: 0.641)"""
    
    def __init__(self):
        # 🎯 ROC 최적화 임계값 (KFall 데이터셋)
        self.thresholds = {
            'acc_mean': (0.918, 1.122),    # AUC 0.843
            'acc_std': 0.134,              # AUC 0.835  
            'step_freq': (1.0, 4.0),       # 생리학적
            'regularity': 0.869,           # AUC 0.833
            'confidence': 0.6
        }
        
        # F1 최적화 가중치
        self.weights = {'acc_mean': 0.25, 'acc_std': 0.25, 'step_freq': 0.35, 'regularity': 0.15}
        
        self.buffer = deque(maxlen=150)  # 1.5초 @ 100Hz
        self.is_walking = False
        self.confidence = 0.0
        self.consecutive_walk = self.consecutive_idle = 0
        print("🎯 ROC Walking Detector: F1=0.641, KFall dataset")

    def detect(self, acc_x, acc_y, acc_z):
        """실시간 보행 감지"""
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        self.buffer.append((acc_mag, time.time()))
        
        if len(self.buffer) >= 150:
            self._analyze()
        return self.is_walking, self.confidence

    def _analyze(self):
        """ROC 기반 분석"""
        acc_data = np.array([x[0] for x in self.buffer])
        time_data = np.array([x[1] for x in self.buffer])
        
        # 특징 계산
        acc_mean, acc_std = np.mean(acc_data), np.std(acc_data)
        acc_smooth = np.convolve(acc_data, np.ones(5)/5, mode='same')
        
        # 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = [i for i in range(5, len(acc_smooth)-5) 
                if acc_smooth[i] > threshold and 
                acc_smooth[i] == np.max(acc_smooth[i-5:i+6])]
        
        # 보행 특징
        step_freq = regularity = 0.0
        if len(peaks) >= 2:
            intervals = np.diff(time_data[peaks])
            if len(intervals) > 0 and np.all(intervals > 0):
                step_freq = 1.0 / np.mean(intervals)
                regularity = 1.0 / (1.0 + np.std(intervals))
        
        # ROC 신뢰도 계산
        confidence = 0.0
        if self.thresholds['acc_mean'][0] <= acc_mean <= self.thresholds['acc_mean'][1]:
            confidence += self.weights['acc_mean']
        if acc_std >= self.thresholds['acc_std']:
            confidence += self.weights['acc_std']
        if self.thresholds['step_freq'][0] <= step_freq <= self.thresholds['step_freq'][1]:
            confidence += self.weights['step_freq']
        if regularity >= self.thresholds['regularity']:
            confidence += self.weights['regularity']
        
        self.confidence = confidence
        
        # 상태 업데이트 (디바운싱)
        if confidence >= self.thresholds['confidence']:
            self.consecutive_walk += 1
            self.consecutive_idle = 0
            if not self.is_walking and self.consecutive_walk >= 3:
                self.is_walking = True
                print(f"🚶 Walking started (ROC: {confidence:.3f})")
        else:
            self.consecutive_idle += 1
            self.consecutive_walk = 0
            if self.is_walking and self.consecutive_idle >= 5:
                self.is_walking = False
                print(f"🚶 Walking stopped")

class CompactSensor:
    """🔧 개선된 센서 (정규화 포함)"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("SMBus2 missing")
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        
        # 🔧 스케일러 로드 추가
        self.scalers = self._load_scalers()
        print(f"📊 Loaded {len(self.scalers)} scalers for normalization")

    def _load_scalers(self):
        """스케일러 로드"""
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
                print(f"⚠️ Failed to load scaler {feature}")
        
        return scalers

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val

    def get_data(self):
        """🔧 정규화된 센서 데이터"""
        raw_data = []
        for reg in ACCEL_REGS:
            raw_data.append(self._read_word_2c(reg) / SENS_ACCEL)
        for reg in GYRO_REGS:
            raw_data.append(self._read_word_2c(reg) / SENS_GYRO)
        
        # 🔧 정규화 적용
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

class CompactFallDetector:
    """🔧 개선된 낙상 감지 (타이밍 로직 수정)"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("🔧 Fall detector with proper timing loaded")

    def add_data(self, data):
        self.buffer.append(data)
        self.counter += 1

    def should_predict(self):
        """🔧 예측 타이밍 체크 추가"""
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0

    def predict(self):
        """🔧 타이밍 체크 후 예측"""
        if not self.should_predict():
            return None
            
        try:
            input_data = np.expand_dims(np.array(list(self.buffer)), axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            prob = float(output.flatten()[0])
            return {'prediction': 1 if prob >= 0.5 else 0, 'probability': prob}
        except Exception as e:
            print(f"🔧 Prediction error: {e}")
            return None

class StateManager:
    """🔧 상태 관리 (타이밍 버그 수정)"""
    def __init__(self):
        self.state = UserState.DAILY
        self.state_time = time.time()
        self.last_fall = None
        print("🔧 State manager with proper timing initialized")
        
    def update(self, is_walking, fall_detected):
        current_time = time.time()
        
        # 🔧 낙상 감지 (쿨다운 포함)
        if fall_detected and (not self.last_fall or current_time - self.last_fall > 10):
            self.state = UserState.FALL
            self.state_time = current_time  # 🔧 타이밍 업데이트 추가
            self.last_fall = current_time
            print(f"🔧 Fall detected! State changed to {self.state.value}")
            return True
            
        # 🔧 보행 시작
        elif self.state == UserState.DAILY and is_walking:
            self.state = UserState.WALKING
            self.state_time = current_time  # 🔧 타이밍 업데이트 추가
            return True
            
        # 🔧 보행 종료 (3초 대기)
        elif self.state == UserState.WALKING and not is_walking:
            if current_time - self.state_time > 3:
                self.state = UserState.DAILY
                self.state_time = current_time  # 🔧 타이밍 업데이트 추가
                return True
                
        # 🔧 낙상 복구 (3초 후)
        elif self.state == UserState.FALL and current_time - self.state_time > 3:
            self.state = UserState.DAILY
            self.state_time = current_time  # 🔧 타이밍 업데이트 추가
            print(f"🔧 Recovered from fall, back to {self.state.value}")
            return True
            
        return False

class DataSender:
    """데이터 전송"""
    def __init__(self):
        self.websocket = None
        self.connected = False
        
    async def connect_and_send(self):
        url = f"{WS_SERVER}/ws/{USER_ID}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    self.websocket = ws
                    self.connected = True
                    print("✅ WebSocket connected")
                    await asyncio.sleep(3600)  # 1시간 유지
            except:
                self.connected = False
                await asyncio.sleep(5)
    
    def send_data(self, data_type, data, analysis=None):
        if not self.websocket:
            return
        try:
            package = {
                'type': data_type,
                'data': {
                    'user_id': USER_ID,
                    'timestamp': datetime.now(KST).isoformat(),
                    **data
                }
            }
            if analysis:
                package['roc_analysis'] = analysis
            
            # 비동기로 전송 시도
            asyncio.create_task(self._async_send(json.dumps(package)))
        except:
            pass
    
    async def _async_send(self, message):
        try:
            if self.websocket:
                await self.websocket.send(message)
        except:
            pass

def main():
    """메인 함수"""
    print("🚀 Compact ROC-based Detection System")
    print("📊 KFall Dataset F1=0.641, 300줄 최적화")
    
    # 초기화
    try:
        sensor = CompactSensor()
        fall_detector = CompactFallDetector()
        walking_detector = ROCWalkingDetector()
        state_manager = StateManager()
        data_sender = DataSender()
    except Exception as e:
        print(f"Init failed: {e}")
        return

    # 종료 핸들러
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    # WebSocket 스레드
    def ws_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(data_sender.connect_and_send())
    
    threading.Thread(target=ws_thread, daemon=True).start()
    
    # 초기 버퍼 채우기
    for _ in range(SEQ_LENGTH):
        fall_detector.add_data(sensor.get_data())
        time.sleep(0.01)
    
    print("🎯 ROC detection started")
    
    # 메인 루프
    last_print = time.time()
    send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # ROC 보행 감지
            is_walking, confidence = walking_detector.detect(data[0], data[1], data[2])
            
            # 🔧 낙상 감지 (타이밍 체크 추가)
            fall_detector.add_data(data)
            fall_result = None
            if fall_detector.should_predict():  # 🔧 타이밍 체크 추가
                fall_result = fall_detector.predict()
            
            fall_detected = fall_result and fall_result['prediction'] == 1
            
            # 상태 업데이트
            state_manager.update(is_walking, fall_detected)
            
            # 데이터 전송
            analysis = {'walking': is_walking, 'confidence': confidence, 'roc_based': True}
            
            if fall_detected:
                print(f"🚨 FALL! Confidence: {fall_result['probability']:.2%}")
                fall_data = {
                    'fall_detected': True,
                    'confidence_score': fall_result['probability'],
                    'sensor_data': {'acc': data[:3].tolist(), 'gyr': data[3:].tolist()}
                }
                data_sender.send_data('fall_detection', fall_data, analysis)
            
            elif state_manager.state == UserState.WALKING:
                send_counter += 1
                if send_counter >= 10:  # 10Hz -> 1Hz
                    imu_data = {
                        'acc_x': float(data[0]), 'acc_y': float(data[1]), 'acc_z': float(data[2]),
                        'gyr_x': float(data[3]), 'gyr_y': float(data[4]), 'gyr_z': float(data[5])
                    }
                    data_sender.send_data('imu_data', imu_data, analysis)
                    send_counter = 0
            
            # 🔧 상태 출력 개선 (5초마다)
            if current_time - last_print >= 5:
                state_duration = current_time - state_manager.state_time
                print(f"📊 {state_manager.state.value} ({state_duration:.1f}s), ROC Walk: {is_walking}, "
                      f"Conf: {confidence:.3f}, WS: {data_sender.connected}")
                last_print = current_time
            
            time.sleep(0.01)  # 100Hz
            
        except Exception as e:
            print(f"🔧 Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main() 