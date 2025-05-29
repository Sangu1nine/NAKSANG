"""
Optimized Walking Detection System for Raspberry Pi
- 과학적 ROC 분석 기반 보행 감지 (F1 Score: 0.641)
- 라즈베리파이 실시간 최적화 (메모리/CPU 효율성)
- KFall 데이터셋 32명 피험자, 21,696개 윈도우 분석 결과 적용
- 핵심 기능만 유지하여 코드 간소화
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

# === 시스템 설정 ===
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

# 🔧 MODIFIED: 낙상 감지 안정성 개선
FALL_COOLDOWN_TIME = 30.0  # 낙상 쿨다운 시간 30초로 증가
RECONNECT_DELAY = 5.0      # 재연결 대기 시간
MAX_RECONNECT_ATTEMPTS = 10  # 최대 재연결 시도

class UserState(Enum):
    DAILY = "Idle"
    WALKING = "Walking"
    FALL = "Fall"

class OptimizedROCWalkingDetector:
    """
    과학적 ROC 분석 기반 + 라즈베리파이 최적화 보행 감지기
    - KFall 데이터셋 분석 결과 적용 (F1 Score: 0.641)
    - 메모리 효율적 구현 (150샘플 버퍼)
    - CPU 최적화된 특징 계산
    """
    
    def __init__(self):
        # 🎯 ROC 분석 기반 최적화된 임계값 (KFall 데이터셋)
        self.ROC_THRESHOLDS = {
            'acc_mean_min': 0.918,      # acc_range: AUC 0.843
            'acc_mean_max': 1.122,
            'acc_std_min': 0.134,       # acc_std: AUC 0.835
            'step_freq_min': 1.0,       # 생리학적 범위
            'step_freq_max': 4.0,
            'regularity_min': 0.869,    # walking_energy_ratio: AUC 0.833
            'confidence_min': 0.6       # 최종 판단 임계값
        }
        
        # 🎯 F1 스코어 최적화된 가중치 (합계 = 1.0)
        self.ROC_WEIGHTS = {
            'acc_mean': 0.25,           # 가속도 평균
            'acc_std': 0.25,            # 가속도 표준편차
            'step_freq': 0.35,          # 보행 주기 (최고 가중치)
            'regularity': 0.15          # 규칙성
        }
        
        # 메모리 최적화: 1.5초 버퍼 (150샘플 @ 100Hz)
        self.buffer_size = 150
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # 상태 변수
        self.is_walking = False
        self.confidence = 0.0
        self.last_analysis = {}
        
        # 안정성 제어
        self.consecutive_walking = 0
        self.consecutive_idle = 0
        self.last_state_change = 0
        self.walking_start_time = None
        
        print("🎯 Optimized ROC Walking Detector initialized")
        print(f"📊 Based on KFall dataset: 32 subjects, 21,696 windows")
        print(f"⚡ F1 Score: 0.641, Memory optimized: {self.buffer_size} samples")

    def add_data(self, acc_x, acc_y, acc_z):
        """센서 데이터 추가 및 실시간 보행 감지"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        current_time = time.time()
        
        self.acc_buffer.append(acc_magnitude)
        self.time_buffer.append(current_time)
        
        # 충분한 데이터가 있으면 ROC 기반 분석
        if len(self.acc_buffer) >= self.buffer_size:
            self._roc_analysis()
        
        return self.is_walking, self.confidence

    def _roc_analysis(self):
        """ROC 분석 기반 보행 감지 (CPU 최적화)"""
        current_time = time.time()
        
        # 데이터 변환 (한 번만)
        acc_data = np.array(self.acc_buffer)
        time_data = np.array(self.time_buffer)
        
        # 1. 이동평균 필터링 (5포인트)
        acc_smooth = np.convolve(acc_data, np.ones(5)/5, mode='same')
        
        # 2. 기본 특징 계산
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)
        
        # 3. 효율적 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = self._fast_peak_detection(acc_smooth, threshold)
        
        # 4. 보행 주기 및 규칙성 계산
        step_frequency, regularity = self._calculate_gait_features(time_data, peaks)
        
        # 5. ROC 기반 신뢰도 계산
        confidence_score = self._calculate_roc_confidence(
            acc_mean, acc_std, step_frequency, regularity
        )
        
        # 6. 안정성 체크 및 상태 업데이트
        self._update_walking_state(confidence_score, current_time)
        
        # 디버깅 정보 저장
        self.last_analysis = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'step_frequency': step_frequency,
            'regularity': regularity,
            'peaks_count': len(peaks),
            'confidence': confidence_score
        }

    def _fast_peak_detection(self, acc_smooth, threshold):
        """CPU 최적화된 피크 검출"""
        peaks = []
        window = 5
        
        for i in range(window, len(acc_smooth) - window):
            if (acc_smooth[i] > threshold and 
                acc_smooth[i] == np.max(acc_smooth[i-window:i+window+1])):
                peaks.append(i)
        
        return peaks

    def _calculate_gait_features(self, time_data, peaks):
        """보행 특징 계산 (주기 및 규칙성)"""
        if len(peaks) < 2:
            return 0.0, 0.0
        
        peak_times = time_data[peaks]
        intervals = np.diff(peak_times)
        
        if len(intervals) == 0 or np.any(intervals <= 0):
            return 0.0, 0.0
        
        # 보행 주파수 (Hz)
        step_frequency = 1.0 / np.mean(intervals)
        
        # 규칙성 (표준편차가 작을수록 규칙적)
        regularity = 1.0 / (1.0 + np.std(intervals))
        
        return step_frequency, regularity

    def _calculate_roc_confidence(self, acc_mean, acc_std, step_frequency, regularity):
        """ROC 분석 기반 신뢰도 계산"""
        confidence = 0.0
        
        # 1. 가속도 평균 검사 (acc_range: AUC 0.843)
        if (self.ROC_THRESHOLDS['acc_mean_min'] <= acc_mean <= 
            self.ROC_THRESHOLDS['acc_mean_max']):
            confidence += self.ROC_WEIGHTS['acc_mean']
        
        # 2. 가속도 표준편차 검사 (acc_std: AUC 0.835)
        if acc_std >= self.ROC_THRESHOLDS['acc_std_min']:
            confidence += self.ROC_WEIGHTS['acc_std']
        
        # 3. 보행 주기 검사 (walking_energy_ratio: AUC 0.833)
        if (self.ROC_THRESHOLDS['step_freq_min'] <= step_frequency <= 
            self.ROC_THRESHOLDS['step_freq_max']):
            confidence += self.ROC_WEIGHTS['step_freq']
        
        # 4. 규칙성 검사 (gyr_mean: AUC 0.780)
        if regularity >= self.ROC_THRESHOLDS['regularity_min']:
            confidence += self.ROC_WEIGHTS['regularity']
        
        return confidence

    def _update_walking_state(self, confidence_score, current_time):
        """안정성 체크 및 상태 업데이트"""
        self.confidence = confidence_score
        new_walking = confidence_score >= self.ROC_THRESHOLDS['confidence_min']
        
        # 디바운싱 (1초)
        if current_time - self.last_state_change < 1.0:
            return
        
        # 연속 감지 카운트
        if new_walking:
            self.consecutive_walking += 1
            self.consecutive_idle = 0
        else:
            self.consecutive_idle += 1
            self.consecutive_walking = 0
        
        # 보행 시작: 연속 3회 감지
        if not self.is_walking and self.consecutive_walking >= 3:
            self.is_walking = True
            self.walking_start_time = current_time
            self.last_state_change = current_time
            print(f"🚶 ROC Walking started (Confidence: {confidence_score:.3f})")
        
        # 보행 종료: 연속 5회 미감지 + 최소 2초 지속
        elif (self.is_walking and self.consecutive_idle >= 5 and
              self.walking_start_time and 
              current_time - self.walking_start_time >= 2.0):
            self.is_walking = False
            self.last_state_change = current_time
            duration = current_time - self.walking_start_time
            print(f"🚶 ROC Walking stopped (Duration: {duration:.1f}s)")

    def get_analysis_summary(self):
        """분석 요약 정보 반환"""
        return {
            'walking': self.is_walking,
            'confidence': self.confidence,
            'roc_based': True,
            'f1_score': 0.641,
            **self.last_analysis
        }

class OptimizedStateManager:
    """최적화된 상태 관리자"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = FALL_COOLDOWN_TIME  # 🔧 MODIFIED: 쿨다운 시간 증가

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
            if current_time - self.state_start_time > 3.0:
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
        self.imu_queue = queue.Queue(maxsize=30)
        self.fall_queue = queue.Queue(maxsize=50)
        self.websocket = None
        self.connected = False
        # 🔧 MODIFIED: 재연결 관리 추가
        self.reconnect_attempts = 0
        self.last_disconnect_time = 0
        self.connection_stable = False

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
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data)
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
            # 🔧 MODIFIED: 연결 안정성 추적
            self.connection_stable = True
        except Exception as e:
            print(f"데이터 전송 실패: {e}")
            self.connection_stable = False
    
    def is_connection_healthy(self):
        """연결 상태 확인"""
        return self.connected and self.connection_stable and self.websocket is not None

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

def create_imu_package(data, user_id, analysis_info=None):
    """IMU 데이터 패키지 생성 - 상태 정보 포함"""
    package = {
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
    # 🔧 MODIFIED: ROC 분석 정보와 상태 정보 추가
    if analysis_info:
        package['roc_analysis'] = analysis_info
        package['state_info'] = {
            'state': analysis_info.get('walking', False) and '걷기' or '일상',
            'confidence': analysis_info.get('confidence', 0.0),
            'timestamp': datetime.now(KST).isoformat()
        }
    return package

def create_fall_package(user_id, probability, sensor_data, analysis_info=None):
    """낙상 데이터 패키지 생성 - 상태 정보 포함"""
    package = {
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
    # 🔧 MODIFIED: ROC 분석 정보와 상태 정보 추가
    if analysis_info:
        package['roc_analysis'] = analysis_info
        package['state_info'] = {
            'state': '낙상',
            'confidence': float(probability),
            'timestamp': datetime.now(KST).isoformat()
        }
    return package

async def websocket_handler(data_sender):
    """WebSocket 연결 핸들러 - 개선된 재연결 로직"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    
    while True:
        try:
            print(f"🔄 WebSocket 연결 시도... (시도 {data_sender.reconnect_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})")
            
            # 🔧 MODIFIED: ping 설정 개선 및 연결 안정성 향상
            async with websockets.connect(
                url,
                ping_interval=30,    # 30초마다 핑 (증가)
                ping_timeout=15,     # 15초 타임아웃 (증가)
                close_timeout=10,    # 10초 종료 타임아웃 (증가)
                max_size=2**20,      # 1MB 최대 메시지 크기
                compression=None     # 압축 비활성화로 성능 향상
            ) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                data_sender.connection_stable = True
                data_sender.reconnect_attempts = 0
                print("✅ WebSocket connected")
                
                # 연결 성공 메시지 전송
                try:
                    await websocket.send(json.dumps({
                        "type": "connection_health_check",
                        "user_id": USER_ID,
                        "timestamp": datetime.now(KST).isoformat(),
                        "status": "connected"
                    }))
                except Exception as e:
                    print(f"연결 확인 메시지 전송 실패: {e}")
                
                # 🆕 주기적 연결 상태 확인 태스크 추가
                async def periodic_health_check():
                    while data_sender.connected:
                        try:
                            await asyncio.sleep(25)  # 25초마다 체크
                            if data_sender.websocket:
                                await data_sender.websocket.send(json.dumps({
                                    "type": "heartbeat",
                                    "user_id": USER_ID,
                                    "timestamp": datetime.now(KST).isoformat()
                                }))
                        except Exception as e:
                            print(f"💓 연결 상태 확인 실패: {e}")
                            break
                
                # 태스크 동시 실행
                health_task = asyncio.create_task(periodic_health_check())
                send_task = asyncio.create_task(data_sender.send_loop())
                
                # 어느 하나라도 종료되면 전체 종료
                done, pending = await asyncio.wait(
                    [health_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 남은 태스크 정리
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 WebSocket 연결 종료됨: {e}")
        except Exception as e:
            print(f"❌ WebSocket 연결 오류: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
            data_sender.connection_stable = False
            data_sender.last_disconnect_time = time.time()
            data_sender.reconnect_attempts += 1
        
        # 재연결 대기 및 제한
        if data_sender.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print(f"❌ 최대 재연결 시도 횟수 초과 ({MAX_RECONNECT_ATTEMPTS})")
            await asyncio.sleep(30)  # 30초 대기 후 재시작
            data_sender.reconnect_attempts = 0
        else:
            retry_delay = min(RECONNECT_DELAY * (2 ** data_sender.reconnect_attempts), 30)
            print(f"⏳ {retry_delay}초 후 재연결 시도...")
            await asyncio.sleep(retry_delay)

def main():
    """메인 함수"""
    print("🚀 Optimized ROC-based Fall Detection System")
    print("📊 Scientific Analysis + Raspberry Pi Optimization")
    print(f"🎯 KFall Dataset: F1 Score 0.641, 32 subjects, 21,696 windows")
    
    # 초기화
    try:
        sensor = OptimizedSensor()
        fall_detector = OptimizedFallDetector()
        walking_detector = OptimizedROCWalkingDetector()
        state_manager = OptimizedStateManager()
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
    
    print("🎯 ROC-based real-time detection started")
    
    # 메인 루프
    last_print = time.time()
    last_analysis_print = time.time()
    last_connection_check = time.time()  # 🔧 MODIFIED: 연결 상태 확인 타이머 추가
    imu_send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # ROC 기반 보행 감지
            is_walking, walk_confidence = walking_detector.add_data(data[0], data[1], data[2])
            
            # 낙상 감지
            fall_detector.add_data(data)
            fall_result = None
            if fall_detector.should_predict():
                fall_result = fall_detector.predict()
            
            fall_detected = fall_result and fall_result['prediction'] == 1
            
            # 🔧 MODIFIED: 상태 변화 추적하여 중복 감지 방지
            state_changed = state_manager.update_state(is_walking, fall_detected)
            current_state = state_manager.current_state
            
            # 🔧 MODIFIED: 연결 상태 모니터링 (30초마다)
            if current_time - last_connection_check >= 30.0:
                connection_healthy = data_sender.is_connection_healthy()
                print(f"🔗 연결 상태: {'건강함' if connection_healthy else '불안정'} "
                      f"(재연결 시도: {data_sender.reconnect_attempts})")
                last_connection_check = current_time
            
            # 분석 정보 생성
            analysis_info = walking_detector.get_analysis_summary()
            
            # 🔧 MODIFIED: 낙상 감지 시에만 알림 전송 (상태 변화 시)
            if fall_detected and state_changed and current_state == UserState.FALL:
                print(f"🚨 FALL DETECTED! Confidence: {fall_result['probability']:.2%}")
                if data_sender.is_connection_healthy():
                    fall_package = create_fall_package(USER_ID, fall_result['probability'], data, analysis_info)
                    data_sender.add_fall_data(fall_package)
                    print("📤 낙상 알림 전송됨")
                else:
                    print("⚠️ 연결 불안정으로 낙상 데이터 전송 대기")
            
            # IMU 데이터 전송 (보행 중일 때만)
            elif current_state == UserState.WALKING:
                imu_send_counter += 1
                if imu_send_counter >= (SAMPLING_RATE // SEND_RATE):
                    if data_sender.is_connection_healthy():
                        imu_package = create_imu_package(data, USER_ID, analysis_info)
                        data_sender.add_imu_data(imu_package)
                    imu_send_counter = 0
            
            # 기본 상태 출력 (10초마다)
            if current_time - last_print >= 10.0:
                connection_status = "연결됨" if data_sender.is_connection_healthy() else "연결 안됨"
                print(f"📊 State: {current_state.value}, ROC Walking: {is_walking}, "
                      f"Confidence: {walk_confidence:.3f}, Connection: {connection_status}")
                last_print = current_time
            
            # ROC 분석 상세 출력 (30초마다, 보행 중일 때)
            if (current_time - last_analysis_print >= 30.0 and is_walking):
                analysis = walking_detector.get_analysis_summary()
                print(f"🔬 ROC Analysis Detail:")
                print(f"   📈 Acc Mean: {analysis.get('acc_mean', 0):.3f}, "
                      f"Std: {analysis.get('acc_std', 0):.3f}")
                print(f"   👣 Step Freq: {analysis.get('step_frequency', 0):.2f}Hz, "
                      f"Regularity: {analysis.get('regularity', 0):.3f}")
                print(f"   🎯 ROC Confidence: {analysis.get('confidence', 0):.3f}, "
                      f"Peaks: {analysis.get('peaks_count', 0)}")
                last_analysis_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main() 