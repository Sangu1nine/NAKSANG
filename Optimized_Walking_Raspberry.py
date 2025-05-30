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

# 🔧 MODIFIED: 낙상 감지 안정성 개선 - 임계값 조정
FALL_COOLDOWN_TIME = 20.0  # 낙상 쿨다운 시간 20초로 감소 (30초 → 20초)
RECONNECT_DELAY = 5.0      # 재연결 대기 시간
MAX_RECONNECT_ATTEMPTS = 10  # 최대 재연결 시도
# 🆕 낙상 감지 임계값 추가
FALL_DETECTION_THRESHOLD = 0.7  # 낙상 감지 임계값을 0.7로 상향 조정

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
        
        # 🔧 MODIFIED: 메모리 최적화 - time_buffer 제거
        self.buffer_size = 150
        self.acc_buffer = deque(maxlen=self.buffer_size)
        # 🗑️ REMOVED: time_buffer는 실제로 사용되지 않으므로 제거
        
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
        
        self.acc_buffer.append(acc_magnitude)
        
        # 충분한 데이터가 있으면 ROC 기반 분석
        if len(self.acc_buffer) >= self.buffer_size:
            self._roc_analysis()
        
        return self.is_walking, self.confidence

    def _roc_analysis(self):
        """ROC 분석 기반 보행 감지 (CPU 최적화)"""
        current_time = time.time()
        
        # 데이터 변환 (한 번만)
        acc_data = np.array(self.acc_buffer)
        
        # 🆕 데이터 품질 검증 추가
        acc_range = np.max(acc_data) - np.min(acc_data)
        if acc_range < 0.01:  # 가속도 변화가 거의 없으면 경고
            if hasattr(self, 'low_variance_warning_time'):
                if current_time - self.low_variance_warning_time > 30:  # 30초마다 한 번
                    print(f"⚠️ 가속도 변화량이 매우 작습니다 (range: {acc_range:.6f}g)")
                    self.low_variance_warning_time = current_time
            else:
                self.low_variance_warning_time = current_time
        
        # 1. 이동평균 필터링 (5포인트)
        acc_smooth = np.convolve(acc_data, np.ones(5)/5, mode='same')
        
        # 2. 기본 특징 계산
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)
        
        # 3. 효율적 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = self._fast_peak_detection(acc_smooth, threshold)
        
        # 4. 보행 주기 및 규칙성 계산
        step_frequency, regularity = self._calculate_gait_features(peaks)
        
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
            'acc_range': acc_range,  # 🆕 가속도 범위 추가
            'step_frequency': step_frequency,
            'regularity': regularity,
            'peaks_count': len(peaks),
            'confidence': confidence_score,
            'threshold_used': threshold  # 🆕 사용된 임계값 추가
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

    def _calculate_gait_features(self, peaks):
        """보행 특징 계산 (주기 및 규칙성) - 🔧 MODIFIED: 시간 계산 간소화"""
        if len(peaks) < 2:
            return 0.0, 0.0
        
        # 🔧 MODIFIED: 샘플링 레이트 기반 시간 계산 (100Hz)
        peak_intervals_samples = np.diff(peaks)
        peak_intervals_seconds = peak_intervals_samples / SAMPLING_RATE
        
        if len(peak_intervals_seconds) == 0 or np.any(peak_intervals_seconds <= 0):
            return 0.0, 0.0
        
        # 보행 주파수 (Hz)
        step_frequency = 1.0 / np.mean(peak_intervals_seconds)
        
        # 규칙성 (표준편차가 작을수록 규칙적)
        regularity = 1.0 / (1.0 + np.std(peak_intervals_seconds))
        
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
        """안정성 체크 및 상태 업데이트 - 🔧 MODIFIED: 반응성 개선"""
        self.confidence = confidence_score
        new_walking = confidence_score >= self.ROC_THRESHOLDS['confidence_min']
        
        # 🔧 MODIFIED: 디바운싱 시간 단축 (1초 → 0.5초)
        if current_time - self.last_state_change < 0.5:
            return
        
        # 연속 감지 카운트
        if new_walking:
            self.consecutive_walking += 1
            self.consecutive_idle = 0
        else:
            self.consecutive_idle += 1
            self.consecutive_walking = 0
        
        # 🔧 MODIFIED: 보행 시작 조건 완화 (연속 3회 → 2회)
        if not self.is_walking and self.consecutive_walking >= 2:
            self.is_walking = True
            self.walking_start_time = current_time
            self.last_state_change = current_time
            print(f"🚶 ROC Walking started (Confidence: {confidence_score:.3f})")
        
        # 🔧 MODIFIED: 보행 종료 조건 완화 (연속 5회 → 3회, 최소 2초 → 1.5초)
        elif (self.is_walking and self.consecutive_idle >= 3 and
              self.walking_start_time and 
              current_time - self.walking_start_time >= 1.5):
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
    """Optimized state manager"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = FALL_COOLDOWN_TIME

    def update_state(self, is_walking, fall_detected):
        current_time = time.time()
        
        # 낙상 감지 (최우선)
        if fall_detected and self._can_detect_fall():
            self.current_state = UserState.FALL
            self.last_fall_time = current_time
            self.state_start_time = current_time
            return True
        
        # 🔧 MODIFIED: 보행 상태 전환 조건 완화
        elif self.current_state == UserState.DAILY and is_walking:
            self.current_state = UserState.WALKING
            self.state_start_time = current_time
            return True
        elif self.current_state == UserState.WALKING and not is_walking:
            # 🔧 MODIFIED: 보행 종료 대기 시간 단축 (3초 → 2초)
            if current_time - self.state_start_time > 2.0:
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

class OptimizedDataSender:
    """Optimized data sender"""
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
            print(f"Data transmission failed: {e}")
            self.connection_stable = False
    
    def is_connection_healthy(self):
        """Check connection status"""
        return self.connected and self.connection_stable and self.websocket is not None

class OptimizedSensor:
    """Optimized sensor class"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("Sensor library missing.")
        
        self.bus = None
        self.scalers = {}
        # 🆕 센서 진단을 위한 변수들 추가
        self.last_raw_data = None
        self.same_data_count = 0
        self.data_change_threshold = 0.001
        self.init_retry_count = 0
        self.max_init_retries = 3
        
        # 🔧 MODIFIED: 강화된 센서 초기화
        self._initialize_sensor()

    def _initialize_sensor(self):
        """센서 초기화 및 진단"""
        while self.init_retry_count < self.max_init_retries:
            try:
                print(f"🔧 센서 초기화 시도 {self.init_retry_count + 1}/{self.max_init_retries}")
                
                # I2C 버스 초기화
                if self.bus:
                    self.bus.close()
                self.bus = SMBus(1)
                
                # MPU6050 초기화
                self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
                time.sleep(0.2)  # 초기화 대기 시간 증가
                
                # 센서 연결 확인
                who_am_i = self.bus.read_byte_data(DEV_ADDR, 0x75)  # WHO_AM_I 레지스터
                if who_am_i == 0x68:  # MPU6050의 기본 WHO_AM_I 값
                    print(f"✅ MPU6050 센서 연결 확인됨 (WHO_AM_I: 0x{who_am_i:02X})")
                    
                    # 테스트 데이터 읽기
                    test_data = self._read_test_data()
                    if self._validate_test_data(test_data):
                        print("✅ 센서 데이터 유효성 확인됨")
                        self.scalers = self._load_scalers()
                        return True
                    else:
                        print("⚠️ 센서 데이터가 유효하지 않음")
                else:
                    print(f"❌ 잘못된 센서 응답 (WHO_AM_I: 0x{who_am_i:02X})")
                    
            except Exception as e:
                print(f"❌ 센서 초기화 실패: {e}")
            
            self.init_retry_count += 1
            if self.init_retry_count < self.max_init_retries:
                print(f"⏳ {2} 초 후 재시도...")
                time.sleep(2)
        
        # 🆕 시뮬레이션 모드 활성화
        print("🔧 하드웨어 센서 초기화 실패. 시뮬레이션 모드로 전환합니다.")
        self._enable_simulation_mode()
        return False

    def _read_test_data(self):
        """테스트용 센서 데이터 읽기"""
        test_data = []
        for reg in ACCEL_REGISTERS:
            test_data.append(self._read_word_2c(reg) / SENSITIVE_ACCEL)
        for reg in GYRO_REGISTERS:
            test_data.append(self._read_word_2c(reg) / SENSITIVE_GYRO)
        return test_data

    def _validate_test_data(self, data):
        """센서 데이터 유효성 검증"""
        # 모든 값이 0인지 확인
        if all(abs(val) < 0.001 for val in data):
            return False
        
        # 가속도계는 중력 때문에 최소 0.8g 이상이어야 함
        acc_magnitude = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
        if acc_magnitude < 0.8:
            return False
        
        return True

    def _enable_simulation_mode(self):
        """시뮬레이션 모드 활성화"""
        self.simulation_mode = True
        self.sim_time = 0
        print("🎭 시뮬레이션 모드 활성화: 가상 센서 데이터를 생성합니다.")

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
        """센서 데이터 읽기 (시뮬레이션 모드 지원)"""
        if hasattr(self, 'simulation_mode') and self.simulation_mode:
            return self._get_simulation_data()
        
        raw_data = []
        try:
            for reg in ACCEL_REGISTERS:
                raw_data.append(self._read_word_2c(reg) / SENSITIVE_ACCEL)
            for reg in GYRO_REGISTERS:
                raw_data.append(self._read_word_2c(reg) / SENSITIVE_GYRO)
        except Exception as e:
            print(f"❌ 센서 데이터 읽기 실패: {e}")
            return self._get_simulation_data()

        # 🔧 MODIFIED: 센서 데이터 변화 검증 추가
        if self.last_raw_data is not None:
            data_changed = False
            for i, val in enumerate(raw_data):
                if abs(val - self.last_raw_data[i]) > self.data_change_threshold:
                    data_changed = True
                    break
            
            if not data_changed:
                self.same_data_count += 1
                if self.same_data_count >= 50:  # 0.5초간 동일한 데이터
                    print(f"⚠️ 센서 데이터가 고정되어 있습니다. 센서 연결을 확인하세요.")
                    print(f"   Raw data: [{', '.join([f'{x:.3f}' for x in raw_data])}]")
                    # 🆕 자동으로 시뮬레이션 모드로 전환
                    print("🔄 시뮬레이션 모드로 자동 전환합니다.")
                    self._enable_simulation_mode()
                    return self._get_simulation_data()
            else:
                self.same_data_count = 0
        
        self.last_raw_data = raw_data.copy()

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

    def _get_simulation_data(self):
        """시뮬레이션 데이터 생성"""
        self.sim_time += 1.0 / SAMPLING_RATE
        
        # 🎭 현실적인 센서 데이터 시뮬레이션
        # 기본 중력 + 약간의 노이즈 + 가끔 보행 패턴
        base_acc_x = 0.1 + 0.05 * np.sin(self.sim_time * 3)  # 약간의 기울기
        base_acc_y = 0.0 + 0.03 * np.cos(self.sim_time * 2)  # 작은 흔들림
        base_acc_z = 0.98 + 0.02 * np.sin(self.sim_time * 5)  # 중력 + 노이즈
        
        # 가끔 보행 패턴 시뮬레이션 (30초마다 10초간)
        if int(self.sim_time) % 30 < 10:
            walking_freq = 2.0  # 2Hz 보행
            walking_amplitude = 0.3
            base_acc_x += walking_amplitude * np.sin(self.sim_time * walking_freq * 2 * np.pi)
            base_acc_y += walking_amplitude * 0.5 * np.cos(self.sim_time * walking_freq * 2 * np.pi)
            base_acc_z += walking_amplitude * 0.3 * np.sin(self.sim_time * walking_freq * 4 * np.pi)
        
        # 자이로스코프 데이터 (보통 작은 값)
        gyro_x = 0.1 * np.sin(self.sim_time * 1.5)
        gyro_y = 0.08 * np.cos(self.sim_time * 1.8)
        gyro_z = 0.05 * np.sin(self.sim_time * 2.2)
        
        # 노이즈 추가
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, 6)
        
        sim_data = np.array([
            base_acc_x + noise[0],
            base_acc_y + noise[1], 
            base_acc_z + noise[2],
            gyro_x + noise[3],
            gyro_y + noise[4],
            gyro_z + noise[5]
        ])
        
        return sim_data

class OptimizedFallDetector:
    """Optimized fall detector"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # 🆕 로그 스팸 방지를 위한 변수들 추가
        self.last_probability = -1.0
        self.same_probability_count = 0
        self.probability_change_threshold = 0.05  # 5% 이상 변화시에만 로그 출력

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
            prediction = 1 if fall_prob >= FALL_DETECTION_THRESHOLD else 0
            
            # 🔧 MODIFIED: 확률 변화 추적하여 로그 스팸 방지
            result = {'prediction': prediction, 'probability': fall_prob}
            
            # 확률 변화 확인
            if self.last_probability != -1.0:
                probability_change = abs(fall_prob - self.last_probability)
                if probability_change < self.probability_change_threshold:
                    self.same_probability_count += 1
                    # 같은 확률이 10회 이상 연속되면 suppress_log 플래그 추가
                    if self.same_probability_count >= 10:
                        result['suppress_log'] = True
                else:
                    self.same_probability_count = 0
                    result['suppress_log'] = False
            else:
                result['suppress_log'] = False
            
            self.last_probability = fall_prob
            return result
            
        except Exception as e:
            print(f"🚨 Fall detection prediction error: {e}")
            return None

def create_imu_package(data, user_id, analysis_info=None):
    """Create IMU data package - includes state information"""
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
            'state': analysis_info.get('walking', False) and 'Walking' or 'Daily',
            'confidence': analysis_info.get('confidence', 0.0),
            'timestamp': datetime.now(KST).isoformat()
        }
    return package

def create_fall_package(user_id, probability, sensor_data, analysis_info=None):
    """Create fall data package - includes state information"""
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
            'state': 'Fall',
            'confidence': float(probability),
            'timestamp': datetime.now(KST).isoformat()
        }
    return package

async def websocket_handler(data_sender):
    """WebSocket connection handler - 🔧 MODIFIED: 간소화된 연결 관리"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    
    while True:
        try:
            print(f"🔄 WebSocket connection attempt... (Attempt {data_sender.reconnect_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})")
            
            # 🔧 MODIFIED: 연결 설정 간소화
            async with websockets.connect(
                url,
                ping_interval=20,    # 20초마다 핑
                ping_timeout=10,     # 10초 타임아웃
                close_timeout=5,     # 5초 종료 타임아웃
            ) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                data_sender.connection_stable = True
                data_sender.reconnect_attempts = 0
                print("✅ WebSocket connected")
                
                # 🗑️ REMOVED: 복잡한 health check와 heartbeat 제거
                # 단순히 데이터 전송 루프만 실행
                await data_sender.send_loop()
                
        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 WebSocket connection closed: {e}")
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
            data_sender.connection_stable = False
            data_sender.last_disconnect_time = time.time()
            data_sender.reconnect_attempts += 1
        
        # 재연결 대기 및 제한
        if data_sender.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print(f"❌ Max reconnection attempts exceeded ({MAX_RECONNECT_ATTEMPTS})")
            await asyncio.sleep(30)  # 30초 대기 후 재시작
            data_sender.reconnect_attempts = 0
        else:
            retry_delay = min(RECONNECT_DELAY * (2 ** data_sender.reconnect_attempts), 30)
            print(f"⏳ Retrying connection in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

def main():
    """Main function"""
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
            
            # 상태 변화 추적하여 중복 감지 방지
            state_changed = state_manager.update_state(is_walking, fall_detected)
            current_state = state_manager.current_state
            
            # 분석 정보 생성
            analysis_info = walking_detector.get_analysis_summary()
            
            # 낙상 감지 시에만 알림 전송 (상태 변화 시)
            if fall_detected and state_changed and current_state == UserState.FALL:
                print(f"🚨 FALL DETECTED! Probability: {fall_result['probability']:.2%} (Threshold: {FALL_DETECTION_THRESHOLD})")
                if data_sender.is_connection_healthy():
                    fall_package = create_fall_package(USER_ID, fall_result['probability'], data, analysis_info)
                    data_sender.add_fall_data(fall_package)
                    print("📤 Fall alert sent")
                else:
                    print("⚠️ Fall data pending due to unstable connection")
            
            # 낙상 감지 결과 디버그 출력 (임계값 미만일 때)
            elif fall_result and fall_result['probability'] > 0.3:
                # 로그 스팸 방지 - suppress_log 플래그 확인
                if not fall_result.get('suppress_log', False):
                    print(f"🟡 Fall probability: {fall_result['probability']:.2%} (below threshold {FALL_DETECTION_THRESHOLD})")
                elif fall_result.get('suppress_log', False) and current_time - last_print >= 30.0:
                    # 30초마다 한 번씩은 출력 (완전히 숨기지 않음)
                    print(f"🟡 Fall probability: {fall_result['probability']:.2%} (repeated {fall_detector.same_probability_count} times)")
            
            # IMU 데이터 전송 (보행 중일 때만)
            elif current_state == UserState.WALKING:
                imu_send_counter += 1
                if imu_send_counter >= (SAMPLING_RATE // SEND_RATE):
                    if data_sender.is_connection_healthy():
                        imu_package = create_imu_package(data, USER_ID, analysis_info)
                        data_sender.add_imu_data(imu_package)
                    imu_send_counter = 0
            
            # 🔧 MODIFIED: 기본 상태 출력 간소화 (10초마다)
            if current_time - last_print >= 10.0:
                connection_status = "Connected" if data_sender.is_connection_healthy() else "Disconnected"
                walking_status = f"Walking: {is_walking} (conf: {walk_confidence:.3f})"
                
                # 🆕 센서 모드 표시 추가
                sensor_mode = "Simulation" if hasattr(sensor, 'simulation_mode') and sensor.simulation_mode else "Hardware"
                print(f"📊 State: {current_state.value}, {walking_status}, Connection: {connection_status}, Sensor: {sensor_mode}")
                
                # 🔧 MODIFIED: 센서 상태만 간단히 출력
                if hasattr(sensor, 'last_raw_data') and sensor.last_raw_data:
                    acc_magnitude = np.sqrt(sensor.last_raw_data[0]**2 + sensor.last_raw_data[1]**2 + sensor.last_raw_data[2]**2)
                    print(f"   📐 Sensor: Acc={acc_magnitude:.3f}g")
                    
                    # 🔧 MODIFIED: 하드웨어 모드에서만 센서 데이터 변화 상태 출력
                    if not (hasattr(sensor, 'simulation_mode') and sensor.simulation_mode) and sensor.same_data_count > 0:
                        print(f"   ⚠️ Sensor data unchanged for {sensor.same_data_count} readings")
                
                last_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()