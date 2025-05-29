"""
Improved Raspberry Pi Fall Detection System
- Short code, stable operation
- State-based data transmission (idle=stop, walking=IMU, fall/emergency=event)
- Advanced walking detection with ROC analysis optimization
- Scikit-learn version warning fixed
- INTEGRATED: KFall dataset analysis results applied (F1 Score = 0.641)
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

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    print("SMBus2 library is missing. Run 'pip install smbus2'.")
    SENSOR_AVAILABLE = False

# === Settings ===
DEV_ADDR = 0x68
PWR_MGMT_1 = 0x6B

# Sensor settings
ACCEL_REGISTERS = [0x3B, 0x3D, 0x3F]
GYRO_REGISTERS = [0x43, 0x45, 0x47]
SENSITIVE_ACCEL = 16384.0
SENSITIVE_GYRO = 131.0

# Model settings
MODEL_PATH = 'models/fall_detection.tflite'
SCALERS_DIR = 'scalers'
SEQ_LENGTH = 150
STRIDE = 5
SAMPLING_RATE = 100
SEND_RATE = 10

# Communication settings
WEBSOCKET_SERVER_IP = '192.168.0.177'
WEBSOCKET_SERVER_PORT = 8000
USER_ID = "raspberry_pi_01"
KST = timezone(timedelta(hours=9))

class UserState(Enum):
    """User state definition"""
    DAILY = "Idle"
    WALKING = "Walking"
    FALL = "Fall"
    EMERGENCY = "Emergency"

class AdvancedWalkingDetector:
    """
    Advanced walking detector with ROC analysis optimization
    - KFall dataset analysis results applied (32 subjects, 21,696 windows)
    - F1 Score = 0.641 optimized
    - ROC analysis based thresholds
    """
    def __init__(self, config=None):
        # 데이터 버퍼 (3초 = 300샘플 @ 100Hz) - 더 긴 분석 윈도우
        self.buffer_size = 300
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)

        # 보행 상태
        self.is_walking = False
        self.confidence = 0.0
        
        # 안정성을 위한 상태 추적
        self.walking_start_time = None
        self.last_state_change = 0
        self.consecutive_walking_count = 0
        self.consecutive_idle_count = 0

        # 🔥 ROC 분석 기반 최적화된 설정 시스템
        self.config = self._load_config(config)
        
        print("🎯 Advanced Walking Detector initialized with ROC optimization:")
        print("📊 ROC Analysis Based Thresholds:")
        for key, value in self.config['thresholds'].items():
            print(f"   {key}: {value}")
        print("⚖️ F1 Score Optimized Weights:")
        for key, value in self.config['weights'].items():
            print(f"   {key}: {value}")

    def _load_config(self, config):
        """ROC 분석 기반 최적화된 설정 로드"""
        default_config = {
            'thresholds': {
                # 🎯 ROC 분석 기반 최적화된 임계값 (AUC 기준)
                'acc_mean_min': 0.918,        # acc_range: 0.843 AUC 기반
                'acc_mean_max': 1.122,        
                'acc_std_min': 0.134,         # acc_std: 0.835 AUC 기반
                
                # 보행 주기 관련 임계값 (walking_energy_ratio: 0.833 AUC)
                'step_freq_min': 1.0,
                'step_freq_max': 4.0,
                'regularity_min': 0.869,      # 더 엄격한 규칙성 요구
                
                # 피크 검출 관련 임계값
                'peak_detection_factor': 0.3,
                'peak_window_size': 5,
                
                # 최종 판단 임계값
                'confidence_min': 0.6,
                
                # 안정성 관련 임계값
                'min_duration': 2.0,          # 최소 보행 지속 시간
                'debounce_time': 1.5,         # 상태 변경 디바운스 시간
                'consecutive_threshold': 3     # 연속 감지 임계값
            },
            'weights': {
                # 🎯 F1 스코어 최적화된 가중치 (합계 = 1.0)
                'acc_mean_weight': 0.25,      # 가속도 평균 가중치
                'acc_std_weight': 0.25,       # 가속도 표준편차 가중치  
                'step_freq_weight': 0.35,     # 보행 주기 가중치 (최고)
                'regularity_weight': 0.15     # 규칙성 가중치
            },
            'filtering': {
                # 필터링 관련 파라미터
                'moving_avg_window': 5,
                'min_peaks_required': 2
            }
        }
        
        # 사용자 설정이 있으면 기본값과 병합
        if config:
            for category in default_config:
                if category in config:
                    default_config[category].update(config[category])
        
        # 가중치 합계 검증 및 정규화
        total_weight = sum(default_config['weights'].values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ 가중치 자동 정규화: {total_weight:.3f} → 1.0")
            for key in default_config['weights']:
                default_config['weights'][key] /= total_weight
        
        return default_config

    def add_data(self, acc_x, acc_y, acc_z):
        """센서 데이터 추가 및 보행 감지"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        current_time = time.time()

        self.acc_buffer.append(acc_magnitude)
        self.time_buffer.append(current_time)

        # 충분한 데이터가 있으면 분석
        if len(self.acc_buffer) >= self.buffer_size:
            self._analyze_with_stability()

        return self.is_walking, self.confidence

    def _analyze_with_stability(self):
        """ROC 분석 기반 보행 분석 + 안정성 체크"""
        current_time = time.time()
        acc_data = np.array(self.acc_buffer)
        time_data = np.array(self.time_buffer)

        # 설정 가능한 이동평균 필터링
        window_size = self.config['filtering']['moving_avg_window']
        acc_smooth = np.convolve(acc_data, np.ones(window_size)/window_size, mode='same')

        # 기본 특징 계산
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)

        # 개선된 피크 검출
        peak_factor = self.config['thresholds']['peak_detection_factor']
        peak_window = self.config['thresholds']['peak_window_size']
        threshold = np.mean(acc_smooth) + peak_factor * np.std(acc_smooth)
        
        peaks = []
        for i in range(peak_window, len(acc_smooth) - peak_window):
            if (acc_smooth[i] > threshold and 
                acc_smooth[i] == max(acc_smooth[i-peak_window:i+peak_window+1])):
                peaks.append(i)

        # 보행 주기 및 규칙성 계산
        step_frequency = 0
        regularity = 0
        if len(peaks) >= self.config['filtering']['min_peaks_required']:
            peak_times = time_data[peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                step_frequency = 1.0 / np.mean(intervals)
                # 개선된 규칙성 계산 (표준편차가 작을수록 규칙적)
                regularity = 1.0 / (1.0 + np.std(intervals))

        # ROC 분석 기반 신뢰도 계산 시스템
        confidence_scores = {}
        
        # 1. 가속도 평균 검사 (acc_range 특징 기반)
        if (self.config['thresholds']['acc_mean_min'] <= acc_mean <= 
            self.config['thresholds']['acc_mean_max']):
            confidence_scores['acc_mean'] = self.config['weights']['acc_mean_weight']
        else:
            confidence_scores['acc_mean'] = 0.0
            
        # 2. 가속도 표준편차 검사 (acc_std: 0.835 AUC)
        if acc_std >= self.config['thresholds']['acc_std_min']:
            confidence_scores['acc_std'] = self.config['weights']['acc_std_weight']
        else:
            confidence_scores['acc_std'] = 0.0
            
        # 3. 보행 주기 검사 (walking_energy_ratio: 0.833 AUC)
        if (self.config['thresholds']['step_freq_min'] <= step_frequency <= 
            self.config['thresholds']['step_freq_max']):
            confidence_scores['step_freq'] = self.config['weights']['step_freq_weight']
        else:
            confidence_scores['step_freq'] = 0.0
            
        # 4. 규칙성 검사 (gyr_mean: 0.780 AUC 기반)
        if regularity >= self.config['thresholds']['regularity_min']:
            confidence_scores['regularity'] = self.config['weights']['regularity_weight']
        else:
            confidence_scores['regularity'] = 0.0

        # 최종 신뢰도 계산
        self.confidence = sum(confidence_scores.values())
        
        # 안정성을 위한 상태 변경 로직
        new_walking_state = self.confidence >= self.config['thresholds']['confidence_min']
        
        # 디바운싱: 너무 빠른 상태 변경 방지
        if current_time - self.last_state_change < self.config['thresholds']['debounce_time']:
            return
        
        # 연속 감지 카운트
        if new_walking_state:
            self.consecutive_walking_count += 1
            self.consecutive_idle_count = 0
        else:
            self.consecutive_idle_count += 1
            self.consecutive_walking_count = 0
        
        # 상태 전환 로직
        old_walking = self.is_walking
        consecutive_threshold = self.config['thresholds']['consecutive_threshold']
        
        # 보행 시작: 연속 3회 이상 감지
        if not self.is_walking and self.consecutive_walking_count >= consecutive_threshold:
            self.is_walking = True
            self.walking_start_time = current_time
            self.last_state_change = current_time
            print(f"🚶 Advanced Walking started (Confidence: {self.confidence:.3f}, Freq: {step_frequency:.2f}Hz)")
        
        # 보행 종료: 연속 5회 이상 미감지 + 최소 지속시간
        elif (self.is_walking and self.consecutive_idle_count >= 5 and
              self.walking_start_time and 
              current_time - self.walking_start_time >= self.config['thresholds']['min_duration']):
            self.is_walking = False
            self.last_state_change = current_time
            duration = current_time - self.walking_start_time if self.walking_start_time else 0
            print(f"🚶 Advanced Walking stopped (Duration: {duration:.1f}s, Final confidence: {self.confidence:.3f})")
        
        # 디버깅 정보 저장
        self._last_analysis = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'step_frequency': step_frequency,
            'regularity': regularity,
            'peaks_count': len(peaks),
            'confidence_breakdown': confidence_scores
        }

    def get_detailed_status(self):
        """상세 상태 정보 반환 (디버깅용)"""
        status = {
            'walking': self.is_walking,
            'confidence': self.confidence
        }
        if hasattr(self, '_last_analysis'):
            status.update(self._last_analysis)
        return status

    def update_config(self, new_config):
        """실시간 설정 업데이트"""
        self.config = self._load_config(new_config)
        print("✅ Advanced Walking Detector 설정이 업데이트되었습니다.")

class ImprovedStateManager:
    """Improved state manager with better transition logic"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = 10.0
        
        # Improved transition parameters
        self.walking_confirm_time = 3.0    # Need 3 seconds of walking to confirm
        self.idle_confirm_time = 8.0       # Need 8 seconds of no walking to return to idle
        self.pending_walking_start = None
        
        print(f"🔄 Improved State Manager initialized: {self.current_state.value}")

    def update_state(self, is_walking, fall_detected):
        """Update state with improved transition logic"""
        current_time = time.time()
        previous_state = self.current_state

        # Fall detection (highest priority, with cooldown)
        if fall_detected and self._can_detect_fall():
            self.current_state = UserState.FALL
            self.last_fall_time = current_time
            self.state_start_time = current_time
            self.pending_walking_start = None
            print(f"🚨 Fall detected: {previous_state.value} → {self.current_state.value}")
            return True

        # Idle → Walking transition (with confirmation period)
        elif self.current_state == UserState.DAILY and is_walking:
            if self.pending_walking_start is None:
                self.pending_walking_start = current_time
            elif current_time - self.pending_walking_start >= self.walking_confirm_time:
                self.current_state = UserState.WALKING
                self.state_start_time = current_time
                self.pending_walking_start = None
                print(f"🚶 Walking confirmed: {previous_state.value} → {self.current_state.value}")
                return True

        # Cancel pending walking if not walking anymore
        elif self.current_state == UserState.DAILY and not is_walking:
            if self.pending_walking_start:
                self.pending_walking_start = None

        # Walking → Idle transition (with extended idle period)
        elif self.current_state == UserState.WALKING and not is_walking:
            if current_time - self.state_start_time > self.idle_confirm_time:
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                print(f"🏠 Returned to Idle: {previous_state.value} → {self.current_state.value}")
                return True

        # Auto-recovery after fall
        elif self.current_state == UserState.FALL:
            if current_time - self.state_start_time > 5.0:  # Longer fall recovery time
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                print(f"✅ Returned from Fall: {previous_state.value} → {self.current_state.value}")
                return True

        return False

    def _can_detect_fall(self):
        """Check if fall can be detected"""
        if self.last_fall_time is None:
            return True
        return time.time() - self.last_fall_time > self.fall_cooldown

    def should_send_data(self):
        """Should send data (do not send during Idle)"""
        return self.current_state != UserState.DAILY

    def get_state_info(self):
        """Return state info with pending status"""
        info = {
            'state': self.current_state.value,
            'duration': time.time() - self.state_start_time,
            'can_detect_fall': self._can_detect_fall()
        }
        
        if self.pending_walking_start:
            info['pending_walking'] = time.time() - self.pending_walking_start
            
        return info

class SafeDataSender:
    """Safe data sending manager"""
    def __init__(self):
        self.imu_queue = queue.Queue(maxsize=50)
        self.fall_queue = queue.Queue(maxsize=100)
        self.websocket = None
        self.connected = False

    def add_imu_data(self, data):
        """Add IMU data"""
        try:
            self.imu_queue.put_nowait(data)
        except queue.Full:
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(data)
            except queue.Empty:
                pass

    def add_fall_data(self, data):
        """Add fall data"""
        try:
            self.fall_queue.put_nowait(data)
            print(f"🚨 Fall data added to queue!")
        except queue.Full:
            print("❌ Fall data queue is full!")

    async def send_loop(self):
        """Data sending loop"""
        while True:
            try:
                # Priority: fall data
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data, is_fall=True)

                # IMU data
                elif self.connected and not self.imu_queue.empty():
                    imu_data = self.imu_queue.get_nowait()
                    await self._send_data(imu_data, is_fall=False)

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Send loop error: {e}")
                await asyncio.sleep(1)

    async def _send_data(self, data, is_fall=False):
        """Actual data transmission"""
        if not self.websocket:
            if is_fall:
                self.fall_queue.put_nowait(data)
            return

        try:
            json_data = json.dumps(data, ensure_ascii=False)
            await self.websocket.send(json_data)

            if is_fall:
                confidence = data['data'].get('confidence_score', 0)
                print(f"🚨 Fall data sent! Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"Data send failed: {e}")
            if is_fall:
                self.fall_queue.put_nowait(data)

class SimpleSensor:
    """Sensor class with improved scaler handling"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("Sensor library is missing.")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        self.scalers = self._load_scalers()
        print("Sensor initialized.")

    def _load_scalers(self):
        """Load scalers with version compatibility"""
        scalers = {}
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        for feature in features:
            try:
                std_path = os.path.join(SCALERS_DIR, f"{feature}_standard_scaler.pkl")
                minmax_path = os.path.join(SCALERS_DIR, f"{feature}_minmax_scaler.pkl")
                
                # Suppress warnings during loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(std_path, 'rb') as f:
                        scalers[f"{feature}_standard"] = pickle.load(f)
                    with open(minmax_path, 'rb') as f:
                        scalers[f"{feature}_minmax"] = pickle.load(f)
            except Exception as e:
                print(f"Failed to load scaler {feature}: {e}")
        
        return scalers

    def _read_word_2c(self, reg):
        """Read 2's complement value"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val

    def get_data(self):
        """Read and normalize sensor data"""
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

class SimpleFallDetector:
    """Fall detector"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Fall detection model loaded.")

    def add_data(self, data):
        """Add data"""
        self.buffer.append(data)
        self.counter += 1

    def should_predict(self):
        """Check prediction timing"""
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0

    def predict(self):
        """Fall prediction"""
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
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

def create_imu_package(data, user_id, state_info=None):
    """Create IMU data package"""
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
    if state_info:
        package['state_info'] = state_info
    return package

def create_fall_package(user_id, probability, sensor_data, state_info=None):
    """Create fall data package"""
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
    if state_info:
        package['state_info'] = state_info
    return package

async def websocket_handler(data_sender):
    """WebSocket connection handler"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    retry_delay = 1
    
    while True:
        try:
            print(f"Trying WebSocket connection: {url}")
            async with websockets.connect(url) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                retry_delay = 1
                print("✅ WebSocket connected.")
                
                await data_sender.send_loop()
                
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
        
        print(f"Waiting {retry_delay} seconds before reconnect.")
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)

def main():
    """Main function"""
    print("🚀 Advanced Fall Detection System Started (v3.0 - ROC Optimized)")
    print(f"📊 KFall Dataset Integration: F1 Score = 0.641")
    print(f"Current time (KST): {datetime.now(KST).isoformat()}")
    
    # Initialization
    try:
        sensor = SimpleSensor()
        fall_detector = SimpleFallDetector()
        walking_detector = AdvancedWalkingDetector()
        state_manager = ImprovedStateManager()
        data_sender = SafeDataSender()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Graceful exit
    def signal_handler(sig, frame):
        print(f"\nExiting program... (Current state: {state_manager.current_state.value})")
        if not data_sender.fall_queue.empty():
            print(f"Pending fall data: {data_sender.fall_queue.qsize()} left")
            time.sleep(3)
        print("Program terminated.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start WebSocket client
    def start_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(data_sender))
    
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    
    print("🔄 Data acquisition started.")
    
    # Initial buffer fill
    for _ in range(SEQ_LENGTH):
        data = sensor.get_data()
        fall_detector.add_data(data)
        time.sleep(1.0 / SAMPLING_RATE)
    
    print("🎯 Advanced state-based data transmission started.")
    
    # Main loop
    last_print = time.time()
    last_detailed_print = time.time()
    imu_send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # 1. Advanced Walking detection
            is_walking, walk_confidence = walking_detector.add_data(
                data[0], data[1], data[2]
            )
            
            # 2. Fall detection
            fall_detector.add_data(data)
            fall_result = None
            if fall_detector.should_predict():
                fall_result = fall_detector.predict()
            
            fall_detected = fall_result and fall_result['prediction'] == 1
            
            # 3. State update
            state_changed = state_manager.update_state(is_walking, fall_detected)
            
            # 4. Data transmission (state-based)
            current_state = state_manager.current_state
            state_info = state_manager.get_state_info()
            
            # Always send fall data
            if fall_detected:
                print(f"\n🚨 Fall detected! Confidence: {fall_result['probability']:.2%}")
                fall_package = create_fall_package(USER_ID, fall_result['probability'], data, state_info)
                data_sender.add_fall_data(fall_package)
                print("🚨 FALL!")
            
            # Send IMU data only in Walking state, at 10Hz
            elif current_state == UserState.WALKING:
                imu_send_counter += 1
                if imu_send_counter >= (SAMPLING_RATE // SEND_RATE):
                    imu_package = create_imu_package(data, USER_ID, state_info)
                    data_sender.add_imu_data(imu_package)
                    imu_send_counter = 0
            
            # 5. Enhanced debug print every 5 seconds
            if current_time - last_print >= 5.0:
                status_msg = f"\n📊 Advanced System Status:"
                status_msg += f"\n   Current state: {current_state.value} ({state_info['duration']:.1f}s)"
                if 'pending_walking' in state_info:
                    status_msg += f"\n   Pending walking: {state_info['pending_walking']:.1f}s"
                status_msg += f"\n   Walking detected: {'🚶' if is_walking else '🚫'} (ROC Confidence: {walk_confidence:.3f})"
                status_msg += f"\n   Data transmission: {'✅' if state_manager.should_send_data() else '❌ (Idle state)'}"
                status_msg += f"\n   Connection status: {'✅' if data_sender.connected else '❌'}"
                status_msg += f"\n   Accel: X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}"
                print(status_msg)
                last_print = current_time
            
            # 6. Detailed walking analysis every 15 seconds (when walking)
            if (current_time - last_detailed_print >= 15.0 and is_walking):
                detailed_status = walking_detector.get_detailed_status()
                if 'confidence_breakdown' in detailed_status:
                    breakdown = detailed_status['confidence_breakdown']
                    print(f"\n🔍 Advanced Walking Analysis:")
                    print(f"   📈 Confidence Breakdown:")
                    print(f"      • Acc Mean: {breakdown.get('acc_mean', 0):.3f} (Range: {detailed_status.get('acc_mean', 0):.3f})")
                    print(f"      • Acc Std: {breakdown.get('acc_std', 0):.3f} (Value: {detailed_status.get('acc_std', 0):.3f})")
                    print(f"      • Step Freq: {breakdown.get('step_freq', 0):.3f} (Value: {detailed_status.get('step_frequency', 0):.2f}Hz)")
                    print(f"      • Regularity: {breakdown.get('regularity', 0):.3f} (Value: {detailed_status.get('regularity', 0):.3f})")
                    print(f"   🎯 Total Confidence: {detailed_status.get('confidence', 0):.3f}")
                    print(f"   👣 Peaks Detected: {detailed_status.get('peaks_count', 0)}")
                last_detailed_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()