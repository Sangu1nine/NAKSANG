"""
개선된 라즈베리파이 낙상 감지 시스템 with 상태 관리
- 상태 기반 데이터 전송 (일상상태에서는 전송 X)
- 걷기 감지 통합
- 중복 낙상 감지 방지
- 응급상황 자동 판정
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
import math

try:
    from smbus2 import SMBus
    SENSOR_AVAILABLE = True
except ImportError:
    print("SMBus2 라이브러리가 없습니다. 'pip install smbus2' 실행하세요.")
    SENSOR_AVAILABLE = False

# === 기본 설정 ===
DEV_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_REGISTERS = [0x3B, 0x3D, 0x3F]
GYRO_REGISTERS = [0x43, 0x45, 0x47]
SENSITIVE_ACCEL = 16384.0
SENSITIVE_GYRO = 131.0

# 모델 및 파일 경로
MODEL_PATH = 'models/fall_detection.tflite'
SCALERS_DIR = 'scalers'
SEQ_LENGTH = 150
STRIDE = 5
SAMPLING_RATE = 100
SEND_RATE = 10

# 통신 설정
WEBSOCKET_SERVER_IP = '192.168.0.177'
WEBSOCKET_SERVER_PORT = 8000
USER_ID = "raspberry_pi_01"
KST = timezone(timedelta(hours=9))

class UserState(Enum):
    """사용자 상태 정의"""
    DAILY = "일상"
    WALKING = "걷기"
    FALL = "낙상"
    EMERGENCY = "응급"

class WalkingDetector:
    """보행 감지기 (기존 코드 통합)"""
    def __init__(self):
        self.buffer_size = 200  # 2초
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.is_walking = False
        self.confidence = 0.0
        
        self.thresholds = {
            'acc_mean_min': 1.022,
            'acc_mean_max': 1.126,
            'acc_std_min': 0.208,
            'step_freq_min': 1.0,
            'step_freq_max': 4.0,
            'regularity_min': 0.417,
            'confidence_min': 0.6
        }
        print("🎯 보행 감지기 초기화 완료")

    def add_data(self, acc_x, acc_y, acc_z, timestamp):
        """센서 데이터 추가 및 분석"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        self.acc_buffer.append(acc_magnitude)
        self.time_buffer.append(timestamp)

        if len(self.acc_buffer) >= self.buffer_size:
            self._analyze()
        return self.is_walking, self.confidence

    def _analyze(self):
        """보행 분석 로직"""
        acc_data = np.array(self.acc_buffer)
        time_data = np.array(self.time_buffer)

        # 이동평균 필터링
        acc_smooth = np.convolve(acc_data, np.ones(5)/5, mode='same')

        # 기본 특징
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)

        # 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = []
        for i in range(5, len(acc_smooth)-5):
            if acc_smooth[i] > threshold and acc_smooth[i] == max(acc_smooth[i-5:i+6]):
                peaks.append(i)

        # 보행 주기 계산
        step_frequency = 0
        if len(peaks) > 1:
            peak_times = time_data[peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                step_frequency = 1.0 / np.mean(intervals)

        # 신뢰도 계산
        confidence = 0.0
        if self.thresholds['acc_mean_min'] <= acc_mean <= self.thresholds['acc_mean_max']:
            confidence += 0.3
        if acc_std >= self.thresholds['acc_std_min']:
            confidence += 0.3
        if self.thresholds['step_freq_min'] <= step_frequency <= self.thresholds['step_freq_max']:
            confidence += 0.4

        self.confidence = confidence
        self.is_walking = confidence >= self.thresholds['confidence_min']

class StateManager:
    """🆕 상태 관리자 - 핵심 개선사항"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = 10.0  # 낙상 후 10초 쿨다운
        self.emergency_criteria = {
            'min_lying_duration': 15.0,  # 15초 이상 엎어져 있으면 응급
            'max_movement_threshold': 0.05  # 움직임 임계값
        }
        self.lying_start_time = None
        self.movement_buffer = deque(maxlen=150)  # 1.5초분 움직임 데이터
        
        print(f"🔄 상태 관리자 초기화: {self.current_state.value}")

    def update_state(self, is_walking, fall_detected, sensor_data):
        """상태 업데이트 로직"""
        current_time = time.time()
        previous_state = self.current_state
        state_changed = False

        # 1. 일상 → 걷기
        if self.current_state == UserState.DAILY and is_walking:
            self.current_state = UserState.WALKING
            self.state_start_time = current_time
            state_changed = True
            print(f"🚶 상태 전환: {previous_state.value} → {self.current_state.value}")

        # 2. 걷기 → 일상 (보행 중단)
        elif self.current_state == UserState.WALKING and not is_walking:
            # 5초 이상 보행이 감지되지 않으면 일상으로 복귀
            if current_time - self.state_start_time > 5.0:
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                state_changed = True
                print(f"🏠 상태 전환: {previous_state.value} → {self.current_state.value}")

        # 3. 낙상 감지 (쿨다운 체크)
        elif fall_detected and self._can_detect_fall():
            self.current_state = UserState.FALL
            self.last_fall_time = current_time
            self.state_start_time = current_time
            self.lying_start_time = current_time
            state_changed = True
            print(f"🚨 상태 전환: {previous_state.value} → {self.current_state.value}")

        # 4. 낙상 → 응급 (움직임 없음)
        elif self.current_state == UserState.FALL:
            if self._is_lying_still(sensor_data, current_time):
                lying_duration = current_time - self.lying_start_time
                if lying_duration >= self.emergency_criteria['min_lying_duration']:
                    self.current_state = UserState.EMERGENCY
                    state_changed = True
                    print(f"🚨 응급상황 판정: {lying_duration:.1f}초간 움직임 없음")
            else:
                # 움직임 감지되면 일상으로 복귀
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                self.lying_start_time = None
                state_changed = True
                print(f"✅ 낙상 후 회복: {previous_state.value} → {self.current_state.value}")

        # 5. 응급 → 일상 (움직임 감지)
        elif self.current_state == UserState.EMERGENCY:
            if not self._is_lying_still(sensor_data, current_time):
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                self.lying_start_time = None
                state_changed = True
                print(f"✅ 응급상황 해제: {previous_state.value} → {self.current_state.value}")

        return state_changed

    def _can_detect_fall(self):
        """낙상 감지 가능 여부 (쿨다운 체크)"""
        if self.last_fall_time is None:
            return True
        return time.time() - self.last_fall_time > self.fall_cooldown

    def _is_lying_still(self, sensor_data, current_time):
        """엎어진 상태 (움직임 없음) 판정"""
        # 센서 데이터로부터 움직임 계산
        movement = np.sqrt(sensor_data[0]**2 + sensor_data[1]**2 + sensor_data[2]**2)
        self.movement_buffer.append(movement)

        if len(self.movement_buffer) < 50:  # 최소 0.5초분 데이터 필요
            return True

        # 최근 1.5초간 움직임 표준편차로 판정
        movement_std = np.std(list(self.movement_buffer))
        return movement_std < self.emergency_criteria['max_movement_threshold']

    def should_send_data(self):
        """데이터 전송 여부 결정"""
        return self.current_state != UserState.DAILY

    def get_state_info(self):
        """상태 정보 반환"""
        current_time = time.time()
        return {
            'state': self.current_state.value,
            'duration': current_time - self.state_start_time,
            'can_detect_fall': self._can_detect_fall(),
            'cooldown_remaining': max(0, self.fall_cooldown - (current_time - (self.last_fall_time or 0)))
        }

class EnhancedDataSender:
    """개선된 데이터 전송 관리자"""
    def __init__(self):
        self.imu_queue = queue.Queue(maxsize=100)
        self.fall_queue = queue.Queue(maxsize=100)
        self.websocket = None
        self.connected = False
        self.state_manager = None  # StateManager 참조용

    def set_state_manager(self, state_manager):
        """StateManager 참조 설정"""
        self.state_manager = state_manager

    def add_imu_data(self, data):
        """IMU 데이터 추가 (상태 체크)"""
        if self.state_manager and self.state_manager.should_send_data():
            try:
                self.imu_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.imu_queue.get_nowait()
                    self.imu_queue.put_nowait(data)
                except queue.Empty:
                    pass

    def add_fall_data(self, data):
        """낙상 데이터 추가 (항상 전송)"""
        try:
            self.fall_queue.put_nowait(data)
            print(f"🚨 낙상 데이터 큐 추가!")
        except queue.Full:
            print("❌ 낙상 데이터 큐 가득참!")

    async def send_loop(self):
        """데이터 전송 루프"""
        while True:
            try:
                # 낙상 데이터 우선 처리
                if not self.fall_queue.empty():
                    fall_data = self.fall_queue.get_nowait()
                    await self._send_data(fall_data, is_fall=True)

                # IMU 데이터 처리 (연결되고 걷기 상태일 때만)
                elif self.connected and not self.imu_queue.empty():
                    imu_data = self.imu_queue.get_nowait()
                    await self._send_data(imu_data, is_fall=False)

                await asyncio.sleep(0.1)  # 10Hz 전송

            except Exception as e:
                print(f"전송 루프 오류: {e}")
                await asyncio.sleep(1)

    async def _send_data(self, data, is_fall=False):
        """실제 데이터 전송"""
        if not self.websocket:
            if is_fall:
                self.fall_queue.put_nowait(data)
            return

        try:
            # 상태 정보 추가
            if self.state_manager:
                data['state_info'] = self.state_manager.get_state_info()

            json_data = json.dumps(data, ensure_ascii=False)
            await self.websocket.send(json_data)

            if is_fall:
                confidence = data['data'].get('confidence_score', 0)
                print(f"🚨 낙상 데이터 전송 성공! 신뢰도: {confidence:.2%}")

        except Exception as e:
            print(f"데이터 전송 실패: {e}")
            if is_fall:
                self.fall_queue.put_nowait(data)

class SimpleSensor:
    """센서 클래스 (기존 유지)"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("센서 라이브러리 없음")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        self.scalers = self._load_scalers()
        print("센서 초기화 완료")

    def _load_scalers(self):
        """스케일러 로드"""
        scalers = {}
        features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        
        for feature in features:
            try:
                std_path = os.path.join(SCALERS_DIR, f"{feature}_standard_scaler.pkl")
                minmax_path = os.path.join(SCALERS_DIR, f"{feature}_minmax_scaler.pkl")
                
                with open(std_path, 'rb') as f:
                    scalers[f"{feature}_standard"] = pickle.load(f)
                with open(minmax_path, 'rb') as f:
                    scalers[f"{feature}_minmax"] = pickle.load(f)
            except Exception as e:
                print(f"스케일러 로드 실패 {feature}: {e}")
        
        return scalers

    def _read_word_2c(self, reg):
        """2의 보수 값 읽기"""
        high = self.bus.read_byte_data(DEV_ADDR, reg)
        low = self.bus.read_byte_data(DEV_ADDR, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val

    def get_data(self):
        """센서 데이터 읽기 및 정규화"""
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
    """낙상 감지기 (기존 유지)"""
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LENGTH)
        self.counter = 0
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("낙상 감지 모델 로드 완료")

    def add_data(self, data):
        """데이터 추가"""
        self.buffer.append(data)
        self.counter += 1

    def should_predict(self):
        """예측 시점 확인"""
        return len(self.buffer) == SEQ_LENGTH and self.counter % STRIDE == 0

    def predict(self):
        """낙상 예측"""
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
            print(f"예측 오류: {e}")
            return None

def create_imu_package(data, user_id, state_info=None):
    """IMU 데이터 패키지 생성"""
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
    """낙상 데이터 패키지 생성"""
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
    """WebSocket 연결 관리"""
    url = f"ws://{WEBSOCKET_SERVER_IP}:{WEBSOCKET_SERVER_PORT}/ws/{USER_ID}"
    retry_delay = 1
    
    while True:
        try:
            print(f"WebSocket 연결 시도: {url}")
            async with websockets.connect(url) as websocket:
                data_sender.websocket = websocket
                data_sender.connected = True
                retry_delay = 1
                print("✅ WebSocket 연결 성공")
                
                await data_sender.send_loop()
                
        except Exception as e:
            print(f"WebSocket 연결 실패: {e}")
        finally:
            data_sender.websocket = None
            data_sender.connected = False
        
        print(f"재연결 대기: {retry_delay}초")
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)

def main():
    """개선된 메인 함수"""
    print("🚀 개선된 낙상 감지 시스템 시작")
    print(f"현재 시간 (KST): {datetime.now(KST).isoformat()}")
    
    # 초기화
    try:
        sensor = SimpleSensor()
        fall_detector = SimpleFallDetector()
        walking_detector = WalkingDetector()
        state_manager = StateManager()  # 🆕 상태 관리자
        data_sender = EnhancedDataSender()
        data_sender.set_state_manager(state_manager)  # 🆕 상태 관리자 연결
    except Exception as e:
        print(f"초기화 실패: {e}")
        return

    # 종료 처리
    def signal_handler(sig, frame):
        print(f"\n프로그램 종료 중... (현재 상태: {state_manager.current_state.value})")
        if not data_sender.fall_queue.empty():
            print(f"남은 낙상 데이터: {data_sender.fall_queue.qsize()}개")
            time.sleep(3)
        print("프로그램 종료")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # WebSocket 클라이언트 시작
    def start_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(data_sender))
    
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    
    print("🔄 데이터 수집 시작")
    
    # 초기 버퍼 채우기
    for _ in range(SEQ_LENGTH):
        data = sensor.get_data()
        fall_detector.add_data(data)
        time.sleep(1.0 / SAMPLING_RATE)
    
    print("🎯 상태 기반 시스템 시작")
    
    # 메인 루프
    last_print = time.time()
    imu_send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # 1. 보행 감지
            is_walking, walk_confidence = walking_detector.add_data(
                data[0], data[1], data[2], current_time
            )
            
            # 2. 낙상 감지 (상태 관리자에서 허용할 때만)
            fall_detector.add_data(data)
            fall_result = None
            if fall_detector.should_predict() and state_manager._can_detect_fall():
                fall_result = fall_detector.predict()
            
            fall_detected = fall_result and fall_result['prediction'] == 1
            
            # 3. 🆕 상태 업데이트
            state_changed = state_manager.update_state(is_walking, fall_detected, data)
            
            # 4. 데이터 전송 (상태에 따라)
            current_state = state_manager.current_state
            
            # 낙상 데이터 전송 (항상)
            if fall_detected:
                print(f"\n🚨 낙상 감지! 신뢰도: {fall_result['probability']:.2%}")
                fall_package = create_fall_package(
                    USER_ID, fall_result['probability'], data, 
                    state_manager.get_state_info()
                )
                data_sender.add_fall_data(fall_package)
                print("🚨 NAKSANG!")
            
            # IMU 데이터 전송 (걷기 상태일 때만, 10Hz)
            elif current_state == UserState.WALKING:
                imu_send_counter += 1
                if imu_send_counter >= (SAMPLING_RATE // SEND_RATE):  # 10Hz로 송신
                    imu_package = create_imu_package(data, USER_ID, state_manager.get_state_info())
                    data_sender.add_imu_data(imu_package)
                    imu_send_counter = 0
            
            # 5. 디버그 출력 (5초마다)
            if current_time - last_print >= 5.0:
                state_info = state_manager.get_state_info()
                print(f"\n📊 시스템 상태:")
                print(f"   현재 상태: {current_state.value} ({state_info['duration']:.1f}초)")
                print(f"   보행 감지: {'🚶' if is_walking else '🚫'} (신뢰도: {walk_confidence:.2f})")
                print(f"   낙상 감지: {'✅' if state_info['can_detect_fall'] else f'❌ ({state_info["cooldown_remaining"]:.1f}초 남음)'}")
                print(f"   데이터 전송: {'✅' if state_manager.should_send_data() else '❌ (일상상태)'}")
                print(f"   연결상태: {'✅' if data_sender.connected else '❌'}")
                print(f"   가속도: X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                last_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"메인 루프 오류: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()