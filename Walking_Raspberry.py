"""
Improved Raspberry Pi Fall Detection System
- Short code, stable operation
- State-based data transmission (idle=stop, walking=IMU, fall/emergency=event)
- Simple walking detection added
- Complex state logic removed
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

class SimpleWalkingDetector:
    """Simple walking detector"""
    def __init__(self):
        self.buffer_size = 200  # 2 seconds
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.is_walking = False
        self.confidence = 0.0
        self.last_walking_time = 0
        
        # Simple thresholds
        self.thresholds = {
            'acc_std_min': 0.15,     # Minimum acceleration variance
            'step_freq_min': 1.0,    # Minimum walking frequency
            'step_freq_max': 4.0,    # Maximum walking frequency
            'confidence_min': 0.6    # Confidence threshold
        }
        print("🚶 Simple Walking Detector initialized.")

    def add_data(self, acc_x, acc_y, acc_z):
        """Add accelerometer data and detect walking"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        self.acc_buffer.append(acc_magnitude)

        if len(self.acc_buffer) >= self.buffer_size:
            self._analyze()
        
        return self.is_walking, self.confidence

    def _analyze(self):
        """Simple walking analysis"""
        acc_data = np.array(self.acc_buffer)
        acc_std = np.std(acc_data)
        threshold = np.mean(acc_data) + 0.2 * np.std(acc_data)
        peaks = []
        for i in range(10, len(acc_data)-10):
            if acc_data[i] > threshold and acc_data[i] == max(acc_data[i-5:i+6]):
                peaks.append(i)
        step_frequency = 0
        if len(peaks) > 1:
            step_frequency = len(peaks) / 2.0  # 2 seconds buffer

        confidence = 0.0
        if acc_std >= self.thresholds['acc_std_min']:
            confidence += 0.5
        if self.thresholds['step_freq_min'] <= step_frequency <= self.thresholds['step_freq_max']:
            confidence += 0.5

        self.confidence = confidence
        old_walking = self.is_walking
        self.is_walking = confidence >= self.thresholds['confidence_min']
        if old_walking != self.is_walking:
            status = "started" if self.is_walking else "stopped"
            print(f"🚶 Walking {status} (Confidence: {confidence:.2f})")

class SimpleStateManager:
    """Simple state manager"""
    def __init__(self):
        self.current_state = UserState.DAILY
        self.state_start_time = time.time()
        self.last_fall_time = None
        self.fall_cooldown = 10.0  # 10 seconds cooldown after fall
        print(f"🔄 State Manager initialized: {self.current_state.value}")

    def update_state(self, is_walking, fall_detected):
        """Update state (simple logic)"""
        current_time = time.time()
        previous_state = self.current_state

        # Fall detection (with cooldown)
        if fall_detected and self._can_detect_fall():
            self.current_state = UserState.FALL
            self.last_fall_time = current_time
            self.state_start_time = current_time
            print(f"🚨 Fall detected: {previous_state.value} → {self.current_state.value}")
            return True

        # Idle ↔ Walking transition
        elif self.current_state == UserState.DAILY and is_walking:
            self.current_state = UserState.WALKING
            self.state_start_time = current_time
            print(f"🚶 Walking started: {previous_state.value} → {self.current_state.value}")
            return True

        elif self.current_state == UserState.WALKING and not is_walking:
            # Return to idle after 5 seconds of inactivity
            if current_time - self.state_start_time > 5.0:
                self.current_state = UserState.DAILY
                self.state_start_time = current_time
                print(f"🏠 Returned to Idle: {previous_state.value} → {self.current_state.value}")
                return True

        # Auto-recovery after fall (emergency handled at server)
        elif self.current_state == UserState.FALL:
            if current_time - self.state_start_time > 3.0:
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
        """Return state info"""
        return {
            'state': self.current_state.value,
            'duration': time.time() - self.state_start_time,
            'can_detect_fall': self._can_detect_fall()
        }

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
    """Sensor class"""
    def __init__(self):
        if not SENSOR_AVAILABLE:
            raise ImportError("Sensor library is missing.")
        
        self.bus = SMBus(1)
        self.bus.write_byte_data(DEV_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        self.scalers = self._load_scalers()
        print("Sensor initialized.")

    def _load_scalers(self):
        """Load scalers"""
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
    print("🚀 Improved Fall Detection System Started.")
    print(f"Current time (KST): {datetime.now(KST).isoformat()}")
    
    # Initialization
    try:
        sensor = SimpleSensor()
        fall_detector = SimpleFallDetector()
        walking_detector = SimpleWalkingDetector()
        state_manager = SimpleStateManager()
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
    
    print("🎯 State-based data transmission started.")
    
    # Main loop
    last_print = time.time()
    imu_send_counter = 0
    
    while True:
        try:
            data = sensor.get_data()
            current_time = time.time()
            
            # 1. Walking detection
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
            
            # 5. Debug print every 5 seconds
            if current_time - last_print >= 5.0:
                print(f"\n📊 System Status:")
                print(f"   Current state: {current_state.value} ({state_info['duration']:.1f}s)")
                print(f"   Walking detected: {'🚶' if is_walking else '🚫'} (Confidence: {walk_confidence:.2f})")
                print(f"   Data transmission: {'✅' if state_manager.should_send_data() else '❌ (Idle state)'}")
                print(f"   Connection status: {'✅' if data_sender.connected else '❌'}")
                print(f"   Accel: X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                last_print = current_time
            
            time.sleep(1.0 / SAMPLING_RATE)
            
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
