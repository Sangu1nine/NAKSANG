import os
import time
import numpy as np
import pandas as pd
# TensorFlow 대신 TFLite 런타임 사용
import tflite_runtime.interpreter as tflite
from collections import deque
import threading
import signal
import sys

# 하드웨어 센서 사용 가능한 경우 SMBus 가져오기
try:
    import smbus
    SENSOR_AVAILABLE = True
except ImportError:
    print("SMBus 라이브러리를 가져올 수 없습니다. 센서 에뮬레이션 모드로 실행합니다.")
    SENSOR_AVAILABLE = False

# MPU6050 I2C 설정
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# 모델 및 데이터 설정
MODEL_PATH = '/home/pi/Final_Project/data/models/tflite_model/fall_detection_method1.tflite'
SCALER_MEAN_PATH = '/home/pi/Final_Project/data/models/tflite_model/extracted_scaler_mean.npy'
SCALER_SCALE_PATH = '/home/pi/Final_Project/data/models/tflite_model/extracted_scaler_scale.npy'
TEST_DATA_PATH = '/home/pi/Final_Project/data/test'  # 테스트 데이터 경로 (시뮬레이션에 사용)
SEQ_LENGTH = 50  # 시퀀스 길이 
N_FEATURES = 9    # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
SAMPLING_RATE = 100  # Hz (데이터셋의 샘플링 레이트에 맞춤)

# CSV 로깅 설정
LOG_DATA = True
LOG_DIR = 'logs'
LOG_FILE = 'imu_data.csv'
LOG_INTERVAL = 600  # 10분마다 새 로그 파일

# 시뮬레이션 모드 설정
SIMULATION_MODE = not SENSOR_AVAILABLE

# 실제 MPU6050 센서 클래스
class MPU6050Sensor:
    def __init__(self):
        """실제 IMU 센서 (MPU6050) 초기화 및 I2C 설정"""
        if not SENSOR_AVAILABLE:
            raise ImportError("smbus 라이브러리가 설치되어 있지 않습니다.")
        
        self.bus = smbus.SMBus(1)  # I2C 버스 1 사용
        self.setup_mpu6050()
        self.frame_counter = 0
        print("MPU6050 센서 초기화 완료")
        
    def setup_mpu6050(self):
        """MPU6050 센서 초기 설정"""
        # 전원 관리 설정 - 슬립 모드 해제
        self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)  # 안정화 시간
        
        # 샘플링 속도 설정 - 100Hz (1000 / (1 + 9))
        self.bus.write_byte_data(MPU6050_ADDR, SMPLRT_DIV, 9)
        
        # 디지털 저역통과 필터 설정
        self.bus.write_byte_data(MPU6050_ADDR, CONFIG, 0)
        
        # 자이로스코프 설정 - ±250°/s 범위
        self.bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0)
        
        # 가속도계 설정 - ±2g 범위
        self.bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, 0)
    
    def read_word(self, reg):
        """16비트 워드 읽기 (2바이트)"""
        high = self.bus.read_byte_data(MPU6050_ADDR, reg)
        low = self.bus.read_byte_data(MPU6050_ADDR, reg + 1)
        value = (high << 8) + low
        return value
    
    def read_word_2c(self, reg):
        """2의 보수 값으로 변환"""
        val = self.read_word(reg)
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val
    
    def get_data(self):
        """IMU 센서 데이터 읽기"""
        # 가속도계 데이터 (g 단위로 변환: ±2g 범위에서 16384 LSB/g)
        accel_x = self.read_word_2c(ACCEL_XOUT_H) / 16384.0
        accel_y = self.read_word_2c(ACCEL_XOUT_H + 2) / 16384.0
        accel_z = self.read_word_2c(ACCEL_XOUT_H + 4) / 16384.0
        
        # 자이로스코프 데이터 (°/s 단위로 변환: ±250°/s 범위에서 131 LSB/°/s)
        gyro_x = self.read_word_2c(GYRO_XOUT_H) / 131.0
        gyro_y = self.read_word_2c(GYRO_XOUT_H + 2) / 131.0
        gyro_z = self.read_word_2c(GYRO_XOUT_H + 4) / 131.0
        
        # 오일러 각도 계산 (단순화된 방법)
        accel_xangle = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * 180 / np.pi
        accel_yangle = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
        accel_zangle = np.arctan2(accel_z, np.sqrt(accel_x**2 + accel_y**2)) * 180 / np.pi
        
        # 프레임 카운터 증가
        self.frame_counter += 1
        
        # 원본 데이터셋 구조와 일치하는 데이터 반환
        return {
            'TimeStamp(s)': time.time(),
            'FrameCounter': self.frame_counter,
            'AccX': accel_x,
            'AccY': accel_y,
            'AccZ': accel_z,
            'GyrX': gyro_x,
            'GyrY': gyro_y,
            'GyrZ': gyro_z,
            'EulerX': accel_xangle,
            'EulerY': accel_yangle,
            'EulerZ': accel_zangle
        }

# 시뮬레이션된 센서 클래스
class SimulatedSensor:
    def __init__(self, data_path=TEST_DATA_PATH):
        """시뮬레이션된 IMU 센서 초기화"""
        print("시뮬레이션된 IMU 센서 초기화 중...")
        self.frame_counter = 0
        
        # 테스트 데이터 로드 시도
        try:
            X_test = np.load(os.path.join(data_path, 'X_test.npy'))
            y_test = np.load(os.path.join(data_path, 'y_test.npy'))
            print(f"시뮬레이션을 위한 테스트 데이터 로드 완료: {X_test.shape[0]}개 샘플")
            
            # 시뮬레이션용 데이터 준비
            self.sim_data = []
            for i in range(min(100, X_test.shape[0])):  # 처음 100개 샘플만 사용
                for j in range(X_test.shape[1]):  # 각 시간 단계
                    sample = {
                        'TimeStamp(s)': time.time() + j * 0.01,  # 100Hz 가정
                        'FrameCounter': self.frame_counter,
                        'AccX': X_test[i, j, 0],
                        'AccY': X_test[i, j, 1],
                        'AccZ': X_test[i, j, 2],
                        'GyrX': X_test[i, j, 3],
                        'GyrY': X_test[i, j, 4],
                        'GyrZ': X_test[i, j, 5],
                        'EulerX': X_test[i, j, 6],
                        'EulerY': X_test[i, j, 7],
                        'EulerZ': X_test[i, j, 8],
                        'Label': y_test[i]
                    }
                    self.sim_data.append(sample)
                    self.frame_counter += 1
            
            print(f"시뮬레이션 데이터 준비 완료: {len(self.sim_data)}개 샘플")
        except Exception as e:
            print(f"테스트 데이터 로드 실패, 더미 데이터 생성: {e}")
            self.sim_data = self._generate_dummy_data(100 * SEQ_LENGTH)  # 더미 데이터 생성
        
        self.data_index = 0
        self.data_length = len(self.sim_data)
    
    def _generate_dummy_data(self, n_samples=5000):
        """시뮬레이션용 더미 데이터 생성"""
        print("더미 데이터 생성 중...")
        dummy_data = []
        
        fall_probability = 0.05  # 5%의 낙상 확률
        
        for i in range(n_samples):
            # 5초마다 낙상 이벤트 생성
            is_fall_sequence = (i // 500) % 20 == 0
            
            if is_fall_sequence and i % 500 >= 450:  # 낙상 시퀀스의 마지막 50 프레임
                # 낙상 이벤트 시뮬레이션 (갑작스러운 가속도 변화)
                acc_magnitude = np.random.uniform(1.5, 3.0)  # g 단위
                gyro_magnitude = np.random.uniform(100, 200)  # °/s 단위
            else:
                # 일반 움직임 시뮬레이션
                acc_magnitude = np.random.uniform(0.8, 1.2)  # g 단위
                gyro_magnitude = np.random.uniform(5, 30)  # °/s 단위
            
            # 랜덤 방향
            acc_direction = np.random.normal(0, 1, 3)
            acc_direction = acc_direction / np.linalg.norm(acc_direction)
            
            gyro_direction = np.random.normal(0, 1, 3)
            gyro_direction = gyro_direction / np.linalg.norm(gyro_direction)
            
            # 가속도 및 자이로스코프 값
            accel_x = acc_direction[0] * acc_magnitude
            accel_y = acc_direction[1] * acc_magnitude
            accel_z = acc_direction[2] * acc_magnitude
            
            gyro_x = gyro_direction[0] * gyro_magnitude
            gyro_y = gyro_direction[1] * gyro_magnitude
            gyro_z = gyro_direction[2] * gyro_magnitude
            
            # 오일러 각도 계산
            euler_x = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * 180 / np.pi
            euler_y = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
            euler_z = np.arctan2(accel_z, np.sqrt(accel_x**2 + accel_y**2)) * 180 / np.pi
            
            # 샘플 데이터
            sample = {
                'TimeStamp(s)': time.time() + (i * 0.01),
                'FrameCounter': self.frame_counter,
                'AccX': accel_x,
                'AccY': accel_y,
                'AccZ': accel_z,
                'GyrX': gyro_x,
                'GyrY': gyro_y,
                'GyrZ': gyro_z,
                'EulerX': euler_x,
                'EulerY': euler_y,
                'EulerZ': euler_z,
                'Label': 1 if is_fall_sequence and i % 500 >= 450 else 0
            }
            
            dummy_data.append(sample)
            self.frame_counter += 1
        
        print(f"더미 데이터 생성 완료: {len(dummy_data)}개 샘플")
        return dummy_data
    
    def get_data(self):
        """다음 시뮬레이션 데이터 가져오기"""
        if self.data_index >= self.data_length:
            self.data_index = 0
            print("시뮬레이션 데이터를 처음부터 다시 사용합니다.")
        
        data = self.sim_data[self.data_index]
        self.data_index += 1
        
        # 현재 타임스탬프로 업데이트
        data['TimeStamp(s)'] = time.time()
        
        return data

# 데이터 로깅 클래스
class DataLogger:
    def __init__(self, log_dir=LOG_DIR, log_file=LOG_FILE, interval=LOG_INTERVAL):
        """데이터 로깅 클래스"""
        self.log_dir = log_dir
        self.log_file = log_file
        self.interval = interval  # 로그 파일 변경 간격(초)
        self.start_time = time.time()
        self.log_count = 0
        self.header_written = False
        
        # 로그 디렉터리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_file = self._get_log_filename()
        
    def _get_log_filename(self):
        """타임스탬프가 포함된 로그 파일 이름 생성"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.log_file)
        return os.path.join(self.log_dir, f"{base}_{timestamp}{ext}")
    
    def log_data(self, data):
        """센서 데이터를 CSV 파일에 기록"""
        # 일정 시간마다 새 로그 파일 생성
        current_time = time.time()
        if current_time - self.start_time > self.interval:
            self.start_time = current_time
            self.current_file = self._get_log_filename()
            self.header_written = False
        
        # 데이터를 DataFrame으로 변환하여 저장
        df = pd.DataFrame([data])
        
        # 파일이 존재하지 않거나 헤더가 아직 쓰여지지 않은 경우
        if not os.path.exists(self.current_file) or not self.header_written:
            df.to_csv(self.current_file, mode='w', index=False)
            self.header_written = True
        else:
            df.to_csv(self.current_file, mode='a', header=False, index=False)
        
        self.log_count += 1
        if self.log_count % 1000 == 0:
            print(f"로그 데이터 {self.log_count}개 저장됨")

# 텍스트 출력 클래스로 변경 (LED/부저 대신 NAKSANG 출력)
class TextOutputDevice:
    def __init__(self):
        """텍스트 출력 클래스"""
        self.alarm_active = False
        print("텍스트 출력 모드 활성화")
    
    def trigger_naksang(self):
        """NAKSANG 문자열 출력"""
        if not self.alarm_active:
            self.alarm_active = True
            print("\n" + "-" * 30)
            print("!!!!!!! NAKSANG !!!!!!!")
            print("-" * 30 + "\n")
    
    def stop_output(self):
        """출력 중지"""
        if self.alarm_active:
            self.alarm_active = False
            print("출력 중지")
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_output()
        print("출력 장치 정리 완료")

# 낙상 감지기 클래스
class FallDetector:
    def __init__(self, model_path, scaler_mean_path, scaler_scale_path, seq_length=50, n_features=9):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        
        # 출력 장치 초기화
        self.output_device = TextOutputDevice()
        
        # 알람 상태
        self.alarm_active = False
        
        # TFLite 모델 로드
        self.interpreter = self.load_model(model_path)
        
        # 스케일러 파라미터 로드
        try:
            self.scaler_mean = np.load(scaler_mean_path)
            self.scaler_scale = np.load(scaler_scale_path)
            print("스케일러 파라미터 로드 완료")
        except Exception as e:
            print(f"스케일러 파라미터 로드 실패: {e}")
            # 기본값 설정 (데이터 정규화를 건너뛰게 됨)
            self.scaler_mean = np.zeros(n_features)
            self.scaler_scale = np.ones(n_features)
        
        # 입력/출력 텐서 설정
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("모델 로드 완료")
        print(f"입력 형태: {self.input_details[0]['shape']}")
        print(f"출력 형태: {self.output_details[0]['shape']}")
        
        # 데이터 로거 설정
        if LOG_DATA:
            try:
                self.logger = DataLogger()
                print("데이터 로거 초기화 완료")
            except Exception as e:
                print(f"데이터 로거 초기화 실패: {e}")
                self.logger = None
        else:
            self.logger = None
        
        # 통계 정보
        self.stats = {
            'total_samples': 0,
            'detections': 0,
            'last_detection_time': 0
        }
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        try:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def add_data_point(self, data_point):
        """데이터 버퍼에 새 데이터 포인트 추가"""
        # 센서 데이터를 배열로 변환
        data_array = np.array([
            data_point['AccX'], data_point['AccY'], data_point['AccZ'],
            data_point['GyrX'], data_point['GyrY'], data_point['GyrZ'],
            data_point['EulerX'], data_point['EulerY'], data_point['EulerZ']
        ])
        
        # 데이터 버퍼에 추가
        self.data_buffer.append(data_array)
        
        # 통계 업데이트
        self.stats['total_samples'] += 1
        
        # 데이터 로깅 (필요한 경우)
        if LOG_DATA and self.logger:
            self.logger.log_data(data_point)
    
    def normalize_data(self, data):
        """데이터 정규화"""
        try:
            # (seq_length, n_features) -> (seq_length * n_features)
            data_flat = data.reshape(-1, self.n_features)
            
            # 정규화 적용
            data_norm = (data_flat - self.scaler_mean) / self.scaler_scale
            
            # 원래 형태로 복원 (배치 차원 추가)
            return data_norm.reshape(1, self.seq_length, self.n_features)
        except Exception as e:
            print(f"정규화 중 오류: {str(e)}")
            # 오류 시 정규화 생략
            return np.expand_dims(data, axis=0)
    
    def predict(self):
        """낙상 예측 수행"""
        try:
            if len(self.data_buffer) < self.seq_length:
                return None  # 충분한 데이터가 없음
            
            # 버퍼에서 데이터 추출 및 배열로 변환
            data = np.array(list(self.data_buffer))
            
            # 데이터 정규화
            data_norm = self.normalize_data(data)
            
            # 모델 입력 설정
            input_data = data_norm.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 추론 실행
            self.interpreter.invoke()
            
            # 결과 가져오기
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 출력 형태에 따라 다르게 처리
            if output_data.size == 1:
                # 단일 값 출력인 경우
                fall_prob = float(output_data.flatten()[0])
            else:
                # 다차원 출력인 경우
                fall_prob = float(output_data[0][0])
            
            # 예측 결과 (0: 정상, 1: 낙상)
            prediction = 1 if fall_prob >= 0.5 else 0
            
            return {
                'prediction': int(prediction),
                'fall_probability': float(fall_prob)
            }
        except Exception as e:
            print(f"예측 중 오류: {str(e)}")
            return None
    
    def trigger_alarm(self):
        """낙상 감지 시 알람 발생"""
        if not self.alarm_active:
            self.alarm_active = True
            self.output_device.trigger_naksang()  # NAKSANG 출력
            print("낙상 감지! NAKSANG 출력")
            
            # 통계 업데이트
            self.stats['detections'] += 1
            self.stats['last_detection_time'] = time.time()
    
    def stop_alarm(self, channel=None):
        """알람 중지"""
        if self.alarm_active:
            self.alarm_active = False
            self.output_device.stop_output()
            print("알람 중지")
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_alarm()
        self.output_device.cleanup()
        print("낙상 감지기 정리 완료")

def main():
    """메인 함수"""
    print("낙상 감지 시스템 시작")
    
    try:
        # 파일 경로 확인
        for path, description in [
            (MODEL_PATH, "모델 파일"),
            (SCALER_MEAN_PATH, "스케일러 평균 파일"),
            (SCALER_SCALE_PATH, "스케일러 스케일 파일")
        ]:
            if os.path.exists(path):
                print(f"{description} 확인: {path} (존재함)")
            else:
                print(f"경고: {description}를 찾을 수 없습니다: {path}")
                if description == "모델 파일":
                    print("모델 파일이 필요합니다. 프로그램을 종료합니다.")
                    return
        
        # 하드웨어 또는 시뮬레이션 모드에 따라 센서 초기화
        if SIMULATION_MODE:
            sensor = SimulatedSensor()
        else:
            try:
                sensor = MPU6050Sensor()
            except Exception as e:
                print(f"MPU6050 센서 초기화 실패: {e}")
                print("시뮬레이션 모드로 전환합니다.")
                sensor = SimulatedSensor()
        
        # 낙상 감지기 초기화
        detector = FallDetector(
            model_path=MODEL_PATH, 
            scaler_mean_path=SCALER_MEAN_PATH,
            scaler_scale_path=SCALER_SCALE_PATH,
            seq_length=SEQ_LENGTH,
            n_features=N_FEATURES
        )
        
        # Ctrl+C 시그널 핸들러
        def signal_handler(sig, frame):
            print("\n프로그램 종료")
            detector.cleanup()
            
            # 통계 출력
            print("\n=== 감지 통계 ===")
            print(f"총 샘플 수: {detector.stats['total_samples']}")
            print(f"낙상 감지 횟수: {detector.stats['detections']}")
            if detector.stats['total_samples'] > 0:
                print(f"감지율: {detector.stats['detections'] / detector.stats['total_samples']:.2%}")
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 낙상 감지 루프
        print("센서 데이터 수집 중...")
        
        # 초기 데이터 버퍼 채우기
        print(f"초기 데이터 버퍼 채우는 중 ({SEQ_LENGTH} 샘플)...")
        for _ in range(SEQ_LENGTH):
            sensor_data = sensor.get_data()
            detector.add_data_point(sensor_data)
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz 샘플링
        
        print("실시간 낙상 감지 시작")
        
        # 메인 감지 루프
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # 센서 데이터 읽기
            sensor_data = sensor.get_data()
            
            # 디버그 출력 (1초마다)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"현재 가속도: X={sensor_data['AccX']:.2f}, Y={sensor_data['AccY']:.2f}, Z={sensor_data['AccZ']:.2f}")
                last_time = current_time
            
            # 데이터 버퍼에 추가
            detector.add_data_point(sensor_data)
            
            # 낙상 예측
            result = detector.predict()
            
            # 예측 결과가 있고 낙상으로 예측된 경우
            if result and result['prediction'] == 1:
                print(f"낙상 감지! 확률: {result['fall_probability']:.2%}")
                detector.trigger_alarm()
                alarm_start_time = current_time
            
            # 알람이 활성화된 경우, 3초 후 자동으로 끄기
            if detector.alarm_active and (current_time - alarm_start_time >= 3.0):
                detector.stop_alarm()
            
            # 샘플링 속도 유지
            sleep_time = 1.0 / SAMPLING_RATE - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.cleanup()


if __name__ == "__main__":
    main()