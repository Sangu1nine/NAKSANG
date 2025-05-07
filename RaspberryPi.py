import time
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque
import signal
import sys

# MPU6050 I2C 설정
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# 모델 설정
MODEL_PATH = '/home/pi/Final_Project/data/models/tflite_model/fall_detection_method1.tflite'
SEQ_LENGTH = 50  # 시퀀스 길이 
N_FEATURES = 9   # 특징 수 (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ)
SAMPLING_RATE = 100  # Hz

# MPU6050 센서 클래스
class MPU6050Sensor:
    def __init__(self):
        """실제 IMU 센서 (MPU6050) 초기화 및 I2C 설정"""
        try:
            import smbus
            self.bus = smbus.SMBus(1)  # I2C 버스 1 사용
            # 전원 관리 설정 - 슬립 모드 해제
            self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
            time.sleep(0.1)  # 안정화 시간
            self.frame_counter = 0
            print("MPU6050 센서 초기화 완료")
        except ImportError:
            print("smbus 라이브러리를 가져올 수 없습니다.")
            raise
    
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
        
        return np.array([
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z,
            accel_xangle, accel_yangle, accel_zangle
        ])

# 낙상 감지기 클래스
class FallDetector:
    def __init__(self, model_path, seq_length=50, n_features=9):
        """낙상 감지 모델 초기화"""
        self.seq_length = seq_length
        self.n_features = n_features
        self.data_buffer = deque(maxlen=seq_length)
        self.alarm_active = False
        
        # TFLite 모델 로드
        self.interpreter = self.load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("모델 로드 완료")
        print(f"입력 형태: {self.input_details[0]['shape']}")
        print(f"출력 형태: {self.output_details[0]['shape']}")
    
    def load_model(self, model_path):
        """TFLite 모델 로드"""
        try:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def add_data_point(self, data_array):
        """데이터 버퍼에 새 데이터 포인트 추가"""
        self.data_buffer.append(data_array)
    
    def predict(self):
        """낙상 예측 수행"""
        try:
            if len(self.data_buffer) < self.seq_length:
                return None  # 충분한 데이터가 없음
            
            # 버퍼에서 데이터 추출 및 배열로 변환
            data = np.array(list(self.data_buffer))
            
            # 데이터 형태 맞추기 (배치 차원 추가)
            input_data = np.expand_dims(data, axis=0).astype(np.float32)
            
            # 모델 입력 설정
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
        """낙상 감지 시 NAKSANG 출력"""
        if not self.alarm_active:
            self.alarm_active = True
            print("\n" + "-" * 30)
            print("!!!!!!! NAKSANG !!!!!!!")
            print("-" * 30 + "\n")
    
    def stop_alarm(self):
        """알람 중지"""
        if self.alarm_active:
            self.alarm_active = False
            print("알람 중지")

def main():
    """메인 함수"""
    print("낙상 감지 시스템 시작")
    
    try:
        # 센서 초기화
        try:
            sensor = MPU6050Sensor()
        except Exception as e:
            print(f"센서 초기화 실패: {e}")
            print("프로그램을 종료합니다.")
            return
        
        # 낙상 감지기 초기화
        detector = FallDetector(
            model_path=MODEL_PATH,
            seq_length=SEQ_LENGTH,
            n_features=N_FEATURES
        )
        
        # Ctrl+C 시그널 핸들러
        def signal_handler(sig, frame):
            print("\n프로그램 종료")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 낙상 감지 루프
        print("센서 데이터 수집 중...")
        
        # 초기 데이터 버퍼 채우기
        print(f"초기 데이터 버퍼 채우는 중 ({SEQ_LENGTH} 샘플)...")
        for _ in range(SEQ_LENGTH):
            data = sensor.get_data()
            detector.add_data_point(data)
            time.sleep(1.0 / SAMPLING_RATE)  # 100Hz 샘플링
        
        print("실시간 낙상 감지 시작")
        
        # 메인 감지 루프
        last_time = time.time()
        alarm_start_time = 0
        
        while True:
            # 센서 데이터 읽기
            data = sensor.get_data()
            
            # 디버그 출력 (1초마다)
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"현재 가속도: X={data[0]:.2f}, Y={data[1]:.2f}, Z={data[2]:.2f}")
                last_time = current_time
            
            # 데이터 버퍼에 추가
            detector.add_data_point(data)
            
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

if __name__ == "__main__":
    main()