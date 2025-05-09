from smbus2 import SMBus
from bitstring import Bits
import math
import time
import pandas as pd
import datetime
import numpy as np
 
bus = SMBus(1)
DEV_ADDR = 0x68
 
register_gyro_xout_h = 0x43
register_gyro_yout_h = 0x45
register_gyro_zout_h = 0x47
sensitive_gyro = 131.0
 
register_accel_xout_h = 0x3B
register_accel_yout_h = 0x3D
register_accel_zout_h = 0x3F
sensitive_accel = 16384.0
 
def read_data(register):
    high = bus.read_byte_data(DEV_ADDR,register)
    low = bus.read_byte_data(DEV_ADDR,register+1)
    val = (high << 8) + low
    return val
 
def twocomplements(val):
    s = Bits(uint=val,length=16)
    return s.int
 
def gyro_dps(val):
    return twocomplements(val)/sensitive_gyro
 
def accel_g(val):
    return twocomplements(val)/sensitive_accel
 
def dist(a,b):
    return math.sqrt((a*a)+(b*b))
 
def get_x_rotation(x,y,z):
    radians = math.atan(x/dist(y,z))
    return radians
 
def get_y_rotation(x,y,z):
    radians = math.atan(y/dist(x,z))
    return radians
 
# 센서 초기화
bus.write_byte_data(DEV_ADDR,0x6B,0b00000000)

# 데이터 프레임 준비
columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
data = []

# 파일명 설정 (현재 시간 기반)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"imu_data_{timestamp}.csv"

print(f"데이터 수집 시작 (100Hz) - 저장 파일: {filename}")
print("Ctrl+C로 수집 종료")

# 초기 시간
start_time = time.time()
sample_count = 0
target_hz = 100  # 목표 샘플링 레이트

try:
    while True:
        # 현재 샘플 시간 계산
        current_time = time.time()
        elapsed = current_time - start_time
        
        # 가속도 데이터 읽기
        accel_x = accel_g(read_data(register_accel_xout_h))
        accel_y = accel_g(read_data(register_accel_yout_h))
        accel_z = accel_g(read_data(register_accel_zout_h))
        
        # 자이로스코프 데이터 읽기
        gyro_x = gyro_dps(read_data(register_gyro_xout_h))
        gyro_y = gyro_dps(read_data(register_gyro_yout_h))
        gyro_z = gyro_dps(read_data(register_gyro_zout_h))
        
        # 데이터 추가
        data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        sample_count += 1
        
        # 1초에 한 번씩 진행 상황 출력
        if sample_count % 100 == 0:
            print(f"샘플 수: {sample_count}, 경과 시간: {elapsed:.2f}초, 샘플링 레이트: {sample_count/elapsed:.2f}Hz")
            print(f"가속도(g): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
            print(f"자이로(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
        
        # 샘플링 레이트 유지 (100Hz = 0.01초 간격)
        next_sample_time = start_time + (sample_count * (1.0 / target_hz))
        sleep_time = next_sample_time - time.time()
        
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n데이터 수집 종료!")
    
    # 데이터프레임 생성
    df = pd.DataFrame(data, columns=columns)
    
    # 인덱스 설정 (시간 기반)
    df.index = np.arange(len(df)) / target_hz
    df.index.name = 'Time(s)'
    
    # CSV 파일로 저장
    df.to_csv(filename)
    print(f"데이터 저장 완료: {filename} (총 {len(df)}개 샘플)")
    
except Exception as e:
    print(f"\n오류 발생: {str(e)}")
finally:
    bus.close()
    print("I2C 버스 닫힘")