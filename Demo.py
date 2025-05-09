from smbus2 import SMBus
from bitstring import Bits
import math
import time
 
bus = SMBus(1)  # I2C 버스 초기화 (라즈베리파이 버전 2 이상은 1번 사용)
DEV_ADDR = 0x68  # MPU6050 I2C 장치 주소
 
# 자이로스코프 레지스터 주소 정의
register_gyro_xout_h = 0x43  # X축 자이로스코프 데이터 상위 바이트 레지스터
register_gyro_yout_h = 0x45  # Y축 자이로스코프 데이터 상위 바이트 레지스터
register_gyro_zout_h = 0x47  # Z축 자이로스코프 데이터 상위 바이트 레지스터
sensitive_gyro = 131.0  # 자이로스코프 감도 - ±250°/s 범위일 때 131 LSB/°/s
 
# 가속도계 레지스터 주소 정의
register_accel_xout_h = 0x3B  # X축 가속도계 데이터 상위 바이트 레지스터
register_accel_yout_h = 0x3D  # Y축 가속도계 데이터 상위 바이트 레지스터
register_accel_zout_h = 0x3F  # Z축 가속도계 데이터 상위 바이트 레지스터
sensitive_accel = 16384.0  # 가속도계 감도 - ±2g 범위일 때 16384 LSB/g
 
def read_data(register):
    """
    지정된 레지스터에서 16비트(2바이트) 데이터를 읽어옴
    
    Args:
        register: 데이터를 읽을 레지스터 주소
        
    Returns:
        16비트 데이터 값
    """
    high = bus.read_byte_data(DEV_ADDR, register)  # 상위 바이트 읽기
    low = bus.read_byte_data(DEV_ADDR, register+1)  # 하위 바이트 읽기
    val = (high << 8) + low  # 두 바이트 결합하여 16비트 값 생성
    return val
 
def twocomplements(val):
    """
    16비트 2의 보수 값을 부호 있는 정수로 변환
    
    Args:
        val: 부호 없는 16비트 정수 값
        
    Returns:
        부호 있는 정수 값
    """
    s = Bits(uint=val, length=16)
    return s.int
 
def gyro_dps(val):
    """
    자이로스코프 raw 값을 도/초(degrees per second) 단위로 변환
    
    Args:
        val: 자이로스코프 raw 데이터
        
    Returns:
        도/초 단위 각속도 값
    """
    return twocomplements(val) / sensitive_gyro
 
def accel_g(val):
    """
    가속도계 raw 값을 g 단위(중력가속도)로 변환
    
    Args:
        val: 가속도계 raw 데이터
        
    Returns:
        g 단위 가속도 값
    """
    return twocomplements(val) / sensitive_accel
 
def dist(a, b):
    """
    두 값의 유클리드 거리(피타고라스) 계산
    
    Args:
        a: 첫 번째 값
        b: 두 번째 값
        
    Returns:
        두 값의 유클리드 거리
    """
    return math.sqrt((a*a) + (b*b))
 
def get_x_rotation(x, y, z):
    """
    가속도계 데이터로부터 X축 기준 회전각(라디안) 계산
    
    Args:
        x, y, z: 각 축의 가속도 값
        
    Returns:
        X축 기준 회전각(라디안)
    """
    radians = math.atan(x / dist(y, z))
    return radians
 
def get_y_rotation(x, y, z):
    """
    가속도계 데이터로부터 Y축 기준 회전각(라디안) 계산
    
    Args:
        x, y, z: 각 축의 가속도 값
        
    Returns:
        Y축 기준 회전각(라디안)
    """
    radians = math.atan(y / dist(x, z))
    return radians
 
# MPU6050 초기화 - 파워 매니지먼트 레지스터(0x6B)에 0을 씀으로써 슬립 모드 해제
bus.write_byte_data(DEV_ADDR, 0x6B, 0b00000000)
 
try:
    while True:
        # 가속도계 데이터 읽기
        x = read_data(register_accel_xout_h)
        y = read_data(register_accel_yout_h)
        z = read_data(register_accel_zout_h)
        
        # X축, Y축 기준 회전각 계산
        aX = get_x_rotation(accel_g(x), accel_g(y), accel_g(z))
        aY = get_y_rotation(accel_g(x), accel_g(y), accel_g(z))
        data = str(aX) + ' , ' + str(aY) + '$'
 
        # 낙상 감지 - 임계값 비교 방식
        if aX > 0.7 or aX < -0.5 or aY < -0.5 or aY > 0.5:
            print("NAKSANG")
        
        print(data)
        time.sleep(0.3)  # 0.3초 대기 (약 3.33Hz 샘플링 속도)
except KeyboardInterrupt:
    print("\nInterrupted!")
except:
    print("\nClosing socket")
finally:
    bus.close()  # 프로그램 종료 시 I2C 버스 닫기
