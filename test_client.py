import socket
import json
import time
import math

# 테스트용 IMU 클라이언트
SERVER_IP = '127.0.0.1'
SERVER_PORT = 5000

def generate_test_data(timestamp):
    """보행 패턴을 시뮬레이션하는 테스트 데이터 생성"""
    # 보행 주기: 약 1.2초 (0.83Hz)
    gait_freq = 0.83
    
    # 기본 가속도 (중력 포함) - y축이 수직
    base_accel_y = 9.8  # 중력 가속도
    
    # 보행 시뮬레이션: 발뒤꿈치 착지와 발가락 떼기 패턴
    gait_phase = (timestamp * gait_freq) % 1.0
    
    if 0.0 <= gait_phase < 0.1:  # Heel Strike 구간
        accel_y = base_accel_y + 15 * math.sin(gait_phase * 20 * math.pi)
    elif 0.6 <= gait_phase < 0.7:  # Toe Off 구간  
        accel_y = base_accel_y - 10 * math.sin((gait_phase - 0.6) * 20 * math.pi)
    else:  # 평상시
        accel_y = base_accel_y + 2 * math.sin(timestamp * 4 * math.pi)  # 작은 진동
    
    # X, Z축은 작은 변동
    accel_x = 0.5 * math.sin(timestamp * 3 * math.pi)
    accel_z = 0.3 * math.cos(timestamp * 2.5 * math.pi)
    
    return {
        'timestamp': timestamp,
        'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
        'gyro': {'x': 0, 'y': 0, 'z': 0}  # 자이로스코프는 0으로
    }

def main():
    try:
        # 서버 연결
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"Connected to {SERVER_IP}:{SERVER_PORT}")
        
        start_time = time.time()
        
        print("Sending test IMU data with simulated gait pattern...")
        print("Look for HS and TO markers on the graphs!")
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 테스트 데이터 생성
            test_data = generate_test_data(elapsed)
            
            # JSON으로 변환하여 전송
            data_json = json.dumps(test_data)
            client_socket.sendall((data_json + '\n').encode('utf-8'))
            
            # 100Hz로 전송
            time.sleep(0.01)
            
            # 10초마다 진행 상황 출력
            if int(elapsed) % 10 == 0 and int(elapsed * 100) % 1000 == 0:
                print(f"Sent {int(elapsed * 100)} samples, elapsed: {elapsed:.1f}s")
    
    except KeyboardInterrupt:
        print("\nTest client stopped")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'client_socket' in locals():
            client_socket.close()
            print("Connection closed")

if __name__ == "__main__":
    main() 