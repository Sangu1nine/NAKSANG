import socket
import json
import threading
import pandas as pd
import numpy as np
import datetime
import time
import os

# 서버 설정
SERVER_IP = '0.0.0.0'  # 모든 네트워크 인터페이스에서 연결 수신
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'

# 수신 데이터 저장 변수
received_data = []
data_lock = threading.Lock()
is_receiving = False

def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"데이터 폴더 생성: {DATA_FOLDER}")

def client_handler(client_socket, client_address):
    global received_data, is_receiving
    
    print(f"클라이언트 연결됨: {client_address}")
    
    buffer = ""
    start_time = time.time()
    sample_count = 0
    
    try:
        is_receiving = True
        
        while is_receiving:
            # 데이터 수신
            data = client_socket.recv(4096)
            
            if not data:
                print("클라이언트 연결 종료")
                break
                
            # 버퍼에 수신된 데이터 추가
            buffer += data.decode('utf-8')
            
            # 완전한 JSON 객체 처리
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    sensor_data = json.loads(line)
                    
                    # 데이터 추출
                    timestamp = sensor_data['timestamp']
                    accel_x = sensor_data['accel']['x']
                    accel_y = sensor_data['accel']['y']
                    accel_z = sensor_data['accel']['z']
                    gyro_x = sensor_data['gyro']['x']
                    gyro_y = sensor_data['gyro']['y']
                    gyro_z = sensor_data['gyro']['z']
                    
                    # 데이터 저장
                    with data_lock:
                        received_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                    
                    sample_count += 1
                    
                    # 주기적으로 상태 출력
                    if sample_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"수신 데이터: {sample_count}개, 경과 시간: {elapsed:.2f}초, 샘플링 레이트: {sample_count/elapsed:.2f}Hz")
                        print(f"최근 데이터 - 가속도(g): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                        print(f"최근 데이터 - 자이로(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON 디코딩 오류: {str(e)}")
    
    except Exception as e:
        print(f"클라이언트 핸들러 오류: {str(e)}")
    
    finally:
        client_socket.close()
        print(f"클라이언트 연결 종료: {client_address}")
        
        # 데이터 저장
        save_received_data()

def save_received_data():
    global received_data
    
    with data_lock:
        if len(received_data) == 0:
            print("저장할 데이터 없음")
            return
            
        # DataFrame 생성
        columns = ['Time(s)', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        df = pd.DataFrame(received_data, columns=columns)
        
        # 파일명 설정
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_FOLDER, f"received_imu_data_{timestamp}.csv")
        
        # CSV 저장
        df.to_csv(filename, index=False)
        print(f"데이터 저장 완료: {filename} (총 {len(df)}개 샘플)")
        
        # 데이터 초기화
        received_data = []

def start_server():
    global is_receiving
    
    # 데이터 폴더 생성
    create_data_folder()
    
    # 서버 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        
        print(f"서버 시작: {SERVER_IP}:{SERVER_PORT}")
        print("Ctrl+C로 서버 종료")
        
        while True:
            print("클라이언트 연결 대기 중...")
            client_socket, client_address = server_socket.accept()
            
            # 새 클라이언트를 위한 스레드 시작
            client_thread = threading.Thread(target=client_handler, args=(client_socket, client_address))
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n서버 종료 요청")
        is_receiving = False
        
    except Exception as e:
        print(f"서버 오류: {str(e)}")
        
    finally:
        server_socket.close()
        print("서버 종료")
        
        # 남은 데이터 저장
        save_received_data()

if __name__ == "__main__":
    start_server() 