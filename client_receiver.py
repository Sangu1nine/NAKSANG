import socket
import json
import threading
import pandas as pd
import numpy as np
import datetime
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# 서버 설정
SERVER_IP = '0.0.0.0'  # 모든 네트워크 인터페이스에서 연결 수신
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'

# 수신 데이터 저장 변수
received_data = []
data_lock = threading.Lock()
is_receiving = False

# 실시간 시각화를 위한 데이터 버퍼
# deque를 사용하여 최대 길이를 제한 (메모리 사용 제한)
buffer_size = 500  # 최근 500개 데이터 포인트만 유지
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# 플롯 객체 저장 변수
fig = None
axes = None
lines = {}
plot_start_time = None
visualization_active = False

def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"데이터 폴더 생성: {DATA_FOLDER}")

def init_visualization():
    global fig, axes, lines, plot_start_time, visualization_active
    
    # 이미 실행 중인 경우 반환
    if visualization_active:
        return
        
    visualization_active = True
    plot_start_time = time.time()
    
    # 그래프 생성
    plt.ion()  # 인터랙티브 모드 켜기
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU 데이터 실시간 모니터링", fontsize=16)
    
    # 가속도 그래프 설정
    axes[0].set_title("가속도 (g)")
    axes[0].set_ylabel("가속도 (g)")
    axes[0].grid(True)
    
    # 자이로스코프 그래프 설정
    axes[1].set_title("자이로스코프 (°/s)")
    axes[1].set_xlabel("시간 (초)")
    axes[1].set_ylabel("각속도 (°/s)")
    axes[1].grid(True)
    
    # 데이터 라인 생성
    lines['accel_x'], = axes[0].plot([], [], 'r-', label='X축')
    lines['accel_y'], = axes[0].plot([], [], 'g-', label='Y축')
    lines['accel_z'], = axes[0].plot([], [], 'b-', label='Z축')
    
    lines['gyro_x'], = axes[1].plot([], [], 'r-', label='X축')
    lines['gyro_y'], = axes[1].plot([], [], 'g-', label='Y축')
    lines['gyro_z'], = axes[1].plot([], [], 'b-', label='Z축')
    
    # 범례 추가
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    
    # 주기적 업데이트를 위한 애니메이션 설정
    ani = FuncAnimation(fig, update_plot, interval=100)  # 100ms마다 업데이트
    
def update_plot(frame):
    global time_buffer, fig, axes, lines, plot_start_time
    
    if len(time_buffer) == 0:
        return
        
    with data_lock:
        # 시간 데이터 (상대적)
        t = list(time_buffer)
        
        # 가속도 데이터
        ax = list(accel_x_buffer)
        ay = list(accel_y_buffer)
        az = list(accel_z_buffer)
        
        # 자이로스코프 데이터
        gx = list(gyro_x_buffer)
        gy = list(gyro_y_buffer)
        gz = list(gyro_z_buffer)
    
    # 데이터 업데이트
    lines['accel_x'].set_data(t, ax)
    lines['accel_y'].set_data(t, ay)
    lines['accel_z'].set_data(t, az)
    
    lines['gyro_x'].set_data(t, gx)
    lines['gyro_y'].set_data(t, gy)
    lines['gyro_z'].set_data(t, gz)
    
    # x축 범위 자동 조정
    if len(t) > 0:
        for ax in axes:
            ax.set_xlim(min(t), max(t))
        
        # y축 범위 자동 조정 (가속도)
        if len(ax) > 0:
            all_accel = ax + ay + az
            min_val = min(all_accel) - 0.1
            max_val = max(all_accel) + 0.1
            axes[0].set_ylim(min_val, max_val)
        
        # y축 범위 자동 조정 (자이로스코프)
        if len(gx) > 0:
            all_gyro = gx + gy + gz
            min_val = min(all_gyro) - 0.1
            max_val = max(all_gyro) + 0.1
            axes[1].set_ylim(min_val, max_val)
    
    # 그래프 갱신
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def client_handler(client_socket, client_address):
    global received_data, is_receiving, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer, plot_start_time
    
    print(f"클라이언트 연결됨: {client_address}")
    
    buffer = ""
    start_time = time.time()
    sample_count = 0
    
    # 시각화 초기화
    if not visualization_active:
        init_visualization_thread = threading.Thread(target=init_visualization)
        init_visualization_thread.daemon = True
        init_visualization_thread.start()
    
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
                        
                        # 시각화 버퍼에 데이터 추가
                        time_buffer.append(timestamp)
                        accel_x_buffer.append(accel_x)
                        accel_y_buffer.append(accel_y)
                        accel_z_buffer.append(accel_z)
                        gyro_x_buffer.append(gyro_x)
                        gyro_y_buffer.append(gyro_y)
                        gyro_z_buffer.append(gyro_z)
                    
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
        
        # 시각화 창 닫기
        if fig is not None:
            plt.close(fig)

if __name__ == "__main__":
    print("IMU 데이터 수신 및 실시간 시각화 서버 시작")
    start_server() 