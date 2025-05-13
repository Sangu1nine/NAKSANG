import socket
import json
import threading
import pandas as pd
import numpy as np
import datetime
import time
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
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
buffer_size = 1000  # 최근 1000개 데이터 포인트만 유지
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# PyQtGraph 관련 변수
app = None
win = None
accel_plot = None
gyro_plot = None
accel_curves = {}
gyro_curves = {}
visualization_active = False

def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"데이터 폴더 생성: {DATA_FOLDER}")

def init_visualization():
    global app, win, accel_plot, gyro_plot, accel_curves, gyro_curves, visualization_active
    
    if visualization_active:
        return
        
    visualization_active = True
    
    # PyQtGraph 초기화
    app = pg.mkQApp("IMU Data Visualization")
    win = pg.GraphicsLayoutWidget(show=True, title="IMU 데이터 실시간 모니터링")
    win.resize(1000, 800)
    
    # 가속도 그래프 설정
    accel_plot = win.addPlot(row=0, col=0, title="가속도 (g)")
    accel_plot.setLabel('left', '가속도', 'g')
    accel_plot.setLabel('bottom', '시간', 's')
    accel_plot.addLegend()
    accel_plot.showGrid(x=True, y=True)
    
    # 자이로스코프 그래프 설정
    win.nextRow()
    gyro_plot = win.addPlot(row=1, col=0, title="자이로스코프 (°/s)")
    gyro_plot.setLabel('left', '각속도', '°/s')
    gyro_plot.setLabel('bottom', '시간', 's')
    gyro_plot.addLegend()
    gyro_plot.showGrid(x=True, y=True)
    
    # 데이터 라인 생성
    accel_curves['x'] = accel_plot.plot(pen='r', name='X축')
    accel_curves['y'] = accel_plot.plot(pen='g', name='Y축')
    accel_curves['z'] = accel_plot.plot(pen='b', name='Z축')
    
    gyro_curves['x'] = gyro_plot.plot(pen='r', name='X축')
    gyro_curves['y'] = gyro_plot.plot(pen='g', name='Y축')
    gyro_curves['z'] = gyro_plot.plot(pen='b', name='Z축')
    
    # 자동 스케일링 설정
    accel_plot.enableAutoRange()
    gyro_plot.enableAutoRange()
    
    # 타이머 설정 (100ms마다 업데이트)
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(100)

def update_plot():
    global time_buffer, accel_plot, gyro_plot, accel_curves, gyro_curves
    
    if len(time_buffer) == 0:
        return
        
    with data_lock:
        # 데이터 가져오기
        t = np.array(time_buffer)
        ax = np.array(accel_x_buffer)
        ay = np.array(accel_y_buffer)
        az = np.array(accel_z_buffer)
        gx = np.array(gyro_x_buffer)
        gy = np.array(gyro_y_buffer)
        gz = np.array(gyro_z_buffer)
    
    # 데이터 업데이트
    accel_curves['x'].setData(t, ax)
    accel_curves['y'].setData(t, ay)
    accel_curves['z'].setData(t, az)
    
    gyro_curves['x'].setData(t, gx)
    gyro_curves['y'].setData(t, gy)
    gyro_curves['z'].setData(t, gz)
    
    # x축 범위 자동 조정
    if len(t) > 0:
        accel_plot.setXRange(min(t), max(t))
        gyro_plot.setXRange(min(t), max(t))

def client_handler(client_socket, client_address):
    global received_data, is_receiving, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer
    
    print(f"클라이언트 연결됨: {client_address}")
    
    # 시각화 초기화를 메인 스레드에서 직접 실행
    if not visualization_active:
        init_visualization()
    
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

if __name__ == "__main__":
    print("IMU 데이터 수신 및 실시간 시각화 서버 시작")
    start_server()
