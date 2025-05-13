import socket
import json
import threading
import datetime
import time
import os
import signal
import sys
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

# 서버 설정
SERVER_IP = '0.0.0.0'  # 모든 네트워크 인터페이스에서 수신
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'

# 데이터 저장 변수
received_data = []
data_lock = threading.Lock()
is_running = False

# 시각화를 위한 데이터 버퍼
buffer_size = 500  # 최근 500개 데이터만 유지
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# 그래프 초기화 데이터
for i in range(5):
    time_buffer.append(i)
    accel_x_buffer.append(0)
    accel_y_buffer.append(0)
    accel_z_buffer.append(0)
    gyro_x_buffer.append(0)
    gyro_y_buffer.append(0)
    gyro_z_buffer.append(0)

# 소켓 및 UI 변수
server_socket = None
app = None
win = None
plots = {}
curves = {}

# 폴더 생성 함수
def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"데이터 폴더 생성: {DATA_FOLDER}")

# 데이터 저장 함수
def save_data():
    global received_data
    
    with data_lock:
        if not received_data:
            return
            
        # 타임스탬프로 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_FOLDER, f"imu_data_{timestamp}.csv")
        
        # CSV 형태로 저장
        with open(filename, 'w') as f:
            f.write("Time,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n")
            for data in received_data:
                f.write(f"{','.join(str(x) for x in data)}\n")
        
        print(f"데이터 저장 완료: {filename} (총 {len(received_data)}개)")
        received_data = []

# 클라이언트 연결 처리
def handle_client(client_socket, address):
    global received_data, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer
    
    print(f"클라이언트 연결됨: {address}")
    buffer = ""
    
    try:
        while is_running:
            data = client_socket.recv(4096)
            if not data:
                break
                
            # 데이터 버퍼에 추가
            buffer += data.decode('utf-8')
            
            # 완전한 JSON 객체 처리
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                try:
                    # JSON 파싱
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
                        
                        # 시각화 버퍼에 추가
                        time_buffer.append(timestamp)
                        accel_x_buffer.append(accel_x)
                        accel_y_buffer.append(accel_y)
                        accel_z_buffer.append(accel_z)
                        gyro_x_buffer.append(gyro_x)
                        gyro_y_buffer.append(gyro_y)
                        gyro_z_buffer.append(gyro_z)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
    
    except Exception as e:
        print(f"클라이언트 처리 오류: {e}")
    
    finally:
        client_socket.close()
        print(f"클라이언트 연결 종료: {address}")

# 그래프 업데이트 함수
def update_plots():
    with data_lock:
        t = list(time_buffer)
        ax = list(accel_x_buffer)
        ay = list(accel_y_buffer)
        az = list(accel_z_buffer)
        gx = list(gyro_x_buffer)
        gy = list(gyro_y_buffer)
        gz = list(gyro_z_buffer)
    
    if len(t) > 0:
        curves['accel_x'].setData(t, ax)
        curves['accel_y'].setData(t, ay)
        curves['accel_z'].setData(t, az)
        curves['gyro_x'].setData(t, gx)
        curves['gyro_y'].setData(t, gy)
        curves['gyro_z'].setData(t, gz)

# 서버 스레드 함수
def server_thread_func():
    global server_socket, is_running
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)  # 타임아웃 설정
        server_socket.listen(1)
        
        print(f"서버 시작: {SERVER_IP}:{SERVER_PORT}")
        
        while is_running:
            try:
                client_socket, client_address = server_socket.accept()
                client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if is_running:
                    print(f"연결 수락 오류: {e}")
                    time.sleep(0.5)
    
    except Exception as e:
        print(f"서버 스레드 오류: {e}")
    
    finally:
        if server_socket:
            server_socket.close()
            print("서버 소켓 종료")

# 종료 처리 함수
def handle_exit():
    global is_running, server_socket
    
    is_running = False
    print("\n종료 요청됨")
    
    # 데이터 저장
    save_data()
    
    # 소켓 종료
    if server_socket:
        try:
            server_socket.close()
        except:
            pass

# 메인 함수
def main():
    global app, win, plots, curves, is_running
    
    # 폴더 생성
    create_data_folder()
    
    # PyQtGraph 설정
    app = QtWidgets.QApplication([])
    
    # 윈도우 생성
    win = pg.GraphicsLayoutWidget(show=True, title="IMU 데이터 모니터링")
    win.resize(1000, 700)
    
    # 종료 시 처리
    app.aboutToQuit.connect(handle_exit)
    
    # 가속도 그래프
    plots['accel'] = win.addPlot(row=0, col=0)
    plots['accel'].setTitle("가속도 (g)")
    plots['accel'].setLabel('left', "가속도", "g")
    plots['accel'].setLabel('bottom', "시간", "s")
    plots['accel'].addLegend()
    plots['accel'].showGrid(x=True, y=True)
    
    # 자이로스코프 그래프
    win.nextRow()
    plots['gyro'] = win.addPlot(row=1, col=0)
    plots['gyro'].setTitle("자이로스코프 (°/s)")
    plots['gyro'].setLabel('left', "각속도", "°/s")
    plots['gyro'].setLabel('bottom', "시간", "s")
    plots['gyro'].addLegend()
    plots['gyro'].showGrid(x=True, y=True)
    
    # X축 연결 (동시 스크롤)
    plots['gyro'].setXLink(plots['accel'])
    
    # 데이터 곡선 생성
    curves['accel_x'] = plots['accel'].plot(pen=(255,0,0), name="X축")
    curves['accel_y'] = plots['accel'].plot(pen=(0,255,0), name="Y축")
    curves['accel_z'] = plots['accel'].plot(pen=(0,0,255), name="Z축")
    
    curves['gyro_x'] = plots['gyro'].plot(pen=(255,0,0), name="X축")
    curves['gyro_y'] = plots['gyro'].plot(pen=(0,255,0), name="Y축")
    curves['gyro_z'] = plots['gyro'].plot(pen=(0,0,255), name="Z축")
    
    # 업데이트 타이머
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(50)  # 50ms마다 업데이트
    
    # 서버 스레드 시작
    is_running = True
    server_thread = threading.Thread(target=server_thread_func)
    server_thread.daemon = True
    server_thread.start()
    
    # GUI 이벤트 루프 시작
    print("IMU 데이터 수신 및 시각화 시작")
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
    finally:
        # 마지막 정리 작업
        is_running = False
        save_data() 