"""
==================================================================
IMU 보행 분석 뷰어 (IMU Gait Analysis Viewer)
==================================================================

이 프로그램은 IMU(관성 측정 장치) 센서 데이터를 실시간으로 수신하고 
보행 패턴을 분석하여 HS(Heel Strike)와 TO(Toe Off)를 감지합니다.

기능:
1. TCP/IP 소켓 서버를 통해 IMU 센서 데이터 수신 (포트 5000)
2. 가속도계(X,Y,Z) 및 자이로스코프(X,Y,Z) 데이터 실시간 그래프 표시
3. 수직 가속도를 이용한 HS/TO 이벤트 감지 및 표시
4. 수신된 데이터를 CSV 파일로 저장 ('received_data' 폴더)
5. 그래프 이미지를 PNG 파일로 저장 ('saved_graphs' 폴더)
6. HS/TO 이벤트 타임스탬프 기록

HS/TO 감지 알고리즘:
- HS: 수직 가속도의 음의 이산 적분 -> 버터워스 필터링 -> 피크 감지
- TO: 수직 가속도의 양의 이산 적분 -> 버터워스 필터링 -> 피크 감지
- 버터워스 필터: 차단 주파수 ω=0.08 rad, 필터 차수 N=4
- 피크 감지: 최소 돌출도 0.1 m/s

사용 방법:
- 프로그램 실행: python imu_gait_viewer.py
- IMU 센서/클라이언트를 포트 5000에 연결하면 자동으로 데이터 수신 시작
- 데이터 형식: JSON {"timestamp": 시간, "accel": {"x":값, "y":값, "z":값}, "gyro": {"x":값, "y":값, "z":값}}
- 키보드 단축키:
  * 'S' 키: 현재 그래프를 PNG 이미지로 저장
  * Ctrl+C: 프로그램 종료 (데이터와 그래프 자동 저장)

요구사항:
- Python 3.6 이상
- PyQtGraph, NumPy, PyQt5, SciPy 라이브러리 필요

개발자: NAKSANG
최종 수정일: 2025년 1월 17일
==================================================================
"""

import socket
import json
import threading
import datetime
import time
import os
import sys
import signal
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

# Server settings
SERVER_IP = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'
GRAPH_FOLDER = 'saved_graphs'
GAIT_LOG_FILE = os.path.join(DATA_FOLDER, 'gait_events.csv')

# Global variables
received_data = []
hs_events = []  # Heel Strike 이벤트 저장
to_events = []  # Toe Off 이벤트 저장
gait_event_queue = []  # 보행 이벤트 UI 업데이트를 위한 큐
data_lock = threading.Lock()
is_running = False
start_time = time.time()

# Data buffers for visualization
buffer_size = 1000  # 1000개 데이터 포인트 유지 (HS/TO 감지를 위해 더 많은 데이터 필요)
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# 이산 적분용 버퍼
integration_buffer_size = 500
accel_z_integration_buffer = deque(maxlen=integration_buffer_size)
time_integration_buffer = deque(maxlen=integration_buffer_size)

# Initialize with some data to prevent empty graphs
for i in range(5):
    time_buffer.append(i)
    accel_x_buffer.append(0)
    accel_y_buffer.append(0)
    accel_z_buffer.append(0)
    gyro_x_buffer.append(0)
    gyro_y_buffer.append(0)
    gyro_z_buffer.append(0)

# Socket and UI variables
server_socket = None
app = None
win = None
plots = {}
curves = {}
hs_markers = []  # HS 이벤트 마커 저장
to_markers = []  # TO 이벤트 마커 저장

# Create data folder
def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Data folder created: {DATA_FOLDER}")
    
    if not os.path.exists(GRAPH_FOLDER):
        os.makedirs(GRAPH_FOLDER)
        print(f"Graph folder created: {GRAPH_FOLDER}")
    
    if not os.path.exists(GAIT_LOG_FILE):
        with open(GAIT_LOG_FILE, 'w') as f:
            f.write("Timestamp,Date_Time,Event_Type\n")
        print(f"Gait event log file created: {GAIT_LOG_FILE}")

# Butterworth filter implementation
def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    """버터워스 저역통과 필터 적용"""
    if len(data) < 10:  # 데이터가 너무 적으면 필터링하지 않음
        return data
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    
    # 차단 주파수가 너무 높으면 조정
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.95
    
    b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy_signal.filtfilt(b, a, data)
    return y

# Discrete integration
def discrete_integration(data, dt):
    """이산 적분 계산"""
    if len(data) < 2:
        return np.array(data)
    
    integrated = np.cumsum(data) * dt
    return integrated

# HS/TO detection function
def detect_gait_events(accel_z_data, time_data):
    """HS와 TO 이벤트를 감지"""
    if len(accel_z_data) < 50:  # 최소 데이터 개수 확인
        return [], []
    
    # 샘플링 주파수 계산
    dt = np.mean(np.diff(time_data)) if len(time_data) > 1 else 0.01
    fs = 1.0 / dt if dt > 0 else 100
    
    # 차단 주파수 설정 (ω=0.08 rad를 Hz로 변환)
    cutoff_freq = 0.08 / (2 * np.pi)  # rad/s to Hz
    
    try:
        # HS 감지: 음의 이산 적분
        neg_integration = discrete_integration(-np.array(accel_z_data), dt)
        filtered_neg = butter_lowpass_filter(neg_integration, cutoff_freq, fs, order=4)
        
        # TO 감지: 양의 이산 적분
        pos_integration = discrete_integration(np.array(accel_z_data), dt)
        filtered_pos = butter_lowpass_filter(pos_integration, cutoff_freq, fs, order=4)
        
        # 피크 감지 (prominence >= 0.1 m/s)
        hs_peaks, _ = find_peaks(filtered_neg, prominence=0.1)
        to_peaks, _ = find_peaks(filtered_pos, prominence=0.1)
        
        # 피크 인덱스를 시간으로 변환
        hs_times = [time_data[i] for i in hs_peaks if i < len(time_data)]
        to_times = [time_data[i] for i in to_peaks if i < len(time_data)]
        
        return hs_times, to_times
        
    except Exception as e:
        print(f"Gait event detection error: {e}")
        return [], []

# Process gait events
def process_gait_events(hs_times, to_times):
    global hs_events, to_events
    
    with data_lock:
        # 새로운 이벤트만 추가 (중복 방지)
        current_time = time.time()
        
        for hs_time in hs_times:
            if not any(abs(hs_time - existing[0]) < 0.1 for existing in hs_events):
                hs_events.append([hs_time, "HS"])
                gait_event_queue.append(("HS", hs_time))
                add_gait_marker(hs_time, "HS")
                print(f"HS detected at: {datetime.datetime.fromtimestamp(hs_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        for to_time in to_times:
            if not any(abs(to_time - existing[0]) < 0.1 for existing in to_events):
                to_events.append([to_time, "TO"])
                gait_event_queue.append(("TO", to_time))
                add_gait_marker(to_time, "TO")
                print(f"TO detected at: {datetime.datetime.fromtimestamp(to_time).strftime('%H:%M:%S.%f')[:-3]}")

# Add gait marker to graph
def add_gait_marker(timestamp, event_type):
    if event_type == "HS":
        line_pen = pg.mkPen(color=(0, 0, 255), width=2, style=QtCore.Qt.DashLine)  # 파란색
        markers_list = hs_markers
    else:  # TO
        line_pen = pg.mkPen(color=(255, 165, 0), width=2, style=QtCore.Qt.DashLine)  # 주황색
        markers_list = to_markers
    
    acc_line = pg.InfiniteLine(pos=timestamp, angle=90, pen=line_pen, movable=False)
    gyr_line = pg.InfiniteLine(pos=timestamp, angle=90, pen=line_pen, movable=False)
    
    markers_list.append((acc_line, gyr_line))
    plots['accel'].addItem(acc_line)
    plots['gyro'].addItem(gyr_line)

# Save received data to CSV
def save_data():
    global received_data, hs_events, to_events
    
    with data_lock:
        if received_data:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_FOLDER, f"imu_gait_data_{timestamp}.csv")
            
            # 메인 데이터 저장
            with open(filename, 'w') as f:
                f.write("Time,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n")
                for data in received_data:
                    f.write(f"{','.join(str(x) for x in data)}\n")
            
            print(f"Data saved: {filename} (Total: {len(received_data)} samples)")
            
            # 보행 이벤트 저장
            gait_filename = os.path.join(DATA_FOLDER, f"gait_events_{timestamp}.csv")
            with open(gait_filename, 'w') as f:
                f.write("Timestamp,Date_Time,Event_Type\n")
                
                # HS와 TO 이벤트를 시간순으로 정렬하여 저장
                all_events = []
                for event in hs_events:
                    all_events.append((event[0], "HS"))
                for event in to_events:
                    all_events.append((event[0], "TO"))
                
                all_events.sort(key=lambda x: x[0])  # 시간순 정렬
                
                for timestamp_val, event_type in all_events:
                    date_time = datetime.datetime.fromtimestamp(timestamp_val).strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"{timestamp_val},{date_time},{event_type}\n")
            
            print(f"Gait events saved: {gait_filename} (HS: {len(hs_events)}, TO: {len(to_events)})")
            
            # 전역 로그 파일에도 저장
            with open(GAIT_LOG_FILE, 'a') as f:
                for timestamp_val, event_type in all_events:
                    date_time = datetime.datetime.fromtimestamp(timestamp_val).strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"{timestamp_val},{date_time},{event_type}\n")
            
            received_data = []
            hs_events = []
            to_events = []

# Client connection handler
def handle_client(client_socket, address):
    global received_data, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer, start_time
    global accel_z_integration_buffer, time_integration_buffer
    
    print(f"Client connected: {address}")
    buffer = ""
    
    try:
        while is_running:
            data = client_socket.recv(4096)
            if not data:
                break
                
            buffer += data.decode('utf-8')
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                try:
                    data_json = json.loads(line)
                    
                    if 'accel' in data_json and 'gyro' in data_json:
                        elapsed_time = data_json['timestamp']
                        timestamp = start_time + elapsed_time
                        
                        accel_x = data_json['accel']['x']
                        accel_y = data_json['accel']['y']
                        accel_z = data_json['accel']['z']
                        gyro_x = data_json['gyro']['x']
                        gyro_y = data_json['gyro']['y']
                        gyro_z = data_json['gyro']['z']
                        
                        with data_lock:
                            received_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                            
                            time_buffer.append(timestamp)
                            accel_x_buffer.append(accel_x)
                            accel_y_buffer.append(accel_y)
                            accel_z_buffer.append(accel_z)
                            gyro_x_buffer.append(gyro_x)
                            gyro_y_buffer.append(gyro_y)
                            gyro_z_buffer.append(gyro_z)
                            
                            # 이산 적분을 위한 데이터 저장
                            accel_z_integration_buffer.append(accel_z)
                            time_integration_buffer.append(timestamp)
                            
                            # 주기적으로 HS/TO 감지 수행 (50개 데이터마다)
                            if len(accel_z_integration_buffer) % 50 == 0 and len(accel_z_integration_buffer) >= 100:
                                hs_times, to_times = detect_gait_events(
                                    list(accel_z_integration_buffer), 
                                    list(time_integration_buffer)
                                )
                                if hs_times or to_times:
                                    process_gait_events(hs_times, to_times)
                
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
    
    except Exception as e:
        print(f"Client handler error: {e}")
    
    finally:
        client_socket.close()
        print(f"Client disconnected: {address}")

# Update plot function
def update_plots():
    global gait_event_queue
    
    with data_lock:
        t = list(time_buffer)
        ax = list(accel_x_buffer)
        ay = list(accel_y_buffer)
        az = list(accel_z_buffer)
        gx = list(gyro_x_buffer)
        gy = list(gyro_y_buffer)
        gz = list(gyro_z_buffer)
        
        local_gait_events = list(gait_event_queue)
        gait_event_queue.clear()
    
    # 보행 이벤트 UI 업데이트
    if local_gait_events:
        for event_type, timestamp in local_gait_events:
            gait_label = plots['gait_list']
            current_text = gait_label.text
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
            new_text = current_text + f"\n• {time_str} - {event_type} 감지됨"
            gait_label.setText(new_text)
    
    if len(t) > 0:
        curves['accel_x'].setData(t, ax)
        curves['accel_y'].setData(t, ay)
        curves['accel_z'].setData(t, az)
        curves['gyro_x'].setData(t, gx)
        curves['gyro_y'].setData(t, gy)
        curves['gyro_z'].setData(t, gz)
        
        if len(t) > 1:
            x_min = max(min(t), t[-1] - 10)
            x_max = t[-1] + 0.5
            plots['accel'].setXRange(x_min, x_max, padding=0)
            
        def format_time_axis(x):
            try:
                dt = datetime.datetime.fromtimestamp(x)
                return dt.strftime('%H:%M:%S')
            except:
                return ''
        
        plots['accel'].getAxis('bottom').setTicks([[(t[i], format_time_axis(t[i])) for i in range(0, len(t), len(t)//5) if i < len(t)]])
        plots['gyro'].getAxis('bottom').setTicks([[(t[i], format_time_axis(t[i])) for i in range(0, len(t), len(t)//5) if i < len(t)]])

# Server thread function
def server_thread_func():
    global server_socket, is_running
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)
        server_socket.listen(1)
        
        print(f"Server started: {SERVER_IP}:{SERVER_PORT}")
        
        while is_running:
            try:
                client_socket, client_address = server_socket.accept()
                client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except OSError:
                if is_running:
                    print("Server socket closed")
                break
            except Exception as e:
                if is_running:
                    print(f"Connection accept error: {e}")
                    time.sleep(0.5)
    
    except Exception as e:
        print(f"Server thread error: {e}")
    
    finally:
        if server_socket:
            try:
                server_socket.close()
            except:
                pass
            print("Server thread terminated")

# Save graphs to image file
def save_graphs():
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(GRAPH_FOLDER, f"gait_graph_{timestamp}.png")
        
        exporter = pg.exporters.ImageExporter(win.centralWidget)
        exporter.parameters()['width'] = 1200
        exporter.export(filename)
        
        print(f"Graphs saved to image: {filename}")
    except Exception as e:
        print(f"Error saving graphs: {e}")

# Key press handling class
class KeyPressWindow(pg.GraphicsLayoutWidget):
    def __init__(self, **kwargs):
        super(KeyPressWindow, self).__init__(**kwargs)
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            save_graphs()
            print("Screenshot taken (pressed S key)")
        else:
            super(KeyPressWindow, self).keyPressEvent(event)

# Exit handler
def handle_exit():
    global is_running, server_socket
    
    print("\nExit requested - cleaning up...")
    is_running = False
    
    print("Saving data...")
    save_data()
    
    print("Saving graphs...")
    save_graphs()
    
    if server_socket:
        try:
            server_socket.close()
            print("Server socket closed")
        except:
            pass
    
    print("Cleanup completed")

# Signal handler for Ctrl+C
def signal_handler(signum, frame):
    print("\nCtrl+C detected - shutting down gracefully...")
    handle_exit()
    if app:
        app.quit()
    sys.exit(0)

# Main function
def main():
    global app, win, plots, curves, is_running, start_time
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time = time.time()
    
    create_data_folder()
    
    app = QtWidgets.QApplication([])
    
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    
    win = KeyPressWindow(show=True, title="IMU Gait Analysis with HS/TO Detection")
    win.resize(1200, 800)
    
    # 보행 이벤트 리스트 표시
    plots['gait_list'] = pg.LabelItem(text="보행 이벤트 기록:\n• HS: Heel Strike (발뒤꿈치 착지)\n• TO: Toe Off (발가락 떼기)", color='#0000aa')
    win.addItem(plots['gait_list'], row=0, col=1, rowspan=2)
    
    # 범례 추가
    legend_label = pg.LabelItem(text="마커 범례:\n• 파란색 세로선: HS 이벤트\n• 주황색 세로선: TO 이벤트", color='#aa0000')
    win.addItem(legend_label, row=2, col=1)
    
    # 단축키 안내
    shortcut_label = pg.LabelItem(text="Press 'S' to save graphs as image, Ctrl+C to exit", color='#000000')
    win.addItem(shortcut_label, row=3, col=0)
    
    # 상태 레이블
    status_label = pg.LabelItem(text="Gait Analysis Status: Monitoring", color='#00aa00')
    win.addItem(status_label, row=2, col=0)
    
    app.aboutToQuit.connect(handle_exit)
    
    # 가속도 그래프
    plots['accel'] = win.addPlot(row=0, col=0)
    plots['accel'].setTitle("Acceleration (g)")
    plots['accel'].setLabel('left', "Acceleration", "g")
    plots['accel'].setLabel('bottom', "Time", "")
    plots['accel'].addLegend()
    plots['accel'].showGrid(x=True, y=True)
    plots['accel'].setYRange(-2, 2)
    plots['accel'].disableAutoRange(axis=pg.ViewBox.YAxis)
    
    # 자이로스코프 그래프
    win.nextRow()
    plots['gyro'] = win.addPlot(row=1, col=0)
    plots['gyro'].setTitle("Gyroscope (°/s)")
    plots['gyro'].setLabel('left', "Angular Velocity", "°/s")
    plots['gyro'].setLabel('bottom', "Time", "")
    plots['gyro'].addLegend()
    plots['gyro'].showGrid(x=True, y=True)
    plots['gyro'].setYRange(-250, 250)
    plots['gyro'].disableAutoRange(axis=pg.ViewBox.YAxis)
    
    plots['gyro'].setXLink(plots['accel'])
    
    # 데이터 곡선 생성
    curves['accel_x'] = plots['accel'].plot(pen=(255,0,0), name="X-axis")
    curves['accel_y'] = plots['accel'].plot(pen=(0,150,0), name="Y-axis")
    curves['accel_z'] = plots['accel'].plot(pen=(0,0,255), name="Z-axis (Vertical)")
    
    curves['gyro_x'] = plots['gyro'].plot(pen=(255,0,0), name="X-axis")
    curves['gyro_y'] = plots['gyro'].plot(pen=(0,150,0), name="Y-axis")
    curves['gyro_z'] = plots['gyro'].plot(pen=(0,0,255), name="Z-axis")
    
    # 업데이트 타이머
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(50)
    
    # 서버 스레드 시작
    is_running = True
    server_thread = threading.Thread(target=server_thread_func)
    server_thread.daemon = True
    server_thread.start()
    
    print("IMU Gait Analysis with HS/TO Detection Started")
    print("Press Ctrl+C to stop and save data")
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        handle_exit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTerminated by keyboard interrupt")
        handle_exit()
    except Exception as e:
        print(f"Unexpected error: {e}")
        handle_exit()
    finally:
        is_running = False
        save_data() 