import socket
import json
import threading
import pandas as pd
import numpy as np
import datetime
import time
import os
import signal
import sys
import queue
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

# Server settings
SERVER_IP = '0.0.0.0'  # Listen on all network interfaces
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'

# Variables for storing received data
received_data = []
data_lock = threading.Lock()
is_receiving = False

# Data buffers for real-time visualization
# Using deque with limited length to restrict memory usage
buffer_size = 500  # Keep only the most recent 500 data points
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

# Add some initial data to prevent empty graphs
for i in range(5):
    time_buffer.append(i)
    accel_x_buffer.append(0)
    accel_y_buffer.append(0)
    accel_z_buffer.append(0)
    gyro_x_buffer.append(0)
    gyro_y_buffer.append(0)
    gyro_z_buffer.append(0)

# PyQtGraph variables
app = None
win = None
plots = {}
curves = {}
visualization_active = False
vis_queue = queue.Queue()  # Queue for visualization commands

# Server socket
server_socket = None

def signal_handler(sig, frame):
    """Handle Ctrl+C signals properly"""
    print("\nServer shutdown requested (Ctrl+C)")
    shutdown_server()
    sys.exit(0)

def shutdown_server():
    """Clean shutdown of the server"""
    global is_receiving, server_socket, app
    
    # Stop receiving data
    is_receiving = False
    
    # Close server socket
    if server_socket:
        try:
            server_socket.close()
            print("Server socket closed")
        except Exception as e:
            print(f"Error closing server socket: {str(e)}")
        finally:
            server_socket = None  # 소켓을 None으로 설정하여 재사용 방지
    
    # Save remaining data
    save_received_data()
    
    # Close visualization window
    if app is not None:
        try:
            # PyQt 애플리케이션 종료
            if hasattr(app, 'activeWindow') and app.activeWindow():
                app.closeAllWindows()
            print("Visualization window closed")
        except Exception as e:
            print(f"Error closing visualization: {str(e)}")
            pass

def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Data folder created: {DATA_FOLDER}")

class IMUVisualizer(QtCore.QObject):
    def __init__(self):
        super().__init__()
        
        # Create timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)  # Update every 100ms
    
    def update_plots(self):
        global time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer
        
        try:
            with data_lock:
                # Make copies of the data to avoid threading issues
                t = list(time_buffer)
                ax = list(accel_x_buffer)
                ay = list(accel_y_buffer)
                az = list(accel_z_buffer)
                gx = list(gyro_x_buffer)
                gy = list(gyro_y_buffer)
                gz = list(gyro_z_buffer)
            
            # Update the curves
            if len(t) > 0:
                curves['accel_x'].setData(t, ax)
                curves['accel_y'].setData(t, ay)
                curves['accel_z'].setData(t, az)
                
                curves['gyro_x'].setData(t, gx)
                curves['gyro_y'].setData(t, gy)
                curves['gyro_z'].setData(t, gz)
                
                # Auto-range if needed
                plots['accel'].enableAutoRange()
                plots['gyro'].enableAutoRange()
        
        except Exception as e:
            print(f"Error updating plots: {str(e)}")

# Function to be run in the main thread for visualization
def start_visualization():
    global app, win, plots, curves, visualization_active
    
    try:
        print("Starting visualization...")
        visualization_active = True
        
        # Create application and window
        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="IMU Data Real-time Monitoring")
        win.resize(1200, 800)
        
        # Configure acceleration graph
        plots['accel'] = win.addPlot(row=0, col=0)
        plots['accel'].setLabel('left', "Acceleration", units='g')
        plots['accel'].setLabel('bottom', "Time", units='s')
        plots['accel'].setTitle("Acceleration (g)")
        plots['accel'].addLegend()
        plots['accel'].showGrid(x=True, y=True)
        
        # Configure gyroscope graph
        win.nextRow()
        plots['gyro'] = win.addPlot(row=1, col=0)
        plots['gyro'].setLabel('left', "Angular Velocity", units='°/s')
        plots['gyro'].setLabel('bottom', "Time", units='s')
        plots['gyro'].setTitle("Gyroscope (°/s)")
        plots['gyro'].addLegend()
        plots['gyro'].showGrid(x=True, y=True)
        
        # Link X axes for simultaneous scrolling
        plots['gyro'].setXLink(plots['accel'])
        
        # Create data curves with initial data
        with data_lock:
            t = list(time_buffer)
            ax = list(accel_x_buffer)
            ay = list(accel_y_buffer)
            az = list(accel_z_buffer)
            gx = list(gyro_x_buffer)
            gy = list(gyro_y_buffer)
            gz = list(gyro_z_buffer)
        
        # Create curves with different colors
        curves['accel_x'] = pg.PlotDataItem(t, ax, pen=(255, 0, 0), name='X-axis')
        curves['accel_y'] = pg.PlotDataItem(t, ay, pen=(0, 255, 0), name='Y-axis')
        curves['accel_z'] = pg.PlotDataItem(t, az, pen=(0, 0, 255), name='Z-axis')
        
        curves['gyro_x'] = pg.PlotDataItem(t, gx, pen=(255, 0, 0), name='X-axis')
        curves['gyro_y'] = pg.PlotDataItem(t, gy, pen=(0, 255, 0), name='Y-axis')
        curves['gyro_z'] = pg.PlotDataItem(t, gz, pen=(0, 0, 255), name='Z-axis')
        
        # Add curves to plots
        plots['accel'].addItem(curves['accel_x'])
        plots['accel'].addItem(curves['accel_y'])
        plots['accel'].addItem(curves['accel_z'])
        
        plots['gyro'].addItem(curves['gyro_x'])
        plots['gyro'].addItem(curves['gyro_y'])
        plots['gyro'].addItem(curves['gyro_z'])
        
        # Create visualizer object with timer for updates
        visualizer = IMUVisualizer()
        
        print("Visualization initialized successfully")
        
        # Set window title
        win.setWindowTitle("IMU Data Real-time Monitoring")
        
        return app, win
    
    except Exception as e:
        print(f"Error in visualization_handler: {str(e)}")
        visualization_active = False
        return None, None

def client_handler(client_socket, client_address):
    global received_data, is_receiving, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer, vis_queue
    
    print(f"Client connected: {client_address}")
    
    buffer = ""
    start_time = time.time()
    sample_count = 0
    
    # Signal that visualization needs to be started if not already active
    if not visualization_active:
        vis_queue.put(True)
    
    try:
        while is_receiving:
            # Receive data
            try:
                data = client_socket.recv(4096)
                
                if not data:
                    print("Client disconnected")
                    break
                    
                # Add received data to buffer
                buffer += data.decode('utf-8')
            except socket.error:
                # Socket error (may happen during shutdown)
                print("Socket error or connection closed")
                break
            
            # Process complete JSON objects
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    sensor_data = json.loads(line)
                    
                    # Extract data
                    timestamp = sensor_data['timestamp']
                    accel_x = sensor_data['accel']['x']
                    accel_y = sensor_data['accel']['y']
                    accel_z = sensor_data['accel']['z']
                    gyro_x = sensor_data['gyro']['x']
                    gyro_y = sensor_data['gyro']['y']
                    gyro_z = sensor_data['gyro']['z']
                    
                    # Store data
                    with data_lock:
                        received_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                        
                        # Add data to visualization buffers
                        time_buffer.append(timestamp)
                        accel_x_buffer.append(accel_x)
                        accel_y_buffer.append(accel_y)
                        accel_z_buffer.append(accel_z)
                        gyro_x_buffer.append(gyro_x)
                        gyro_y_buffer.append(gyro_y)
                        gyro_z_buffer.append(gyro_z)
                    
                    sample_count += 1
                    
                    # Periodically display status
                    if sample_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Received data: {sample_count}, Elapsed time: {elapsed:.2f}s, Sampling rate: {sample_count/elapsed:.2f}Hz")
                        print(f"Latest data - Acceleration(g): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
                        print(f"Latest data - Gyroscope(°/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {str(e)}")
    
    except Exception as e:
        print(f"Client handler error: {str(e)}")
    
    finally:
        try:
            client_socket.close()
            print(f"Client connection closed: {client_address}")
        except:
            pass
        
        # Save data
        save_received_data()

def save_received_data():
    global received_data
    
    with data_lock:
        if len(received_data) == 0:
            print("No data to save")
            return
            
        # Create DataFrame
        columns = ['Time(s)', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
        df = pd.DataFrame(received_data, columns=columns)
        
        # Set filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_FOLDER, f"received_imu_data_{timestamp}.csv")
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved: {filename} (Total {len(df)} samples)")
        
        # Reset data
        received_data = []

def start_server():
    global is_receiving, server_socket, vis_queue, app, win
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create data folder
    create_data_folder()
    
    try:
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind and listen
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)  # Add timeout to allow checking for shutdown
        server_socket.listen(1)
        
        print(f"Server started: {SERVER_IP}:{SERVER_PORT}")
        print("Press Ctrl+C to stop server")
        
        is_receiving = True
        
        # Check if we need to start visualization immediately
        if not vis_queue.empty():
            vis_queue.get()
            app, win = start_visualization()
        
        # Start server loop in a separate thread
        server_thread = threading.Thread(target=server_loop)
        server_thread.daemon = True
        server_thread.start()
        
        # If PyQtGraph app is running, enter its event loop
        if app is not None:
            try:
                app.exec_()
            except Exception as e:
                print(f"Error in Qt event loop: {str(e)}")
        else:
            # GUI가 실행되지 않은 경우 기본 루프로 대기
            try:
                while is_receiving:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nServer shutdown requested (KeyboardInterrupt)")
        
        # 종료시 정리
        shutdown_server()
        
    except socket.error as e:
        print(f"Socket error during server startup: {str(e)}")
        shutdown_server()
    except KeyboardInterrupt:
        # This is a backup handler
        print("\nServer shutdown requested (KeyboardInterrupt)")
        shutdown_server()
    except Exception as e:
        print(f"Server error: {str(e)}")
        shutdown_server()

def server_loop():
    global is_receiving, server_socket, vis_queue, app, win
    
    try:
        while is_receiving:
            # Check if we need to start visualization
            try:
                if not vis_queue.empty():
                    vis_queue.get()
                    # We can't start visualization from non-main thread
                    # Just notify user they need to restart
                    print("Visualization requested but can't be started from a thread")
                    print("Please restart the application")
            except Exception as e:
                print(f"Visualization error: {str(e)}")
                
            try:
                # 소켓 상태 확인
                if server_socket is None:
                    print("Server socket is no longer valid")
                    is_receiving = False
                    break
                    
                print("Waiting for client connection...")
                client_socket, client_address = server_socket.accept()
                
                # Start thread for new client
                client_thread = threading.Thread(target=client_handler, args=(client_socket, client_address))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                # Timeout on accept, just continue the loop (allows checking is_receiving)
                continue
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\nServer shutdown requested (Ctrl+C caught in accept)")
                is_receiving = False
                break
            except OSError as e:
                if not is_receiving:
                    # 서버가 종료 중일 때는 예외를 무시함
                    break
                print(f"Socket error: {str(e)}")
                # 소켓 오류가 발생하면 루프를 벗어남
                is_receiving = False
                break
            except Exception as e:
                print(f"Error accepting connection: {str(e)}")
                if is_receiving:
                    # Wait a bit before retrying
                    time.sleep(0.5)
            
    except KeyboardInterrupt:
        # This is a backup handler
        print("\nServer shutdown requested (KeyboardInterrupt)")
    except Exception as e:
        print(f"Server loop error: {str(e)}")
    finally:
        # 마지막 청소 작업
        is_receiving = False

if __name__ == "__main__":
    print("IMU Data Reception and Real-time Visualization Server Started")
    
    try:
        start_server()
    except KeyboardInterrupt:
        # Final backup handler
        print("\nServer shutdown requested (main KeyboardInterrupt)")
        shutdown_server() 