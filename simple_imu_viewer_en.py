import socket
import json
import threading
import datetime
import time
import os
import sys
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

# Server settings
SERVER_IP = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000
DATA_FOLDER = 'received_data'

# Global variables
received_data = []
data_lock = threading.Lock()
is_running = False

# Data buffers for visualization
buffer_size = 500  # Keep only the most recent 500 data points
time_buffer = deque(maxlen=buffer_size)
accel_x_buffer = deque(maxlen=buffer_size)
accel_y_buffer = deque(maxlen=buffer_size)
accel_z_buffer = deque(maxlen=buffer_size)
gyro_x_buffer = deque(maxlen=buffer_size)
gyro_y_buffer = deque(maxlen=buffer_size)
gyro_z_buffer = deque(maxlen=buffer_size)

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

# Create data folder
def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Data folder created: {DATA_FOLDER}")

# Save received data to CSV
def save_data():
    global received_data
    
    with data_lock:
        if not received_data:
            return
            
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_FOLDER, f"imu_data_{timestamp}.csv")
        
        # Save to CSV
        with open(filename, 'w') as f:
            f.write("Time,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n")
            for data in received_data:
                f.write(f"{','.join(str(x) for x in data)}\n")
        
        print(f"Data saved: {filename} (Total: {len(received_data)} samples)")
        received_data = []

# Client connection handler
def handle_client(client_socket, address):
    global received_data, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer
    
    print(f"Client connected: {address}")
    buffer = ""
    
    try:
        while is_running:
            data = client_socket.recv(4096)
            if not data:
                break
                
            # Add to buffer
            buffer += data.decode('utf-8')
            
            # Process complete JSON objects
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                try:
                    # Parse JSON
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
                        
                        # Add to visualization buffers
                        time_buffer.append(timestamp)
                        accel_x_buffer.append(accel_x)
                        accel_y_buffer.append(accel_y)
                        accel_z_buffer.append(accel_z)
                        gyro_x_buffer.append(gyro_x)
                        gyro_y_buffer.append(gyro_y)
                        gyro_z_buffer.append(gyro_z)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
    
    except Exception as e:
        print(f"Client handler error: {e}")
    
    finally:
        client_socket.close()
        print(f"Client disconnected: {address}")

# Update plot function (called by timer)
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

# Server thread function
def server_thread_func():
    global server_socket, is_running
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)  # Set timeout
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
            except Exception as e:
                if is_running:
                    print(f"Connection accept error: {e}")
                    time.sleep(0.5)
    
    except Exception as e:
        print(f"Server thread error: {e}")
    
    finally:
        if server_socket:
            server_socket.close()
            print("Server socket closed")

# Exit handler
def handle_exit():
    global is_running, server_socket
    
    is_running = False
    print("\nExit requested")
    
    # Save data
    save_data()
    
    # Close socket
    if server_socket:
        try:
            server_socket.close()
        except:
            pass

# Main function
def main():
    global app, win, plots, curves, is_running
    
    # Create folder
    create_data_folder()
    
    # Setup PyQtGraph
    app = QtWidgets.QApplication([])
    
    # Create window
    win = pg.GraphicsLayoutWidget(show=True, title="IMU Data Monitoring")
    win.resize(1000, 700)
    
    # Connect exit handler
    app.aboutToQuit.connect(handle_exit)
    
    # Accelerometer graph
    plots['accel'] = win.addPlot(row=0, col=0)
    plots['accel'].setTitle("Acceleration (g)")
    plots['accel'].setLabel('left', "Acceleration", "g")
    plots['accel'].setLabel('bottom', "Time", "s")
    plots['accel'].addLegend()
    plots['accel'].showGrid(x=True, y=True)
    
    # Gyroscope graph
    win.nextRow()
    plots['gyro'] = win.addPlot(row=1, col=0)
    plots['gyro'].setTitle("Gyroscope (°/s)")
    plots['gyro'].setLabel('left', "Angular Velocity", "°/s")
    plots['gyro'].setLabel('bottom', "Time", "s")
    plots['gyro'].addLegend()
    plots['gyro'].showGrid(x=True, y=True)
    
    # Link X axes for simultaneous scrolling
    plots['gyro'].setXLink(plots['accel'])
    
    # Create data curves
    curves['accel_x'] = plots['accel'].plot(pen=(255,0,0), name="X-axis")
    curves['accel_y'] = plots['accel'].plot(pen=(0,255,0), name="Y-axis")
    curves['accel_z'] = plots['accel'].plot(pen=(0,0,255), name="Z-axis")
    
    curves['gyro_x'] = plots['gyro'].plot(pen=(255,0,0), name="X-axis")
    curves['gyro_y'] = plots['gyro'].plot(pen=(0,255,0), name="Y-axis")
    curves['gyro_z'] = plots['gyro'].plot(pen=(0,0,255), name="Z-axis")
    
    # Update timer
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(50)  # Update every 50ms
    
    # Start server thread
    is_running = True
    server_thread = threading.Thread(target=server_thread_func)
    server_thread.daemon = True
    server_thread.start()
    
    # Start GUI event loop
    print("IMU Data Reception and Visualization Started")
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTerminated by keyboard interrupt")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Final cleanup
        is_running = False
        save_data() 