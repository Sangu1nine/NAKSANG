import socket
import json
import threading
import pandas as pd
import numpy as np
import datetime
import time
import os
import matplotlib
# 스레드 관련 경고 해결을 위한 백엔드 설정
matplotlib.use('TkAgg')  # GUI 백엔드 설정
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import signal
import sys
import queue

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

# Plot object variables
fig = None
axes = None
lines = {}
plot_start_time = None
visualization_active = False
animation = None  # Store animation object to stop it properly
vis_queue = queue.Queue()  # Queue for visualization commands
anim_ref = None  # Reference to keep animation alive

# Server socket
server_socket = None

def signal_handler(sig, frame):
    """Handle Ctrl+C signals properly"""
    print("\nServer shutdown requested (Ctrl+C)")
    shutdown_server()
    sys.exit(0)

def shutdown_server():
    """Clean shutdown of the server"""
    global is_receiving, server_socket, fig, animation
    
    # Stop receiving data
    is_receiving = False
    
    # Close server socket
    if server_socket:
        try:
            server_socket.close()
            print("Server socket closed")
        except:
            pass
    
    # Save remaining data
    save_received_data()
    
    # Close visualization window
    if fig is not None:
        try:
            # Stop animation
            if animation is not None:
                animation.event_source.stop()
            plt.close(fig)
            print("Visualization window closed")
        except:
            pass

def create_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Data folder created: {DATA_FOLDER}")

# Function to be run in the main thread for visualization
def visualization_handler():
    global fig, axes, lines, animation, anim_ref, visualization_active
    
    visualization_active = True
    
    # Create graphs
    plt.ion()  # Enable interactive mode
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU Data Real-time Monitoring", fontsize=16)
    
    # Configure acceleration graph
    axes[0].set_title("Acceleration (g)")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(True)
    
    # Configure gyroscope graph
    axes[1].set_title("Gyroscope (°/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Velocity (°/s)")
    axes[1].grid(True)
    
    # Create data lines
    lines['accel_x'], = axes[0].plot([], [], 'r-', label='X-axis')
    lines['accel_y'], = axes[0].plot([], [], 'g-', label='Y-axis')
    lines['accel_z'], = axes[0].plot([], [], 'b-', label='Z-axis')
    
    lines['gyro_x'], = axes[1].plot([], [], 'r-', label='X-axis')
    lines['gyro_y'], = axes[1].plot([], [], 'g-', label='Y-axis')
    lines['gyro_z'], = axes[1].plot([], [], 'b-', label='Z-axis')
    
    # Add legends
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Set up animation for periodic updates - fix warning by specifying save_count
    animation = FuncAnimation(fig, update_plot, interval=100, save_count=100)
    
    # Keep reference to prevent garbage collection
    anim_ref = animation
    
    # Show plot with blocking to keep main thread in matplotlib's event loop
    plt.show(block=False)
    
    return animation

def update_plot(frame):
    global time_buffer, fig, axes, lines
    
    # Skip if no data or server is shutting down
    if len(time_buffer) == 0 or not is_receiving:
        return
        
    with data_lock:
        # Time data (relative)
        t = list(time_buffer)
        
        # Acceleration data
        ax = list(accel_x_buffer)
        ay = list(accel_y_buffer)
        az = list(accel_z_buffer)
        
        # Gyroscope data
        gx = list(gyro_x_buffer)
        gy = list(gyro_y_buffer)
        gz = list(gyro_z_buffer)
    
    # Update data
    if len(t) > 0:
        lines['accel_x'].set_data(t, ax)
        lines['accel_y'].set_data(t, ay)
        lines['accel_z'].set_data(t, az)
        
        lines['gyro_x'].set_data(t, gx)
        lines['gyro_y'].set_data(t, gy)
        lines['gyro_z'].set_data(t, gz)
        
        # Auto-adjust x-axis range
        for ax in axes:
            ax.set_xlim(min(t), max(t))
        
        # Auto-adjust y-axis range (acceleration)
        if len(ax) > 0:
            all_accel = ax + ay + az
            min_val = min(all_accel) - 0.1
            max_val = max(all_accel) + 0.1
            axes[0].set_ylim(min_val, max_val)
        
        # Auto-adjust y-axis range (gyroscope)
        if len(gx) > 0:
            all_gyro = gx + gy + gz
            min_val = min(all_gyro) - 0.1
            max_val = max(all_gyro) + 0.1
            axes[1].set_ylim(min_val, max_val)
    
    # Update is handled automatically by FuncAnimation

def client_handler(client_socket, client_address):
    global received_data, is_receiving, time_buffer, accel_x_buffer, accel_y_buffer, accel_z_buffer, gyro_x_buffer, gyro_y_buffer, gyro_z_buffer, plot_start_time, vis_queue
    
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
    global is_receiving, server_socket, vis_queue, anim_ref
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create data folder
    create_data_folder()
    
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.settimeout(0.5)  # Add timeout to allow checking for shutdown
        server_socket.listen(1)
        
        print(f"Server started: {SERVER_IP}:{SERVER_PORT}")
        print("Press Ctrl+C to stop server")
        
        is_receiving = True
        
        while is_receiving:
            # Check if we need to start visualization
            try:
                if not vis_queue.empty():
                    vis_queue.get()
                    # Start visualization in the main thread
                    anim_ref = visualization_handler()
            except Exception as e:
                print(f"Visualization error: {str(e)}")
                
            try:
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
            except Exception as e:
                print(f"Error accepting connection: {str(e)}")
                if is_receiving:
                    # Wait a bit before retrying
                    time.sleep(0.5)
            
            # Update the plot if it's active
            if visualization_active and fig is not None:
                try:
                    plt.pause(0.01)  # Allow GUI events to be processed
                except:
                    pass
            
    except KeyboardInterrupt:
        # This is a backup handler
        print("\nServer shutdown requested (KeyboardInterrupt)")
    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        # Shutdown server
        shutdown_server()

if __name__ == "__main__":
    print("IMU Data Reception and Real-time Visualization Server Started")
    
    try:
        start_server()
    except KeyboardInterrupt:
        # Final backup handler
        print("\nServer shutdown requested (main KeyboardInterrupt)")
        shutdown_server() 