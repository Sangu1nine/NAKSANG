#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
보행 이벤트 감지 시스템 (Gait Event Detector)
MODIFIED [2024-12]: IMU 센서 데이터를 활용한 HS/TO 감지 및 실시간 시각화

기능:
- IMU 센서 데이터 실시간 수신 및 처리
- 수직 가속도(acc_y) 및 적분값 시각화
- HS(Heel Strike), TO(Toe Off) 이벤트 감지
- 버터워스 필터링 및 피크 감지
- 스크린샷 저장 및 디버깅 정보 출력
- 설정 파일 기반 매개변수 관리
"""

import sys
import socket
import json
import threading
import time
from datetime import datetime
import numpy as np
from scipy import signal, integrate
from scipy.signal import find_peaks, butter, filtfilt
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QHBoxLayout, QWidget, QPushButton, QLabel, 
                            QTextEdit, QGroupBox, QGridLayout, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont
import collections

# 설정 파일 import
from config import (NETWORK_CONFIG, SENSOR_CONFIG, SIGNAL_PROCESSING, 
                   PEAK_DETECTION, GUI_CONFIG, DEBUG_CONFIG, OUTPUT_CONFIG,
                   FILTER_PRESETS, update_config_from_preset)

class DataReceiver(QObject):
    """TCP 서버로 IMU 데이터를 수신하는 클래스"""
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(str)
    
    def __init__(self, server_ip=None, server_port=None):
        super().__init__()
        self.server_ip = server_ip or '0.0.0.0'  # 모든 인터페이스에서 연결 수신
        self.server_port = server_port or NETWORK_CONFIG['server_port']
        self.server_socket = None
        self.client_socket = None
        self.running = False
        
    def start_receiving(self):
        """서버 시작 및 클라이언트 연결 대기"""
        self.running = True
        try:
            # 서버 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.server_ip, self.server_port))
            self.server_socket.settimeout(1.0)  # 타임아웃 설정
            self.server_socket.listen(1)
            
            self.connection_status.emit(f"서버 시작됨: 포트 {self.server_port}에서 연결 대기 중...")
            
            while self.running:
                try:
                    # 클라이언트 연결 대기
                    self.client_socket, client_address = self.server_socket.accept()
                    self.connection_status.emit(f"클라이언트 연결됨: {client_address}")
                    
                    # 클라이언트 데이터 처리
                    self._handle_client()
                    
                except socket.timeout:
                    continue  # 타임아웃 시 계속 대기
                except OSError:
                    if self.running:
                        self.connection_status.emit("서버 소켓 오류")
                    break
                except Exception as e:
                    if self.running:
                        self.connection_status.emit(f"연결 수락 오류: {str(e)}")
                        time.sleep(0.5)
                        
        except Exception as e:
            self.connection_status.emit(f"서버 시작 실패: {str(e)}")
        finally:
            self.stop_receiving()
    
    def _handle_client(self):
        """클라이언트 데이터 처리"""
        buffer = ""
        
        try:
            while self.running and self.client_socket:
                try:
                    data = self.client_socket.recv(4096)
                    if not data:
                        break
                        
                    buffer += data.decode('utf-8')
                    
                    # 완전한 JSON 라인 처리
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        
                        if line.strip():
                            try:
                                sensor_data = json.loads(line.strip())
                                self.data_received.emit(sensor_data)
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    if self.running:
                        self.connection_status.emit(f"데이터 수신 오류: {str(e)}")
                    break
                    
        except Exception as e:
            self.connection_status.emit(f"클라이언트 처리 오류: {str(e)}")
        finally:
            if self.client_socket:
                try:
                    self.client_socket.close()
                    self.connection_status.emit("클라이언트 연결 종료됨")
                except:
                    pass
                self.client_socket = None
            
    def stop_receiving(self):
        """서버 중지"""
        self.running = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
            
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
            
        self.connection_status.emit("서버 중지됨")

class GaitEventDetector:
    """보행 이벤트 감지 알고리즘"""
    
    def __init__(self, sampling_rate=None):
        self.sampling_rate = sampling_rate or SENSOR_CONFIG['sampling_rate']
        self.dt = 1.0 / self.sampling_rate
        
        # 설정 파일에서 필터 설정 로드
        self.update_filter_settings()
        
        # 데이터 버퍼
        buffer_size = self.sampling_rate * SENSOR_CONFIG['buffer_duration']
        self.acc_y_buffer = collections.deque(maxlen=buffer_size)
        self.acc_y_detrended = collections.deque(maxlen=buffer_size)  # 중력 제거된 가속도
        self.timestamp_buffer = collections.deque(maxlen=buffer_size)
        self.integration_positive = collections.deque(maxlen=buffer_size)
        self.integration_negative = collections.deque(maxlen=buffer_size)
        
        # 중력 성분 제거를 위한 이동 평균 창
        self.gravity_window_size = int(self.sampling_rate * 2.0)  # 2초 창
        self.gravity_buffer = collections.deque(maxlen=self.gravity_window_size)
        
        # 감지된 이벤트
        self.hs_events = []  # (timestamp, value)
        self.to_events = []  # (timestamp, value)
        
        # 적분값 초기화
        self.integral_pos = 0.0
        self.integral_neg = 0.0
        
        # 적분 drift 보정을 위한 변수
        self.integral_reset_interval = self.sampling_rate * 5  # 5초마다 리셋
        self.sample_count = 0
        
        self.debug_counter = 0
        
    def update_filter_settings(self):
        """설정 파일에서 필터 설정 업데이트"""
        filter_config = SIGNAL_PROCESSING['filter']
        peak_config = PEAK_DETECTION
        
        # 버터워스 필터 설정 - rad/s를 Hz로 변환
        cutoff_freq_rad_per_sec = filter_config['cutoff_frequency_rad']
        self.cutoff_freq_hz = cutoff_freq_rad_per_sec / (2 * np.pi)
        self.filter_order = filter_config['filter_order']
        
        # 디버깅 정보 출력
        if DEBUG_CONFIG['enable_debug']:
            nyquist = self.sampling_rate / 2
            normalized_cutoff = self.cutoff_freq_hz / nyquist
            print(f"[DEBUG] 필터 설정 업데이트:")
            print(f"  - 차단 주파수(rad/s): {cutoff_freq_rad_per_sec:.3f}")
            print(f"  - 차단 주파수(Hz): {self.cutoff_freq_hz:.3f}")
            print(f"  - 정규화된 차단 주파수: {normalized_cutoff:.3f}")
            print(f"  - 필터 차수: {self.filter_order}")
        
        # 피크 감지 설정
        self.hs_min_prominence = peak_config['hs_detection']['min_prominence']
        self.hs_min_distance = int(peak_config['hs_detection']['min_distance_seconds'] * self.sampling_rate)
        
        self.to_min_prominence = peak_config['to_detection']['min_prominence']
        self.to_min_distance = int(peak_config['to_detection']['min_distance_seconds'] * self.sampling_rate)
        
    def add_data(self, timestamp, acc_y):
        """새로운 데이터 포인트 추가"""
        self.timestamp_buffer.append(timestamp)
        self.acc_y_buffer.append(acc_y)
        self.gravity_buffer.append(acc_y)
        self.sample_count += 1
        
        # 중력 성분 제거 (이동 평균 사용)
        if len(self.gravity_buffer) >= self.gravity_window_size:
            gravity_component = np.mean(self.gravity_buffer)
            acc_y_detrended = acc_y - gravity_component
        else:
            # 충분한 데이터가 없으면 단순히 전체 평균 사용
            if len(self.acc_y_buffer) > 10:
                gravity_component = np.mean(list(self.acc_y_buffer)[-min(len(self.acc_y_buffer), 50):])
                acc_y_detrended = acc_y - gravity_component
            else:
                acc_y_detrended = acc_y
        
        self.acc_y_detrended.append(acc_y_detrended)
        
        # 적분 계산 (중력 제거된 가속도 사용)
        if len(self.acc_y_detrended) > 1:
            dt = timestamp - self.timestamp_buffer[-2]
            
            # 양의 적분 (TO 감지용)
            self.integral_pos += acc_y_detrended * dt
            self.integration_positive.append(self.integral_pos)
            
            # 음의 적분 (HS 감지용)
            self.integral_neg += (-acc_y_detrended) * dt
            self.integration_negative.append(self.integral_neg)
        else:
            self.integration_positive.append(0.0)
            self.integration_negative.append(0.0)
        
        # 적분 drift 보정 (주기적으로 적분값 리셋)
        if self.sample_count % self.integral_reset_interval == 0:
            self._reset_integration_drift()
        
        # 충분한 데이터가 있을 때 이벤트 감지
        if len(self.acc_y_detrended) >= self.sampling_rate * 2:  # 2초 데이터
            self._detect_events()
    
    def _reset_integration_drift(self):
        """적분 drift 보정"""
        if len(self.integration_positive) > 0 and len(self.integration_negative) > 0:
            # 최근 적분값들의 평균을 빼서 drift 제거
            recent_window = min(self.sampling_rate, len(self.integration_positive))
            pos_recent = list(self.integration_positive)[-recent_window:]
            neg_recent = list(self.integration_negative)[-recent_window:]
            
            pos_offset = np.mean(pos_recent)
            neg_offset = np.mean(neg_recent)
            
            # 적분값 보정
            self.integral_pos -= pos_offset
            self.integral_neg -= neg_offset
            
            # 버퍼의 모든 값에서 offset 제거
            for i in range(len(self.integration_positive)):
                self.integration_positive[i] -= pos_offset
            for i in range(len(self.integration_negative)):
                self.integration_negative[i] -= neg_offset
            
            if DEBUG_CONFIG['enable_debug']:
                print(f"[DEBUG] 적분 drift 보정: pos_offset={pos_offset:.3f}, neg_offset={neg_offset:.3f}")
    
    def _detect_events(self):
        """HS와 TO 이벤트 감지"""
        if len(self.integration_positive) < self.sampling_rate:
            return
            
        # 최근 데이터만 사용 (1초)
        window_size = self.sampling_rate
        
        # TO 감지 (양의 적분)
        pos_data = np.array(list(self.integration_positive)[-window_size:])
        pos_timestamps = np.array(list(self.timestamp_buffer)[-window_size:])
        
        if len(pos_data) > self.filter_order * 3:
            pos_filtered = self._apply_butterworth_filter(pos_data)
            to_peaks, to_props = find_peaks(pos_filtered, 
                                          prominence=self.to_min_prominence,
                                          distance=self.to_min_distance)
            
            # 새로운 TO 이벤트 확인
            for peak_idx in to_peaks:
                peak_time = pos_timestamps[peak_idx]
                peak_value = pos_filtered[peak_idx]
                
                # 이미 감지된 이벤트인지 확인
                if not any(abs(event[0] - peak_time) < 0.05 for event in self.to_events):
                    self.to_events.append((peak_time, peak_value))
                    self._debug_print(f"TO 감지: 시간={peak_time:.2f}초, 값={peak_value:.3f}")
        
        # HS 감지 (음의 적분)
        neg_data = np.array(list(self.integration_negative)[-window_size:])
        neg_timestamps = np.array(list(self.timestamp_buffer)[-window_size:])
        
        if len(neg_data) > self.filter_order * 3:
            neg_filtered = self._apply_butterworth_filter(neg_data)
            hs_peaks, hs_props = find_peaks(neg_filtered,
                                          prominence=self.hs_min_prominence,
                                          distance=self.hs_min_distance)
            
            # 새로운 HS 이벤트 확인
            for peak_idx in hs_peaks:
                peak_time = neg_timestamps[peak_idx]
                peak_value = neg_filtered[peak_idx]
                
                # 이미 감지된 이벤트인지 확인
                if not any(abs(event[0] - peak_time) < 0.05 for event in self.hs_events):
                    self.hs_events.append((peak_time, peak_value))
                    self._debug_print(f"HS 감지: 시간={peak_time:.2f}초, 값={peak_value:.3f}")
    
    def _apply_butterworth_filter(self, data):
        """버터워스 필터 적용"""
        try:
            # 데이터 길이 체크 - 필터 차수의 3배 이상 필요
            min_length = self.filter_order * 3
            if len(data) < min_length:
                if DEBUG_CONFIG['enable_debug']:
                    print(f"[DEBUG] 필터 적용 스킵: 데이터 길이 {len(data)} < 최소 필요 길이 {min_length}")
                return data
            
            nyquist = self.sampling_rate / 2
            normalized_cutoff = self.cutoff_freq_hz / nyquist
            
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.99
            
            # padlen 파라미터를 데이터 길이에 맞게 조정
            padlen = min(self.filter_order * 3, len(data) // 4)
            if padlen < 1:
                padlen = None
                
            b, a = butter(self.filter_order, normalized_cutoff, btype='low')
            
            if padlen is not None:
                filtered_data = filtfilt(b, a, data, padlen=padlen)
            else:
                # padlen이 너무 작으면 사용하지 않음
                filtered_data = filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            if DEBUG_CONFIG['enable_debug']:
                print(f"[DEBUG] 필터 적용 오류: {e}, 데이터 길이: {len(data)}")
            return data
    
    def _debug_print(self, message):
        """디버깅 메시지 출력 (제한적으로)"""
        if not DEBUG_CONFIG['enable_debug']:
            return
            
        self.debug_counter += 1
        if self.debug_counter % DEBUG_CONFIG['debug_interval'] == 0:
            print(f"[DEBUG] {message}")
    
    def get_filtered_data(self):
        """필터링된 데이터 반환"""
        if len(self.integration_positive) < self.filter_order * 3:
            return np.array([]), np.array([])
            
        pos_data = np.array(list(self.integration_positive))
        neg_data = np.array(list(self.integration_negative))
        
        pos_filtered = self._apply_butterworth_filter(pos_data)
        neg_filtered = self._apply_butterworth_filter(neg_data)
        
        return pos_filtered, neg_filtered

class MainWindow(QMainWindow):
    """메인 GUI 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("보행 이벤트 감지 시스템")
        window_size = GUI_CONFIG['window_size']
        self.setGeometry(100, 100, window_size[0], window_size[1])
        
        # 데이터 수신기와 감지기 초기화
        self.data_receiver = DataReceiver()
        self.gait_detector = GaitEventDetector()
        
        # 데이터 버퍼
        buffer_seconds = GUI_CONFIG['plot_buffer_seconds']
        self.plot_buffer_size = SENSOR_CONFIG['sampling_rate'] * buffer_seconds
        self.time_data = collections.deque(maxlen=self.plot_buffer_size)
        self.acc_y_data = collections.deque(maxlen=self.plot_buffer_size)
        
        # GUI 설정
        self.setup_ui()
        self.setup_connections()
        
        # 데이터 수신 스레드
        self.receiver_thread = QThread()
        self.data_receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.data_receiver.start_receiving)
        
        # 업데이트 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(GUI_CONFIG['update_interval_ms'])
        
        # 상태 변수
        self.is_recording = False
        self.start_time = None
        
    def setup_ui(self):
        """UI 구성"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 제어 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 설정 패널
        settings_panel = self.create_settings_panel()
        layout.addWidget(settings_panel)
        
        # 그래프 영역
        self.setup_plots(layout)
        
        # 상태 및 로그 영역
        status_layout = QHBoxLayout()
        
        # 상태 표시
        self.status_label = QLabel("상태: 대기 중")
        self.status_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.status_label)
        
        # 이벤트 카운터
        self.event_counter_label = QLabel("HS: 0, TO: 0")
        self.event_counter_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.event_counter_label)
        
        layout.addLayout(status_layout)
        
        # 로그 영역
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
    
    def create_control_panel(self):
        """제어 패널 생성"""
        group = QGroupBox("제어")
        layout = QGridLayout(group)
        
        # 연결/시작/중지 버튼
        self.connect_btn = QPushButton("서버 시작")
        self.connect_btn.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_btn, 0, 0)
        
        self.record_btn = QPushButton("기록 시작")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        layout.addWidget(self.record_btn, 0, 1)
        
        # 스크린샷 버튼
        self.screenshot_btn = QPushButton("스크린샷 저장")
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        layout.addWidget(self.screenshot_btn, 0, 2)
        
        # 클리어 버튼
        self.clear_btn = QPushButton("데이터 클리어")
        self.clear_btn.clicked.connect(self.clear_data)
        layout.addWidget(self.clear_btn, 0, 3)
        
        return group
    
    def create_settings_panel(self):
        """설정 패널 생성"""
        group = QGroupBox("설정")
        layout = QGridLayout(group)
        
        # 프리셋 선택
        layout.addWidget(QLabel("보행 프리셋:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(FILTER_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        layout.addWidget(self.preset_combo, 0, 1)
        
        # 필터 차단 주파수
        layout.addWidget(QLabel("차단 주파수 (rad/s):"), 0, 2)
        self.cutoff_spinbox = QDoubleSpinBox()
        self.cutoff_spinbox.setRange(0.1, 50.0)
        self.cutoff_spinbox.setSingleStep(0.1)
        self.cutoff_spinbox.setDecimals(1)
        self.cutoff_spinbox.setValue(SIGNAL_PROCESSING['filter']['cutoff_frequency_rad'])
        self.cutoff_spinbox.valueChanged.connect(self.on_settings_changed)
        layout.addWidget(self.cutoff_spinbox, 0, 3)
        
        # Prominence 설정
        layout.addWidget(QLabel("Prominence:"), 1, 0)
        self.prominence_spinbox = QDoubleSpinBox()
        self.prominence_spinbox.setRange(0.01, 1.0)
        self.prominence_spinbox.setSingleStep(0.01)
        self.prominence_spinbox.setDecimals(3)
        self.prominence_spinbox.setValue(PEAK_DETECTION['hs_detection']['min_prominence'])
        self.prominence_spinbox.valueChanged.connect(self.on_settings_changed)
        layout.addWidget(self.prominence_spinbox, 1, 1)
        
        # 최소 거리 설정
        layout.addWidget(QLabel("최소 거리 (초):"), 1, 2)
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(0.1, 2.0)
        self.distance_spinbox.setSingleStep(0.1)
        self.distance_spinbox.setDecimals(1)
        self.distance_spinbox.setValue(PEAK_DETECTION['hs_detection']['min_distance_seconds'])
        self.distance_spinbox.valueChanged.connect(self.on_settings_changed)
        layout.addWidget(self.distance_spinbox, 1, 3)
        
        return group
    
    def setup_plots(self, layout):
        """그래프 설정"""
        colors = GUI_CONFIG['colors']
        
        # 가속도 그래프
        self.acc_plot = pg.PlotWidget(title="수직 가속도 (acc_y)")
        self.acc_plot.setLabel('left', '가속도', units='m/s²')
        self.acc_plot.setLabel('bottom', '경과 시간', units='초')
        self.acc_plot.showGrid(True, True)
        
        self.acc_curve = self.acc_plot.plot(pen=colors['acc_y'], name='원본 acc_y')
        self.acc_detrended_curve = self.acc_plot.plot(pen='orange', name='중력 제거된 acc_y')
        
        # 범례 추가
        self.acc_plot.addLegend()
        layout.addWidget(self.acc_plot)
        
        # 적분 그래프
        self.integral_plot = pg.PlotWidget(title="적분값 및 필터링된 신호")
        self.integral_plot.setLabel('left', '적분값', units='m/s')
        self.integral_plot.setLabel('bottom', '경과 시간', units='초')
        self.integral_plot.showGrid(True, True)
        
        self.pos_integral_curve = self.integral_plot.plot(pen=colors['pos_integral'], name='양의 적분 (TO)')
        self.neg_integral_curve = self.integral_plot.plot(pen=colors['neg_integral'], name='음의 적분 (HS)')
        self.pos_filtered_curve = self.integral_plot.plot(pen=colors['pos_filtered'], name='필터링됨 (TO)')
        self.neg_filtered_curve = self.integral_plot.plot(pen=colors['neg_filtered'], name='필터링됨 (HS)')
        
        # 범례 추가
        self.integral_plot.addLegend()
        
        layout.addWidget(self.integral_plot)
    
    def setup_connections(self):
        """신호 연결 설정"""
        self.data_receiver.data_received.connect(self.on_data_received)
        self.data_receiver.connection_status.connect(self.on_connection_status)
    
    def on_preset_changed(self, preset_name):
        """프리셋 변경 처리"""
        update_config_from_preset(preset_name)
        
        # UI 업데이트 - rad/s 단위로 올바르게 표시
        self.cutoff_spinbox.setValue(SIGNAL_PROCESSING['filter']['cutoff_frequency_rad'])
        self.prominence_spinbox.setValue(PEAK_DETECTION['hs_detection']['min_prominence'])
        self.distance_spinbox.setValue(PEAK_DETECTION['hs_detection']['min_distance_seconds'])
        
        # 감지기 설정 업데이트
        self.gait_detector.update_filter_settings()
        
        # 현재 설정값 디버깅 출력
        if DEBUG_CONFIG['enable_debug']:
            cutoff_hz = SIGNAL_PROCESSING['filter']['cutoff_frequency_rad'] / (2 * np.pi)
            print(f"[DEBUG] 프리셋 '{preset_name}' 적용됨:")
            print(f"  - 차단 주파수: {SIGNAL_PROCESSING['filter']['cutoff_frequency_rad']:.1f} rad/s ({cutoff_hz:.3f} Hz)")
            print(f"  - Prominence: {PEAK_DETECTION['hs_detection']['min_prominence']:.3f}")
            print(f"  - 최소 거리: {PEAK_DETECTION['hs_detection']['min_distance_seconds']:.1f}초")
        
        self.log_message(f"프리셋 '{preset_name}'이 적용되었습니다.")
    
    def on_settings_changed(self):
        """수동 설정 변경 처리"""
        # 설정값 업데이트
        SIGNAL_PROCESSING['filter']['cutoff_frequency_rad'] = self.cutoff_spinbox.value()
        PEAK_DETECTION['hs_detection']['min_prominence'] = self.prominence_spinbox.value()
        PEAK_DETECTION['to_detection']['min_prominence'] = self.prominence_spinbox.value()
        PEAK_DETECTION['hs_detection']['min_distance_seconds'] = self.distance_spinbox.value()
        PEAK_DETECTION['to_detection']['min_distance_seconds'] = self.distance_spinbox.value()
        
        # 감지기 설정 업데이트
        self.gait_detector.update_filter_settings()
        
        self.log_message("설정이 업데이트되었습니다.")
    
    def toggle_connection(self):
        """서버 시작/중지"""
        if not self.receiver_thread.isRunning():
            self.receiver_thread.start()
            self.connect_btn.setText("서버 중지")
            self.record_btn.setEnabled(True)
        else:
            self.data_receiver.stop_receiving()
            self.receiver_thread.quit()
            self.receiver_thread.wait()
            self.connect_btn.setText("서버 시작")
            self.record_btn.setEnabled(False)
            self.is_recording = False
            self.record_btn.setText("기록 시작")
    
    def toggle_recording(self):
        """기록 시작/중지"""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText("기록 중지")
            self.gait_detector = GaitEventDetector()  # 리셋
            self.clear_data()
        else:
            self.record_btn.setText("기록 시작")
    
    def on_data_received(self, data):
        """데이터 수신 처리"""
        try:
            # 상대적 시간으로 변환 (앱 시작 시점부터의 경과 시간)
            elapsed_time = data.get('timestamp', 0)
            relative_timestamp = elapsed_time  # 이미 경과 시간이므로 그대로 사용
            
            target_axis = SENSOR_CONFIG['target_axis']
            acc_target = data['accel'][target_axis]  # 설정된 축의 가속도
            
            # 데이터 버퍼에 추가
            self.time_data.append(relative_timestamp)
            self.acc_y_data.append(acc_target)
            
            # 기록 중일 때만 보행 이벤트 감지
            if self.is_recording:
                self.gait_detector.add_data(relative_timestamp, acc_target)
            
        except KeyError as e:
            self.log_message(f"데이터 형식 오류: {e}")
    
    def on_connection_status(self, status):
        """연결 상태 업데이트"""
        self.status_label.setText(f"상태: {status}")
        self.log_message(status)
    
    def update_plots(self):
        """그래프 업데이트"""
        if len(self.time_data) < 2:
            return
            
        # 가속도 그래프 업데이트
        time_array = np.array(self.time_data)
        acc_array = np.array(self.acc_y_data)
        self.acc_curve.setData(time_array, acc_array)
        
        # 중력 제거된 가속도 표시
        if len(self.gait_detector.acc_y_detrended) > 0 and len(self.gait_detector.timestamp_buffer) > 0:
            detrended_timestamps = np.array(self.gait_detector.timestamp_buffer)
            detrended_acc = np.array(self.gait_detector.acc_y_detrended)
            
            # 시간 범위 맞추기
            if len(detrended_timestamps) == len(detrended_acc):
                self.acc_detrended_curve.setData(detrended_timestamps, detrended_acc)
        
        # 이벤트 마커 설정
        colors = GUI_CONFIG['colors']
        markers = GUI_CONFIG['markers']
        
        # HS 이벤트 표시
        for event_time, event_value in self.gait_detector.hs_events:
            if event_time >= time_array[0] and event_time <= time_array[-1]:
                # 해당 시간의 가속도 값 찾기
                idx = np.argmin(np.abs(time_array - event_time))
                acc_value = acc_array[idx]
                self.acc_plot.plot([event_time], [acc_value], pen=None, 
                                 symbol=markers['hs_symbol'], 
                                 symbolBrush=colors['hs_marker'], 
                                 symbolSize=markers['marker_size'])
        
        # TO 이벤트 표시  
        for event_time, event_value in self.gait_detector.to_events:
            if event_time >= time_array[0] and event_time <= time_array[-1]:
                idx = np.argmin(np.abs(time_array - event_time))
                acc_value = acc_array[idx]
                self.acc_plot.plot([event_time], [acc_value], pen=None,
                                 symbol=markers['to_symbol'], 
                                 symbolBrush=colors['to_marker'], 
                                 symbolSize=markers['marker_size'])
        
        # 적분 그래프 업데이트
        if len(self.gait_detector.timestamp_buffer) > 0:
            timestamps = np.array(self.gait_detector.timestamp_buffer)
            pos_integral = np.array(self.gait_detector.integration_positive)
            neg_integral = np.array(self.gait_detector.integration_negative)
            
            self.pos_integral_curve.setData(timestamps, pos_integral)
            self.neg_integral_curve.setData(timestamps, neg_integral)
            
            # 필터링된 데이터
            pos_filtered, neg_filtered = self.gait_detector.get_filtered_data()
            if len(pos_filtered) > 0:
                self.pos_filtered_curve.setData(timestamps, pos_filtered)
                self.neg_filtered_curve.setData(timestamps, neg_filtered)
        
        # 이벤트 카운터 업데이트
        hs_count = len(self.gait_detector.hs_events)
        to_count = len(self.gait_detector.to_events)
        self.event_counter_label.setText(f"HS: {hs_count}, TO: {to_count}")
        
        # 케이던스 계산 (보폭당 분)
        if hs_count >= 2:
            recent_hs = [event[0] for event in self.gait_detector.hs_events[-10:]]  # 최근 10개
            if len(recent_hs) >= 2:
                avg_stride_time = (recent_hs[-1] - recent_hs[0]) / (len(recent_hs) - 1)
                cadence = 60 / avg_stride_time if avg_stride_time > 0 else 0
                self.event_counter_label.setText(f"HS: {hs_count}, TO: {to_count}, 케이던스: {cadence:.1f}/분")
    
    def save_screenshot(self):
        """스크린샷 저장"""
        timestamp = datetime.now().strftime(OUTPUT_CONFIG['filename_timestamp_format'])
        format_ext = OUTPUT_CONFIG['screenshot_format']
        
        # 가속도 그래프 저장
        exporter = pg.exporters.ImageExporter(self.acc_plot.plotItem)
        exporter.export(f'gait_acc_{timestamp}.{format_ext}')
        
        # 적분 그래프 저장
        exporter = pg.exporters.ImageExporter(self.integral_plot.plotItem)
        exporter.export(f'gait_integral_{timestamp}.{format_ext}')
        
        self.log_message(f"스크린샷 저장됨: gait_*_{timestamp}.{format_ext}")
    
    def clear_data(self):
        """데이터 클리어"""
        self.time_data.clear()
        self.acc_y_data.clear()
        self.gait_detector.hs_events.clear()
        self.gait_detector.to_events.clear()
        self.log_message("데이터가 클리어되었습니다.")
    
    def log_message(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """윈도우 종료 처리"""
        if self.receiver_thread.isRunning():
            self.data_receiver.stop_receiving()
            self.receiver_thread.quit()
            self.receiver_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 