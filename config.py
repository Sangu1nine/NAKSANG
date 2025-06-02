#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
보행 이벤트 감지 시스템 설정 파일
MODIFIED [2024-12]: 시스템 매개변수 중앙 관리

이 파일에서 필터링, 피크 감지, 네트워크 설정 등을 조정할 수 있습니다.
"""

# 네트워크 설정
NETWORK_CONFIG = {
    'server_ip': '192.168.0.177',  # IMU 센서 데이터 송신 IP
    'server_port': 5000,           # 통신 포트
    'timeout': 10                  # 연결 타임아웃 (초)
}

# 센서 설정
SENSOR_CONFIG = {
    'sampling_rate': 100,          # 샘플링 레이트 (Hz)
    'target_axis': 'y',            # 보행 감지에 사용할 축 ('x', 'y', 'z')
    'buffer_duration': 30          # 데이터 버퍼 지속시간 (초)
}

# 신호 처리 설정
SIGNAL_PROCESSING = {
    # 버터워스 필터 설정
    'filter': {
        'cutoff_frequency_rad': 12,  # 차단 주파수 (rad)
        'filter_order': 4,             # 필터 차수
        'type': 'low'                  # 필터 타입 ('low', 'high', 'band')
    },
    
    # 적분 계산 설정
    'integration': {
        'method': 'trapezoidal',       # 적분 방법 ('trapezoidal', 'simpson')
        'detrend': True                # 선형 추세 제거 여부
    }
}

# 피크 감지 설정
PEAK_DETECTION = {
    'hs_detection': {  # Heel Strike 감지
        'min_prominence': 0.1,         # 최소 prominence (m/s)
        'min_distance_seconds': 0.3,   # 최소 피크 간격 (초)
        'min_height': None,            # 최소 피크 높이 (None은 자동)
        'threshold': None              # 임계값 (None은 자동)
    },
    
    'to_detection': {  # Toe Off 감지
        'min_prominence': 0.1,         # 최소 prominence (m/s)
        'min_distance_seconds': 0.3,   # 최소 피크 간격 (초)
        'min_height': None,            # 최소 피크 높이 (None은 자동)
        'threshold': None              # 임계값 (None은 자동)
    }
}

# GUI 설정
GUI_CONFIG = {
    'window_size': (1400, 900),        # 창 크기 (너비, 높이)
    'update_interval_ms': 50,          # 그래프 업데이트 간격 (밀리초)
    'plot_buffer_seconds': 10,         # 그래프에 표시할 데이터 시간 (초)
    
    'colors': {
        'acc_y': 'blue',               # 가속도 신호 색상
        'pos_integral': 'green',       # 양의 적분 색상
        'neg_integral': 'red',         # 음의 적분 색상
        'pos_filtered': 'cyan',        # 필터링된 양의 적분 색상
        'neg_filtered': 'magenta',     # 필터링된 음의 적분 색상
        'hs_marker': 'red',            # HS 마커 색상
        'to_marker': 'green'           # TO 마커 색상
    },
    
    'markers': {
        'hs_symbol': 'o',              # HS 마커 심볼
        'to_symbol': 's',              # TO 마커 심볼
        'marker_size': 8               # 마커 크기
    }
}

# 디버깅 설정
DEBUG_CONFIG = {
    'enable_debug': True,              # 디버깅 출력 활성화
    'debug_interval': 10,              # 디버깅 출력 간격 (매 N번째마다)
    'save_debug_data': False,          # 디버깅 데이터 파일 저장 여부
    'debug_data_path': 'debug_data'    # 디버깅 데이터 저장 경로
}

# 파일 출력 설정
OUTPUT_CONFIG = {
    'screenshot_format': 'png',        # 스크린샷 형식 ('png', 'jpg', 'svg')
    'screenshot_dpi': 300,             # 스크린샷 해상도 (DPI)
    'data_export_format': 'csv',       # 데이터 내보내기 형식 ('csv', 'json', 'pickle')
    'filename_timestamp_format': '%Y%m%d_%H%M%S'  # 파일명 타임스탬프 형식
}

# 보행 분석 설정 (향후 확장용)
GAIT_ANALYSIS = {
    'stride_detection': {
        'min_stride_time': 0.8,        # 최소 보폭 시간 (초)
        'max_stride_time': 2.0,        # 최대 보폭 시간 (초)
    },
    
    'cadence_calculation': {
        'window_size': 10,             # 케이던스 계산 윈도우 크기 (초)
        'update_interval': 1           # 케이던스 업데이트 간격 (초)
    }
}

# 보행 환경별 필터 프리셋
FILTER_PRESETS = {
    'normal_walking': {
        'cutoff_frequency_rad': 12,
        'filter_order': 4,
        'min_prominence': 0.1,
        'min_distance_seconds': 0.3
    },
    
    'fast_walking': {
        'cutoff_frequency_rad': 15,
        'filter_order': 4,
        'min_prominence': 0.08,
        'min_distance_seconds': 0.25
    },
    
    'slow_walking': {
        'cutoff_frequency_rad': 4.0,
        'filter_order': 6,
        'min_prominence': 0.15,
        'min_distance_seconds': 0.4
    },
    
    'pathological_gait': {
        'cutoff_frequency_rad': 2.0,
        'filter_order': 6,
        'min_prominence': 0.05,
        'min_distance_seconds': 0.2
    }
}

def get_preset_config(preset_name='normal_walking'):
    """
    특정 프리셋 설정을 반환합니다.
    
    Args:
        preset_name (str): 프리셋 이름
        
    Returns:
        dict: 프리셋 설정
    """
    if preset_name in FILTER_PRESETS:
        return FILTER_PRESETS[preset_name]
    else:
        print(f"경고: '{preset_name}' 프리셋을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        return FILTER_PRESETS['normal_walking']

def update_config_from_preset(preset_name):
    """
    프리셋을 사용하여 현재 설정을 업데이트합니다.
    
    Args:
        preset_name (str): 사용할 프리셋 이름
    """
    preset = get_preset_config(preset_name)
    
    # 신호 처리 설정 업데이트
    SIGNAL_PROCESSING['filter']['cutoff_frequency_rad'] = preset['cutoff_frequency_rad']
    SIGNAL_PROCESSING['filter']['filter_order'] = preset['filter_order']
    
    # 피크 감지 설정 업데이트
    PEAK_DETECTION['hs_detection']['min_prominence'] = preset['min_prominence']
    PEAK_DETECTION['hs_detection']['min_distance_seconds'] = preset['min_distance_seconds']
    PEAK_DETECTION['to_detection']['min_prominence'] = preset['min_prominence']
    PEAK_DETECTION['to_detection']['min_distance_seconds'] = preset['min_distance_seconds']
    
    print(f"설정이 '{preset_name}' 프리셋으로 업데이트되었습니다.") 