# 라즈베리파이용 실시간 보행 감지기
# 🎯 KFall 데이터셋 분석 결과 적용됨 (보행 vs 일상활동)
# 📝 낙상 감지는 별도 딥러닝 모델에서 처리
# 
# 📊 ROC 분석 결과 (AUC 기준 상위 특징):
#    - acc_range: 0.843 (가속도 범위)
#    - acc_std: 0.835 (가속도 표준편차) 
#    - walking_energy_ratio: 0.833 (보행 주파수 에너지 비율)
#    - gyr_mean: 0.780 (자이로 평균)
# 
# 🎯 최적화 성능: F1 Score = 0.641
# 📈 데이터: 32명 피험자, 21,696개 윈도우 (보행 42.7%, 비보행 57.3%)
#
# MODIFIED [2024-12-19]: 임계값 구조 전면 개선 - 모든 파라미터 설정화, 가중치 시스템 체계화
# UPDATED [2024-12-19]: ROC 분석 기반 최적 임계값 적용
import numpy as np
from collections import deque
import time

class WalkingDetector:
    def __init__(self, config=None):
        # 데이터 버퍼 (2초 = 200샘플 @ 100Hz)
        self.buffer_size = 200
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)

        # 보행 상태
        self.is_walking = False
        self.confidence = 0.0

        # 🔥 개선된 설정 시스템 - 모든 파라미터 설정 가능
        self.config = self._load_config(config)
        
        print("🎯 개선된 보행 감지 파라미터 로드됨:")
        print("📊 임계값:")
        for key, value in self.config['thresholds'].items():
            print(f"   {key}: {value}")
        print("⚖️ 가중치:")
        for key, value in self.config['weights'].items():
            print(f"   {key}: {value}")
        print("🔧 필터링:")
        for key, value in self.config['filtering'].items():
            print(f"   {key}: {value}")

    def _load_config(self, config):
        """설정 로드 및 기본값 설정"""
        default_config = {
            'thresholds': {
                # 🎯 ROC 분석 기반 최적화된 임계값
                'acc_mean_min': 0.918,        # 기존 1.022 → 0.918
                'acc_mean_max': 1.122,        # 기존 1.126 → 1.122  
                'acc_std_min': 0.134,         # 기존 0.208 → 0.134
                
                # 보행 주기 관련 임계값
                'step_freq_min': 1.0,
                'step_freq_max': 4.0,
                'regularity_min': 0.869,      # 기존 0.417 → 0.869 (더 엄격)
                
                # 피크 검출 관련 임계값
                'peak_detection_factor': 0.3,
                'peak_window_size': 5,
                
                # 최종 판단 임계값
                'confidence_min': 0.6
            },
            'weights': {
                # 🎯 F1 스코어 최적화된 가중치 (합계 = 1.0)
                'acc_mean_weight': 0.25,      # 동일 유지
                'acc_std_weight': 0.25,       # 동일 유지
                'step_freq_weight': 0.35,     # 동일 유지
                'regularity_weight': 0.15     # 동일 유지
            },
            'filtering': {
                # 필터링 관련 파라미터
                'moving_avg_window': 5,
                'min_peaks_required': 2
            }
        }
        
        # 사용자 설정이 있으면 기본값과 병합
        if config:
            for category in default_config:
                if category in config:
                    default_config[category].update(config[category])
        
        # 가중치 합계 검증
        total_weight = sum(default_config['weights'].values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ 경고: 가중치 합계가 1.0이 아닙니다 ({total_weight:.3f})")
            # 자동 정규화
            for key in default_config['weights']:
                default_config['weights'][key] /= total_weight
            print("✅ 가중치가 자동으로 정규화되었습니다.")
        
        return default_config

    def add_data(self, acc_x, acc_y, acc_z, timestamp):
        """새 센서 데이터 추가"""
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        self.acc_buffer.append(acc_magnitude)
        self.time_buffer.append(timestamp)

        # 충분한 데이터가 있으면 분석
        if len(self.acc_buffer) >= self.buffer_size:
            self._analyze()

        return self.is_walking, self.confidence

    def _analyze(self):
        """개선된 보행 분석"""
        acc_data = np.array(self.acc_buffer)
        time_data = np.array(self.time_buffer)

        # 설정 가능한 이동평균 필터링
        window_size = self.config['filtering']['moving_avg_window']
        acc_smooth = np.convolve(acc_data, np.ones(window_size)/window_size, mode='same')

        # 기본 특징 계산
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)

        # 개선된 피크 검출
        peak_factor = self.config['thresholds']['peak_detection_factor']
        peak_window = self.config['thresholds']['peak_window_size']
        threshold = np.mean(acc_smooth) + peak_factor * np.std(acc_smooth)
        
        peaks = []
        for i in range(peak_window, len(acc_smooth) - peak_window):
            if (acc_smooth[i] > threshold and 
                acc_smooth[i] == max(acc_smooth[i-peak_window:i+peak_window+1])):
                peaks.append(i)

        # 보행 주기 및 규칙성 계산
        step_frequency = 0
        regularity = 0
        if len(peaks) >= self.config['filtering']['min_peaks_required']:
            peak_times = time_data[peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                step_frequency = 1.0 / np.mean(intervals)
                # 개선된 규칙성 계산 (표준편차가 작을수록 규칙적)
                regularity = 1.0 / (1.0 + np.std(intervals))

        # 개선된 신뢰도 계산 시스템
        confidence_scores = {}
        
        # 1. 가속도 평균 검사
        if (self.config['thresholds']['acc_mean_min'] <= acc_mean <= 
            self.config['thresholds']['acc_mean_max']):
            confidence_scores['acc_mean'] = self.config['weights']['acc_mean_weight']
        else:
            confidence_scores['acc_mean'] = 0.0
            
        # 2. 가속도 표준편차 검사
        if acc_std >= self.config['thresholds']['acc_std_min']:
            confidence_scores['acc_std'] = self.config['weights']['acc_std_weight']
        else:
            confidence_scores['acc_std'] = 0.0
            
        # 3. 보행 주기 검사
        if (self.config['thresholds']['step_freq_min'] <= step_frequency <= 
            self.config['thresholds']['step_freq_max']):
            confidence_scores['step_freq'] = self.config['weights']['step_freq_weight']
        else:
            confidence_scores['step_freq'] = 0.0
            
        # 4. 규칙성 검사 (새로 추가!)
        if regularity >= self.config['thresholds']['regularity_min']:
            confidence_scores['regularity'] = self.config['weights']['regularity_weight']
        else:
            confidence_scores['regularity'] = 0.0

        # 최종 신뢰도 계산
        self.confidence = sum(confidence_scores.values())
        
        # 상태 업데이트
        self.is_walking = self.confidence >= self.config['thresholds']['confidence_min']
        
        # 디버깅 정보 저장 (선택적)
        self._last_analysis = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'step_frequency': step_frequency,
            'regularity': regularity,
            'peaks_count': len(peaks),
            'confidence_breakdown': confidence_scores
        }

    def get_status(self):
        """현재 상태 반환"""
        return {
            'walking': self.is_walking,
            'confidence': self.confidence
        }
    
    def get_detailed_status(self):
        """상세 상태 정보 반환 (디버깅용)"""
        status = self.get_status()
        if hasattr(self, '_last_analysis'):
            status.update(self._last_analysis)
        return status
    
    def update_config(self, new_config):
        """실시간 설정 업데이트"""
        self.config = self._load_config(new_config)
        print("✅ 설정이 업데이트되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 기본 설정으로 초기화
    detector = WalkingDetector()
    
    # 또는 커스텀 설정으로 초기화
    # custom_config = {
    #     'thresholds': {
    #         'confidence_min': 0.7  # 더 엄격한 판단
    #     },
    #     'weights': {
    #         'step_freq_weight': 0.5,  # 보행 주기에 더 높은 가중치
    #         'acc_mean_weight': 0.2,
    #         'acc_std_weight': 0.2,
    #         'regularity_weight': 0.1
    #     }
    # }
    # detector = WalkingDetector(custom_config)

    # 센서 데이터 루프 (구현 필요)
    while True:
        # IMU 센서에서 데이터 읽기
        acc_x, acc_y, acc_z = read_imu_data()  # 실제 센서 함수로 교체
        timestamp = time.time()

        # 보행 감지
        walking, confidence = detector.add_data(acc_x, acc_y, acc_z, timestamp)

        if walking:
            print(f"🚶 보행 중 (신뢰도: {confidence:.3f})")
            
            # 상세 정보 출력 (디버깅 시)
            # detailed = detector.get_detailed_status()
            # print(f"   📊 상세: {detailed}")

        time.sleep(0.01)  # 100Hz
