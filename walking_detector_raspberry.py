# 라즈베리파이용 실시간 보행 감지기
# 🎯 KFall 데이터셋 분석 결과 적용됨
import numpy as np
from collections import deque
import time

class WalkingDetector:
    def __init__(self):
        # 데이터 버퍼 (2초 = 200샘플 @ 100Hz)
        self.buffer_size = 200
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)

        # 보행 상태
        self.is_walking = False
        self.confidence = 0.0

        # 🔥 코랩 분석으로 최적화된 임계값 (실제 KFall 데이터 기반)
        self.thresholds = {
            'acc_mean_min': 1.022,
            'acc_mean_max': 1.126,
            'acc_std_min': 0.208,
            'step_freq_min': 1.0,
            'step_freq_max': 4.0,
            'regularity_min': 0.417,
            'confidence_min': 0.6
        }

        print("🎯 KFall 최적화 파라미터 로드됨:")
        for key, value in self.thresholds.items():
            print(f"   {key}: {value}")

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
        """보행 분석"""
        acc_data = np.array(self.acc_buffer)
        time_data = np.array(self.time_buffer)

        # 이동평균 필터링
        acc_smooth = np.convolve(acc_data, np.ones(5)/5, mode='same')

        # 기본 특징
        acc_mean = np.mean(acc_data)
        acc_std = np.std(acc_data)

        # 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = []
        for i in range(5, len(acc_smooth)-5):
            if acc_smooth[i] > threshold and acc_smooth[i] == max(acc_smooth[i-5:i+6]):
                peaks.append(i)

        # 보행 주기 계산
        step_frequency = 0
        regularity = 0
        if len(peaks) > 1:
            peak_times = time_data[peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                step_frequency = 1.0 / np.mean(intervals)
                regularity = 1.0 / (1.0 + np.std(intervals))

        # 신뢰도 계산
        confidence = 0.0

        if self.thresholds['acc_mean_min'] <= acc_mean <= self.thresholds['acc_mean_max']:
            confidence += 0.3
        if acc_std >= self.thresholds['acc_std_min']:
            confidence += 0.3
        if self.thresholds['step_freq_min'] <= step_frequency <= self.thresholds['step_freq_max']:
            confidence += 0.4

        # 상태 업데이트
        self.confidence = confidence
        self.is_walking = confidence >= self.thresholds['confidence_min']

    def get_status(self):
        """현재 상태 반환"""
        return {
            'walking': self.is_walking,
            'confidence': self.confidence
        }

# 사용 예시
if __name__ == "__main__":
    detector = WalkingDetector()

    # 센서 데이터 루프 (구현 필요)
    while True:
        # IMU 센서에서 데이터 읽기
        acc_x, acc_y, acc_z = read_imu_data()  # 실제 센서 함수로 교체
        timestamp = time.time()

        # 보행 감지
        walking, confidence = detector.add_data(acc_x, acc_y, acc_z, timestamp)

        if walking:
            print(f"🚶 보행 중 (신뢰도: {confidence:.2f})")

        time.sleep(0.01)  # 100Hz
