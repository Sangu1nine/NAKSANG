# =============================================================================
# 구글 코랩용: KFall 데이터 분석 및 파라미터 추출
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# 1단계: 데이터 로딩
# =============================================================================

def load_walking_data(base_path='/content/drive/MyDrive/KFall_dataset/data/walking_data'):
    """KFall 보행 데이터 로딩"""

    walking_activities = ['06', '07', '08', '09', '10', '35', '36']
    all_data = []

    # 모든 피험자 폴더 탐색
    for subject_folder in glob.glob(os.path.join(base_path, 'SA*')):
        subject_id = os.path.basename(subject_folder)

        # CSV 파일들 탐색
        for csv_file in glob.glob(os.path.join(subject_folder, '*.csv')):
            filename = os.path.basename(csv_file)
            try:
                activity_num = filename.split('T')[1][:2]
                if activity_num in walking_activities:
                    df = pd.read_csv(csv_file)
                    df['subject_id'] = subject_id
                    df['activity_num'] = activity_num
                    all_data.append(df)
            except:
                continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ {len(all_data)}개 파일 로드 완료")
        print(f"📊 피험자: {combined_df['subject_id'].nunique()}명")
        print(f"🚶 활동 종류: {sorted(combined_df['activity_num'].unique())}")
        return combined_df
    else:
        print("❌ 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

# =============================================================================
# 2단계: 특징 추출 및 분석
# =============================================================================

def extract_walking_features(data):
    """보행 특징 추출"""

    def analyze_file_group(df_group):
        # 가속도 벡터 크기
        acc_mag = np.sqrt(df_group['AccX']**2 + df_group['AccY']**2 + df_group['AccZ']**2)
        gyr_mag = np.sqrt(df_group['GyrX']**2 + df_group['GyrY']**2 + df_group['GyrZ']**2)

        # 이동평균 필터링
        acc_smooth = np.convolve(acc_mag, np.ones(5)/5, mode='same')

        # 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = []
        for i in range(5, len(acc_smooth)-5):
            if acc_smooth[i] > threshold and acc_smooth[i] == max(acc_smooth[i-5:i+6]):
                peaks.append(i)

        # 특징 계산
        features = {
            'acc_mean': np.mean(acc_mag),
            'acc_std': np.std(acc_mag),
            'gyr_mean': np.mean(gyr_mag),
            'peak_count': len(peaks)
        }

        # 보행 주기 분석
        if len(peaks) > 1:
            time_stamps = df_group['TimeStamp(s)'].values
            peak_times = time_stamps[peaks]
            intervals = np.diff(peak_times)
            features['step_frequency'] = 1.0 / np.mean(intervals)
            features['step_regularity'] = 1.0 / (1.0 + np.std(intervals))
        else:
            features['step_frequency'] = 0
            features['step_regularity'] = 0

        return features

    # 모든 파일 분석
    results = []
    grouped = data.groupby(['subject_id', 'activity_num'])

    for (subject, activity), group in grouped:
        # 파일별로 분리 (같은 subject+activity도 여러 파일 가능)
        for file_group in [group]:  # 실제로는 더 세분화 필요시 추가
            features = analyze_file_group(file_group)
            features['subject_id'] = subject
            features['activity_num'] = activity
            results.append(features)

    return pd.DataFrame(results)

# =============================================================================
# 3단계: 최적 파라미터 계산
# =============================================================================

def calculate_optimal_parameters(features_df):
    """보행 감지용 최적 파라미터 계산"""

    # 기본 통계
    params = {
        'acc_mean_min': features_df['acc_mean'].quantile(0.1),
        'acc_mean_max': features_df['acc_mean'].quantile(0.9),
        'acc_std_min': features_df['acc_std'].quantile(0.2),
        'step_freq_min': 1.0,
        'step_freq_max': 4.0,
        'regularity_min': features_df['step_regularity'].quantile(0.3),
        'confidence_threshold': 0.6
    }

    print("🎯 최적 파라미터:")
    for key, value in params.items():
        print(f"   {key}: {value:.3f}")

    return params

# =============================================================================
# 4단계: 시각화
# =============================================================================

def visualize_patterns(features_df):
    """보행 패턴 시각화"""

    plt.figure(figsize=(12, 8))

    # 활동별 가속도 패턴
    plt.subplot(2, 2, 1)
    sns.boxplot(data=features_df, x='activity_num', y='acc_mean')
    plt.title('활동별 평균 가속도')

    # 보행 주파수
    plt.subplot(2, 2, 2)
    sns.boxplot(data=features_df, x='activity_num', y='step_frequency')
    plt.title('활동별 보행 주파수')

    # 개인차 (처음 6명만)
    plt.subplot(2, 2, 3)
    sample_subjects = features_df['subject_id'].unique()[:6]
    sample_data = features_df[features_df['subject_id'].isin(sample_subjects)]
    sns.boxplot(data=sample_data, x='subject_id', y='acc_mean')
    plt.title('개인별 차이')
    plt.xticks(rotation=45)

    # 특징 분포
    plt.subplot(2, 2, 4)
    plt.scatter(features_df['acc_mean'], features_df['step_frequency'],
                c=features_df['activity_num'].astype('category').cat.codes, alpha=0.6)
    plt.xlabel('평균 가속도')
    plt.ylabel('보행 주파수')
    plt.title('특징 분포')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 5단계: 라즈베리파이 코드 생성
# =============================================================================

def generate_raspberry_code(params):
    """라즈베리파이용 코드 생성 (코랩 분석 결과 반영)"""

    code_template = f'''# 라즈베리파이용 실시간 보행 감지기
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
        self.thresholds = {{
            'acc_mean_min': {params['acc_mean_min']:.3f},
            'acc_mean_max': {params['acc_mean_max']:.3f},
            'acc_std_min': {params['acc_std_min']:.3f},
            'step_freq_min': {params['step_freq_min']:.1f},
            'step_freq_max': {params['step_freq_max']:.1f},
            'regularity_min': {params['regularity_min']:.3f},
            'confidence_min': {params['confidence_threshold']:.1f}
        }}

        print("🎯 KFall 최적화 파라미터 로드됨:")
        for key, value in self.thresholds.items():
            print(f"   {{key}}: {{value}}")

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
        return {{
            'walking': self.is_walking,
            'confidence': self.confidence
        }}

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
            print(f"🚶 보행 중 (신뢰도: {{confidence:.2f}})")

        time.sleep(0.01)  # 100Hz
'''

    return code_template

# =============================================================================
# 전체 실행 함수
# =============================================================================

def run_complete_analysis():
    """전체 분석 실행"""

    print("🚀 KFall 보행 감지 시스템 분석 시작")
    print("="*50)

    # 1단계: 데이터 로딩
    print("\n1️⃣ 데이터 로딩...")
    data = load_walking_data()
    if data is None:
        return

    # 2단계: 특징 추출
    print("\n2️⃣ 특징 추출...")
    features_df = extract_walking_features(data)
    print(f"✅ {len(features_df)}개 샘플 분석 완료")

    # 3단계: 파라미터 최적화
    print("\n3️⃣ 파라미터 최적화...")
    optimal_params = calculate_optimal_parameters(features_df)

    # 4단계: 시각화
    print("\n4️⃣ 패턴 시각화...")
    visualize_patterns(features_df)

    # 5단계: 라즈베리파이 코드 생성
    print("\n5️⃣ 라즈베리파이 코드 생성...")
    raspberry_code = generate_raspberry_code(optimal_params)

    # 파일 저장
    with open('/content/drive/MyDrive/KFall_dataset/data/walking_data/walking_detector_raspberry.py', 'w', encoding='utf-8') as f:
        f.write(raspberry_code)

    print("✅ 완료!")
    print("📁 라즈베리파이 코드: /content/drive/MyDrive/KFall_dataset/data/walking_data/walking_detector_raspberry.py")
    print("🎯 실제 KFall 데이터 기반 최적 파라미터가 적용되었습니다!")

    # 주요 파라미터 다시 출력
    print("\n📊 적용된 주요 파라미터:")
    print(f"   평균 가속도 범위: {optimal_params['acc_mean_min']:.3f} ~ {optimal_params['acc_mean_max']:.3f}")
    print(f"   보행 주파수 범위: {optimal_params['step_freq_min']:.1f} ~ {optimal_params['step_freq_max']:.1f} Hz")
    print(f"   신뢰도 임계값: {optimal_params['confidence_threshold']:.1f}")

    return features_df, optimal_params

# =============================================================================
# 실행
# =============================================================================

# 전체 분석 실행
results = run_complete_analysis()