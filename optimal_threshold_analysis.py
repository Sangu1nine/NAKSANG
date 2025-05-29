# =============================================================================
# 보행 감지 최적 임계값 추출 시스템
# 🎯 이진 분류 접근법 + ROC 분석 + 교차 검증
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# 1단계: 개선된 직접 데이터 로딩
# =============================================================================

def load_binary_classification_data(base_path='/content/drive/MyDrive/KFall_dataset/data/sensor_data_new'):
    """sensor_data에서 바로 보행/비보행 데이터 로딩 (개선된 방식)"""
    
    # 활동 분류 정의 (낙상 관련 활동 제외)
    activity_mapping = {
        # 보행 활동 (Walking = 1)
        'walking': ['06', '07', '08', '09', '10', '35', '36'],
        
        # 비보행 활동 (Non-walking = 0) - 일상 활동만 포함
        'non_walking': [
            '01', '02', '03', '04', '05',  # 일상 활동 (앉기, 서기, 누워있기 등)
            '11', '12', '13', '14', '15',  # 기타 활동
            '16', '17', '18', '19'         # 기타 비보행 활동
            # 20~34 낙상 관련 활동은 제외 (별도 딥러닝 모델에서 처리)
        ]
    }
    
    all_data = []
    file_stats = {
        'walking_files': 0,
        'non_walking_files': 0,
        'excluded_fall_files': 0,
        'error_files': 0,
        'total_subjects': set()
    }
    
    print("📂 sensor_data에서 직접 로딩 시작...")
    
    # 모든 피험자 폴더 탐색 (SA06~SA38, SA34 제외)
    for subject_num in range(6, 39):
        if subject_num == 34:  # SA34 제외
            continue
            
        subject_id = f'SA{subject_num:02d}'
        subject_folder = os.path.join(base_path, subject_id)
        
        if not os.path.exists(subject_folder):
            continue
            
        file_stats['total_subjects'].add(subject_id)
        
        # CSV 파일들 탐색
        csv_files = glob.glob(os.path.join(subject_folder, '*.csv'))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            
            try:
                # 파일명에서 Task 번호 추출 (예: S06T06R01.csv → 06)
                t_match = re.search(r'T(\d+)', filename)
                if not t_match:
                    continue
                    
                activity_num = t_match.group(1)
                
                # 활동 분류
                if activity_num in activity_mapping['walking']:
                    label = 1  # 보행
                    activity_type = 'walking'
                    file_stats['walking_files'] += 1
                elif activity_num in activity_mapping['non_walking']:
                    label = 0  # 비보행
                    activity_type = 'non_walking'
                    file_stats['non_walking_files'] += 1
                elif int(activity_num) >= 20 and int(activity_num) <= 34:
                    # 낙상 관련 활동 제외
                    file_stats['excluded_fall_files'] += 1
                    continue
                else:
                    continue  # 정의되지 않은 활동
                
                # CSV 파일 로딩 (메모리 효율적)
                try:
                    df = pd.read_csv(csv_file)
                    
                    # 필요한 컬럼만 선택 (메모리 절약)
                    required_columns = ['TimeStamp(s)', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"⚠️ {filename}: 누락된 컬럼 {missing_columns}")
                        file_stats['error_files'] += 1
                        continue
                    
                    df = df[required_columns].copy()
                    
                    # 메타데이터 추가
                    df['subject_id'] = subject_id
                    df['activity_num'] = activity_num
                    df['label'] = label
                    df['activity_type'] = activity_type
                    df['filename'] = filename
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"❌ {filename} 로딩 실패: {e}")
                    file_stats['error_files'] += 1
                    continue
                    
            except Exception as e:
                print(f"❌ {filename} 처리 실패: {e}")
                file_stats['error_files'] += 1
                continue
    
    # 결과 통합
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("\n✅ 로딩 완료!")
        print(f"📊 총 피험자: {len(file_stats['total_subjects'])}명")
        print(f"🚶 보행 파일: {file_stats['walking_files']}개")
        print(f"🛑 비보행 파일: {file_stats['non_walking_files']}개")
        print(f"🚫 제외된 낙상 파일: {file_stats['excluded_fall_files']}개")
        print(f"❌ 오류 파일: {file_stats['error_files']}개")
        print(f"📈 총 데이터 포인트: {len(combined_df):,}개")
        print(f"⚖️ 보행 비율: {combined_df['label'].mean():.2%}")
        
        # 메모리 사용량 출력
        memory_usage = combined_df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 메모리 사용량: {memory_usage:.1f} MB")
        
        return combined_df
    else:
        print("❌ 로딩된 데이터가 없습니다.")
        return None

# =============================================================================
# 2단계: 윈도우 기반 특징 추출
# =============================================================================

def extract_windowed_features(data, window_size=200, overlap=0.5):
    """윈도우 기반 특징 추출 (실제 감지기와 동일한 조건)"""
    
    def calculate_features(window_data):
        """단일 윈도우에서 특징 계산"""
        # 가속도 벡터 크기
        acc_mag = np.sqrt(window_data['AccX']**2 + window_data['AccY']**2 + window_data['AccZ']**2)
        gyr_mag = np.sqrt(window_data['GyrX']**2 + window_data['GyrY']**2 + window_data['GyrZ']**2)
        
        # 이동평균 필터링
        acc_smooth = np.convolve(acc_mag, np.ones(5)/5, mode='same')
        
        # 피크 검출
        threshold = np.mean(acc_smooth) + 0.3 * np.std(acc_smooth)
        peaks = []
        for i in range(5, len(acc_smooth)-5):
            if acc_smooth[i] > threshold and acc_smooth[i] == max(acc_smooth[i-5:i+6]):
                peaks.append(i)
        
        # 기본 통계 특징
        features = {
            'acc_mean': np.mean(acc_mag),
            'acc_std': np.std(acc_mag),
            'acc_max': np.max(acc_mag),
            'acc_min': np.min(acc_mag),
            'acc_range': np.max(acc_mag) - np.min(acc_mag),
            'gyr_mean': np.mean(gyr_mag),
            'gyr_std': np.std(gyr_mag),
            'peak_count': len(peaks),
            'peak_density': len(peaks) / len(acc_mag) * 100  # 100샘플당 피크 수
        }
        
        # 보행 주기 특징
        if len(peaks) >= 2:
            time_stamps = window_data['TimeStamp(s)'].values
            peak_times = time_stamps[peaks]
            intervals = np.diff(peak_times)
            
            if len(intervals) > 0 and np.all(intervals > 0):  # 양수 간격만 허용
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # 안전한 계산 (0으로 나누기 방지)
                if mean_interval > 0:
                    features['step_frequency'] = 1.0 / mean_interval
                else:
                    features['step_frequency'] = 0
                    
                features['step_regularity'] = 1.0 / (1.0 + std_interval) if std_interval >= 0 else 0
                features['step_interval_std'] = std_interval
            else:
                features['step_frequency'] = 0
                features['step_regularity'] = 0
                features['step_interval_std'] = 0
        else:
            features['step_frequency'] = 0
            features['step_regularity'] = 0
            features['step_interval_std'] = 0
        
        # 주파수 도메인 특징 (안전한 계산)
        try:
            if len(acc_mag) > 1:
                fft_acc = np.abs(fft(acc_mag))
                freqs = fftfreq(len(acc_mag), d=0.01)  # 100Hz 샘플링
                
                # 0.5-5Hz 대역의 에너지 (보행 주파수 대역)
                walking_band = (freqs >= 0.5) & (freqs <= 5.0)
                walking_energy = np.sum(fft_acc[walking_band])
                total_energy = np.sum(fft_acc)
                
                # 안전한 비율 계산
                if total_energy > 0:
                    features['walking_band_energy'] = walking_energy
                    features['total_energy'] = total_energy
                    features['walking_energy_ratio'] = walking_energy / total_energy
                else:
                    features['walking_band_energy'] = 0
                    features['total_energy'] = 0
                    features['walking_energy_ratio'] = 0
            else:
                features['walking_band_energy'] = 0
                features['total_energy'] = 0
                features['walking_energy_ratio'] = 0
        except:
            # FFT 계산 실패시 기본값
            features['walking_band_energy'] = 0
            features['total_energy'] = 0
            features['walking_energy_ratio'] = 0
        
        # 무한대 값과 NaN 값 처리
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
            elif value > 1e10:  # 너무 큰 값 제한
                features[key] = 1e10
            elif value < -1e10:  # 너무 작은 값 제한
                features[key] = -1e10
        
        return features
    
    # 윈도우별 특징 추출
    all_features = []
    step_size = int(window_size * (1 - overlap))
    
    grouped = data.groupby(['subject_id', 'activity_num', 'label', 'activity_type'])
    
    for (subject, activity, label, act_type), group in grouped:
        # 시간순 정렬
        group = group.sort_values('TimeStamp(s)')
        
        # 윈도우 슬라이딩
        for start_idx in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start_idx:start_idx + window_size]
            
            if len(window) == window_size:  # 완전한 윈도우만 사용
                features = calculate_features(window)
                features.update({
                    'subject_id': subject,
                    'activity_num': activity,
                    'label': label,
                    'activity_type': act_type
                })
                all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    print(f"✅ {len(features_df)}개 윈도우 특징 추출 완료")
    print(f"🚶 보행 윈도우: {len(features_df[features_df['label']==1]):,}개")
    print(f"🛑 비보행 윈도우: {len(features_df[features_df['label']==0]):,}개")
    
    return features_df

# =============================================================================
# 3단계: ROC 기반 최적 임계값 계산
# =============================================================================

def find_optimal_thresholds_roc(features_df):
    """ROC 분석을 통한 최적 임계값 계산"""
    
    # 특징 선택
    feature_columns = [
        'acc_mean', 'acc_std', 'acc_range', 'gyr_mean', 'gyr_std',
        'step_frequency', 'step_regularity', 'peak_count', 'peak_density',
        'walking_energy_ratio'
    ]
    
    # 데이터 정리 및 검증
    print("🔍 데이터 검증 및 정리 중...")
    
    # 무한대 값과 NaN 값 확인
    for col in feature_columns:
        if col in features_df.columns:
            # 무한대 값 처리
            inf_mask = np.isinf(features_df[col])
            if inf_mask.any():
                print(f"⚠️ {col}: {inf_mask.sum()}개 무한대 값 발견 → 0으로 대체")
                features_df.loc[inf_mask, col] = 0.0
            
            # NaN 값 처리
            nan_mask = np.isnan(features_df[col])
            if nan_mask.any():
                print(f"⚠️ {col}: {nan_mask.sum()}개 NaN 값 발견 → 0으로 대체")
                features_df.loc[nan_mask, col] = 0.0
            
            # 너무 큰 값 제한
            large_mask = np.abs(features_df[col]) > 1e10
            if large_mask.any():
                print(f"⚠️ {col}: {large_mask.sum()}개 큰 값 발견 → 제한")
                features_df.loc[large_mask, col] = np.sign(features_df.loc[large_mask, col]) * 1e10
    
    X = features_df[feature_columns]
    y = features_df['label']
    
    # 최종 데이터 검증
    print(f"📊 최종 데이터 검증:")
    print(f"   무한대 값: {np.isinf(X).sum().sum()}개")
    print(f"   NaN 값: {np.isnan(X).sum().sum()}개")
    print(f"   유효한 샘플: {len(X)}개")
    
    # 훈련/테스트 분할 (피험자별 분할)
    subjects = features_df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
    
    train_mask = features_df['subject_id'].isin(train_subjects)
    test_mask = features_df['subject_id'].isin(test_subjects)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"📊 훈련 데이터: {len(X_train)}개 ({y_train.mean():.2%} 보행)")
    print(f"📊 테스트 데이터: {len(X_test)}개 ({y_test.mean():.2%} 보행)")
    
    # 각 특징별 최적 임계값 계산
    optimal_thresholds = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(feature_columns):
        plt.subplot(2, 5, i+1)
        
        try:
            # 특징 값 범위 확인
            feature_values = X_train[feature]
            print(f"🔍 {feature}: 범위 [{feature_values.min():.3f}, {feature_values.max():.3f}]")
            
            # ROC 곡선 계산
            fpr, tpr, thresholds = roc_curve(y_train, feature_values)
            roc_auc = auc(fpr, tpr)
            
            # Youden's J statistic으로 최적점 찾기
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            optimal_thresholds[feature] = {
                'threshold': optimal_threshold,
                'auc': roc_auc,
                'sensitivity': tpr[optimal_idx],
                'specificity': 1 - fpr[optimal_idx]
            }
            
            # ROC 곡선 그리기
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{feature}\nThreshold: {optimal_threshold:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"❌ {feature} ROC 계산 실패: {e}")
            # 기본값 설정
            optimal_thresholds[feature] = {
                'threshold': 0.5,
                'auc': 0.5,
                'sensitivity': 0.5,
                'specificity': 0.5
            }
            plt.text(0.5, 0.5, f'Error: {feature}', ha='center', va='center')
            plt.title(f'{feature}\nError')
    
    plt.tight_layout()
    plt.show()
    
    return optimal_thresholds, (X_train, X_test, y_train, y_test)

# =============================================================================
# 4단계: 앙상블 기반 임계값 최적화
# =============================================================================

def optimize_ensemble_thresholds(features_df, optimal_thresholds):
    """앙상블 방식으로 최종 임계값 최적화"""
    
    # 핵심 특징들 선택 (AUC 기준)
    feature_importance = {k: v['auc'] for k, v in optimal_thresholds.items()}
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:6]
    
    print("🏆 상위 특징들 (AUC 기준):")
    for feature, auc_score in top_features:
        print(f"   {feature}: {auc_score:.3f}")
    
    # 보행 감지기와 유사한 규칙 기반 시스템 구성
    walking_rules = {
        'acc_mean_range': (
            optimal_thresholds['acc_mean']['threshold'] * 0.9,  # 하한
            optimal_thresholds['acc_mean']['threshold'] * 1.1   # 상한
        ),
        'acc_std_min': optimal_thresholds['acc_std']['threshold'],
        'step_freq_range': (1.0, 4.0),  # 생리학적 범위
        'step_regularity_min': optimal_thresholds['step_regularity']['threshold'],
        'walking_energy_ratio_min': optimal_thresholds['walking_energy_ratio']['threshold']
    }
    
    # 가중치 최적화 (그리드 서치)
    best_weights = None
    best_f1 = 0
    
    weight_combinations = [
        {'acc': 0.25, 'std': 0.25, 'freq': 0.35, 'reg': 0.15},
        {'acc': 0.20, 'std': 0.30, 'freq': 0.40, 'reg': 0.10},
        {'acc': 0.30, 'std': 0.20, 'freq': 0.30, 'reg': 0.20},
        {'acc': 0.15, 'std': 0.25, 'freq': 0.45, 'reg': 0.15},
    ]
    
    for weights in weight_combinations:
        # 규칙 기반 분류 수행
        predictions = []
        for _, row in features_df.iterrows():
            confidence = 0.0
            
            # 가속도 평균 체크
            if walking_rules['acc_mean_range'][0] <= row['acc_mean'] <= walking_rules['acc_mean_range'][1]:
                confidence += weights['acc']
            
            # 가속도 표준편차 체크
            if row['acc_std'] >= walking_rules['acc_std_min']:
                confidence += weights['std']
            
            # 보행 주파수 체크
            if walking_rules['step_freq_range'][0] <= row['step_frequency'] <= walking_rules['step_freq_range'][1]:
                confidence += weights['freq']
            
            # 규칙성 체크
            if row['step_regularity'] >= walking_rules['step_regularity_min']:
                confidence += weights['reg']
            
            predictions.append(1 if confidence >= 0.6 else 0)
        
        # F1 스코어 계산
        from sklearn.metrics import f1_score
        f1 = f1_score(features_df['label'], predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
    
    print(f"\n🎯 최적 가중치 (F1: {best_f1:.3f}):")
    for key, value in best_weights.items():
        print(f"   {key}: {value}")
    
    return walking_rules, best_weights

# =============================================================================
# 5단계: 최종 임계값 생성
# =============================================================================

def generate_final_thresholds(optimal_thresholds, walking_rules, best_weights):
    """최종 임계값 딕셔너리 생성"""
    
    final_config = {
        'thresholds': {
            # 가속도 관련
            'acc_mean_min': walking_rules['acc_mean_range'][0],
            'acc_mean_max': walking_rules['acc_mean_range'][1],
            'acc_std_min': walking_rules['acc_std_min'],
            
            # 보행 주기 관련
            'step_freq_min': walking_rules['step_freq_range'][0],
            'step_freq_max': walking_rules['step_freq_range'][1],
            'regularity_min': walking_rules['step_regularity_min'],
            
            # 피크 검출 관련
            'peak_detection_factor': 0.3,
            'peak_window_size': 5,
            
            # 최종 판단
            'confidence_min': 0.6
        },
        'weights': {
            'acc_mean_weight': best_weights['acc'],
            'acc_std_weight': best_weights['std'],
            'step_freq_weight': best_weights['freq'],
            'regularity_weight': best_weights['reg']
        },
        'filtering': {
            'moving_avg_window': 5,
            'min_peaks_required': 2
        }
    }
    
    print("\n🎯 최종 최적화된 설정:")
    print("📊 임계값:")
    for key, value in final_config['thresholds'].items():
        print(f"   {key}: {value:.3f}")
    print("⚖️ 가중치:")
    for key, value in final_config['weights'].items():
        print(f"   {key}: {value:.3f}")
    
    return final_config

# =============================================================================
# 전체 실행 함수
# =============================================================================

def run_optimal_threshold_analysis():
    """최적 임계값 분석 전체 실행"""
    
    print("🚀 보행 감지 최적 임계값 분석 시작")
    print("="*60)
    
    # 1단계: 이진 분류 데이터 로딩
    print("\n1️⃣ 이진 분류 데이터 로딩...")
    data = load_binary_classification_data()
    if data is None:
        return
    
    # 2단계: 윈도우 기반 특징 추출
    print("\n2️⃣ 윈도우 기반 특징 추출...")
    features_df = extract_windowed_features(data, window_size=200, overlap=0.5)
    
    # 3단계: ROC 기반 최적 임계값 계산
    print("\n3️⃣ ROC 분석...")
    optimal_thresholds, split_data = find_optimal_thresholds_roc(features_df)
    
    # 4단계: 앙상블 최적화
    print("\n4️⃣ 앙상블 최적화...")
    walking_rules, best_weights = optimize_ensemble_thresholds(features_df, optimal_thresholds)
    
    # 5단계: 최종 설정 생성
    print("\n5️⃣ 최종 설정 생성...")
    final_config = generate_final_thresholds(optimal_thresholds, walking_rules, best_weights)
    
    print("\n✅ 최적 임계값 분석 완료!")
    print("🎯 이 설정을 walking_detector_raspberry.py에 적용하세요.")
    
    return final_config, features_df, optimal_thresholds

# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    results = run_optimal_threshold_analysis() 