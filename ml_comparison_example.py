# =============================================================================
# 보행 감지: 머신러닝 vs 규칙 기반 접근법 비교
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

# =============================================================================
# 1. 머신러닝 접근법
# =============================================================================

class MLWalkingDetector:
    """머신러닝 기반 보행 감지기"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_scaler = None
        
    def train(self, features_df):
        """모델 학습"""
        # 특징 선택
        feature_columns = [
            'acc_mean', 'acc_std', 'acc_range', 'gyr_mean', 'gyr_std',
            'step_frequency', 'step_regularity', 'peak_count', 'peak_density',
            'walking_energy_ratio'
        ]
        
        X = features_df[feature_columns]
        y = features_df['label']
        
        # 피험자별 분할
        subjects = features_df['subject_id'].unique()
        train_subjects, test_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
        
        train_mask = features_df['subject_id'].isin(train_subjects)
        test_mask = features_df['subject_id'].isin(test_subjects)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # 모델 선택 및 학습
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
        
        # 학습
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        print(f"🤖 {self.model_type} 모델 성능:")
        print(f"   F1 Score: {f1:.3f}")
        print(classification_report(y_test, y_pred))
        
        return f1
    
    def predict(self, features):
        """예측 (실시간 사용)"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특징 벡터를 2D 배열로 변환
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # 예측
        probability = self.model.predict_proba(features)[0][1]  # 보행 확률
        prediction = self.model.predict(features)[0]
        
        return prediction, probability
    
    def save_model(self, filepath):
        """모델 저장"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """모델 로드"""
        self.model = joblib.load(filepath)

# =============================================================================
# 2. 규칙 기반 접근법 (기존 방식)
# =============================================================================

class RuleBasedWalkingDetector:
    """규칙 기반 보행 감지기"""
    
    def __init__(self, config):
        self.config = config
        
    def predict(self, features):
        """예측 (실시간 사용)"""
        confidence = 0.0
        
        # 가속도 평균 체크
        if (self.config['thresholds']['acc_mean_min'] <= features['acc_mean'] <= 
            self.config['thresholds']['acc_mean_max']):
            confidence += self.config['weights']['acc_mean_weight']
        
        # 가속도 표준편차 체크
        if features['acc_std'] >= self.config['thresholds']['acc_std_min']:
            confidence += self.config['weights']['acc_std_weight']
        
        # 보행 주파수 체크
        if (self.config['thresholds']['step_freq_min'] <= features['step_frequency'] <= 
            self.config['thresholds']['step_freq_max']):
            confidence += self.config['weights']['step_freq_weight']
        
        # 규칙성 체크
        if features['step_regularity'] >= self.config['thresholds']['regularity_min']:
            confidence += self.config['weights']['regularity_weight']
        
        prediction = 1 if confidence >= self.config['thresholds']['confidence_min'] else 0
        
        return prediction, confidence

# =============================================================================
# 3. 성능 비교
# =============================================================================

def compare_approaches(features_df):
    """머신러닝 vs 규칙 기반 성능 비교"""
    
    print("🔬 머신러닝 vs 규칙 기반 접근법 비교")
    print("="*50)
    
    # 1. 머신러닝 모델들 테스트
    ml_results = {}
    
    for model_type in ['random_forest', 'svm', 'neural_network']:
        print(f"\n🤖 {model_type} 학습 중...")
        ml_detector = MLWalkingDetector(model_type)
        f1_score = ml_detector.train(features_df)
        ml_results[model_type] = f1_score
    
    # 2. 규칙 기반 시스템 테스트
    print(f"\n📏 규칙 기반 시스템 테스트...")
    
    # 예시 설정 (실제로는 optimal_threshold_analysis.py 결과 사용)
    rule_config = {
        'thresholds': {
            'acc_mean_min': 0.9,
            'acc_mean_max': 1.2,
            'acc_std_min': 0.2,
            'step_freq_min': 1.0,
            'step_freq_max': 4.0,
            'regularity_min': 0.4,
            'confidence_min': 0.6
        },
        'weights': {
            'acc_mean_weight': 0.25,
            'acc_std_weight': 0.25,
            'step_freq_weight': 0.35,
            'regularity_weight': 0.15
        }
    }
    
    rule_detector = RuleBasedWalkingDetector(rule_config)
    
    # 규칙 기반 성능 계산
    predictions = []
    for _, row in features_df.iterrows():
        pred, conf = rule_detector.predict(row)
        predictions.append(pred)
    
    rule_f1 = f1_score(features_df['label'], predictions)
    
    # 결과 비교
    print(f"\n📊 성능 비교 결과:")
    print(f"   규칙 기반:        F1 = {rule_f1:.3f}")
    for model, f1 in ml_results.items():
        print(f"   {model:15}: F1 = {f1:.3f}")
    
    return ml_results, rule_f1

# =============================================================================
# 4. 실시간 성능 비교
# =============================================================================

def compare_realtime_performance():
    """실시간 성능 비교 (속도, 메모리 등)"""
    
    print("\n⚡ 실시간 성능 비교:")
    print("="*30)
    
    comparison = {
        '구분': ['규칙 기반', '랜덤포레스트', 'SVM', '신경망'],
        '예측 속도': ['~0.1ms', '~1-5ms', '~0.5-2ms', '~2-10ms'],
        '메모리 사용': ['~1KB', '~100KB-1MB', '~10KB-100KB', '~1MB-10MB'],
        '배터리 영향': ['최소', '보통', '보통', '높음'],
        '해석 가능성': ['높음', '보통', '낮음', '낮음'],
        '안정성': ['높음', '높음', '보통', '보통']
    }
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    print(f"\n💡 결론:")
    print(f"   • 라즈베리파이 실시간 감지: 규칙 기반이 최적")
    print(f"   • 높은 정확도가 필요한 연구: 머신러닝 고려")
    print(f"   • 배터리 수명 중요: 규칙 기반 선택")

# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # 가상의 특징 데이터로 테스트
    # 실제로는 optimal_threshold_analysis.py의 features_df 사용
    
    print("🎯 보행 감지 접근법 비교 분석")
    print("="*40)
    
    # 실제 사용시에는 다음과 같이:
    # features_df = extract_windowed_features(data)
    # ml_results, rule_f1 = compare_approaches(features_df)
    
    compare_realtime_performance()
    
    print(f"\n✅ 결론: 현재 구현은 '통계적 최적화된 규칙 기반 시스템'입니다!")
    print(f"   머신러닝이 아닌, 데이터 기반으로 최적화된 if-else 규칙들입니다.") 