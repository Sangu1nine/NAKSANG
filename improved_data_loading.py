# =============================================================================
# 개선된 데이터 로딩: sensor_data에서 바로 처리
# 🚀 별도 추출 없이 메모리 효율적으로 처리
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_data_directly_from_sensor(base_path='/content/drive/MyDrive/KFall_dataset/data/sensor_data_new'):
    """sensor_data에서 바로 보행/비보행 데이터 로딩 (개선된 방식)"""
    
    # 활동 분류 정의
    activity_mapping = {
        # 보행 활동 (Walking = 1)
        'walking': ['06', '07', '08', '09', '10', '35', '36'],
        
        # 비보행 활동 (Non-walking = 0) - 일상 활동만
        'non_walking': [
            '01', '02', '03', '04', '05',  # 일상 활동
            '11', '12', '13', '14', '15',  # 기타 활동
            '16', '17', '18', '19'         # 기타 비보행 활동
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
        
        return combined_df, file_stats
    else:
        print("❌ 로딩된 데이터가 없습니다.")
        return None, file_stats

# =============================================================================
# 기존 방식과 개선된 방식 비교
# =============================================================================

def compare_loading_methods():
    """기존 추출 방식 vs 직접 로딩 방식 비교"""
    
    print("🔄 데이터 로딩 방식 비교")
    print("="*50)
    
    comparison = {
        '구분': ['기존 추출 방식', '직접 로딩 방식'],
        '디스크 사용량': ['2배 (원본 + 복사본)', '1배 (원본만)'],
        '처리 시간': ['느림 (복사 + 로딩)', '빠름 (로딩만)'],
        '메모리 효율성': ['낮음 (전체 로딩)', '높음 (필요한 컬럼만)'],
        '유연성': ['낮음 (고정된 Task)', '높음 (동적 필터링)'],
        '유지보수': ['복잡 (2단계)', '간단 (1단계)']
    }
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    print(f"\n💡 권장사항:")
    print(f"   • 일회성 분석: 직접 로딩 방식 사용")
    print(f"   • 반복 분석: 첫 실행시 pickle로 저장 후 재사용")
    print(f"   • 메모리 부족시: 청크 단위로 처리")

# =============================================================================
# 메모리 효율적인 청크 처리
# =============================================================================

def load_data_in_chunks(base_path, chunk_size=10):
    """메모리 효율적인 청크 단위 처리"""
    
    print(f"🔄 청크 단위 처리 (청크 크기: {chunk_size}개 파일)")
    
    # 모든 파일 목록 먼저 수집
    all_files = []
    for subject_num in range(6, 39):
        if subject_num == 34:
            continue
        subject_id = f'SA{subject_num:02d}'
        subject_folder = os.path.join(base_path, subject_id)
        if os.path.exists(subject_folder):
            csv_files = glob.glob(os.path.join(subject_folder, '*.csv'))
            all_files.extend(csv_files)
    
    print(f"📁 총 {len(all_files)}개 파일 발견")
    
    # 청크 단위로 처리
    for i in range(0, len(all_files), chunk_size):
        chunk_files = all_files[i:i+chunk_size]
        print(f"🔄 청크 {i//chunk_size + 1} 처리 중... ({len(chunk_files)}개 파일)")
        
        # 청크 처리 로직
        chunk_data = []
        for file_path in chunk_files:
            # 파일 처리 (위의 로직과 동일)
            pass
        
        # 청크별 결과 처리
        yield chunk_data

# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    print("🚀 개선된 데이터 로딩 시스템")
    print("="*40)
    
    # 방식 비교
    compare_loading_methods()
    
    print(f"\n" + "="*40)
    
    # 실제 로딩 (예시)
    # data, stats = load_data_directly_from_sensor()
    
    print(f"\n✅ 결론:")
    print(f"   기존 추출 방식 대신 직접 로딩 방식을 사용하세요!")
    print(f"   - 더 빠르고 효율적")
    print(f"   - 디스크 공간 절약") 
    print(f"   - 유연한 필터링 가능") 