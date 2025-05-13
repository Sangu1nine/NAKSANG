import os
import pandas as pd
import numpy as np
import glob
import re
import tqdm
from pathlib import Path

# 기본 디렉토리 경로
LABEL_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/label_data_new'
SENSOR_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/sensor_data_new'
OUTPUT_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/extracted_data'

# 파라미터 설정
FRAMES_BEFORE_ONSET = 150
FRAMES_AFTER_IMPACT = 150
TARGET_TASKS = list(range(20, 35))  # T20부터 T34까지

# 출력 디렉토리 생성
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 모든 레이블 파일과 센서 데이터 폴더를 처리하는 함수
def process_all_data():
    # 결과 저장용 리스트
    extraction_summary = []
    failed_files = []

    # 레이블 파일 처리 (SA06~SA38, SA34 제외)
    for subject_num in range(6, 39):
        if subject_num == 34:  # SA34 건너뛰기
            continue

        subject_id = f'SA{subject_num:02d}'
        label_file_path = os.path.join(LABEL_BASE_DIR, f"{subject_id}_label.xlsx")

        if not os.path.exists(label_file_path):
            print(f"Label file not found: {label_file_path}")
            continue

        # 센서 데이터 폴더 확인
        sensor_dir = os.path.join(SENSOR_BASE_DIR, subject_id)
        if not os.path.exists(sensor_dir):
            print(f"Sensor data folder not found: {sensor_dir}")
            continue

        # 출력 디렉토리 생성
        output_subject_dir = os.path.join(OUTPUT_BASE_DIR, subject_id)
        os.makedirs(output_subject_dir, exist_ok=True)

        # 레이블 파일 읽기
        try:
            label_df = pd.read_excel(label_file_path)
            print(f"\nProcessing: {subject_id} - Loaded label data ({len(label_df)} items)")
        except Exception as e:
            print(f"Error reading label file: {label_file_path}, Error: {e}")
            continue

        # 센서 데이터 파일 목록 가져오기
        sensor_files = glob.glob(os.path.join(sensor_dir, "*.csv"))

        # T20~T34 파일만 필터링
        target_files = []
        for file_path in sensor_files:
            file_name = os.path.basename(file_path)
            t_match = re.search(r'T(\d+)', file_name)
            if t_match and int(t_match.group(1)) in TARGET_TASKS:
                target_files.append(file_path)

        print(f"{subject_id} - Found {len(target_files)} sensor data files in T20-T34 range")

        # 각 센서 파일 처리
        for sensor_file in tqdm.tqdm(target_files, desc=f"Processing {subject_id}"):
            try:
                file_name = os.path.basename(sensor_file)

                # 파일명에서 정보 추출 (예: S06T20R01.csv)
                file_parts = Path(file_name).stem.split('T')
                if len(file_parts) != 2:
                    print(f"Cannot parse filename: {file_name}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'filename_parse_error'})
                    continue

                # T 뒤의 값에서 task_id 추출
                tr_parts = file_parts[1].split('R')
                if len(tr_parts) != 2:
                    print(f"Cannot parse task/trial from filename: {file_name}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'filename_parse_error'})
                    continue

                task_id = int(tr_parts[0])
                trial_id = int(tr_parts[1])

                # 해당 Trial에 대한 레이블 찾기
                relevant_labels = label_df[label_df['Trial ID'] == trial_id]

                if relevant_labels.empty:
                    print(f"No label found for: {subject_id}, Task {task_id}, Trial {trial_id}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'label_not_found'})
                    continue

                # 센서 데이터 읽기
                sensor_data = pd.read_csv(sensor_file)

                # Onset과 Impact 프레임 가져오기
                onset_frame = relevant_labels['Fall_onset_frame'].iloc[0]
                impact_frame = relevant_labels['Fall_impact_frame'].iloc[0]

                # 프레임 범위 계산
                start_frame = max(1, onset_frame - FRAMES_BEFORE_ONSET)
                end_frame = impact_frame + FRAMES_AFTER_IMPACT

                # 필요한 데이터 추출 (FrameCounter 열을 기준으로)
                extracted_data = sensor_data[(sensor_data['FrameCounter'] >= start_frame) &
                                             (sensor_data['FrameCounter'] <= end_frame)].copy()

                if len(extracted_data) == 0:
                    print(f"No data extracted: {file_name}, Frame range: {start_frame}-{end_frame}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name,
                                        'reason': 'no_frames_in_range',
                                        'start_frame': start_frame,
                                        'end_frame': end_frame})
                    continue

                # 메타데이터 추가
                extracted_data['original_frame'] = extracted_data['FrameCounter'].copy()
                extracted_data['rel_to_onset'] = extracted_data['FrameCounter'] - onset_frame
                extracted_data['rel_to_impact'] = extracted_data['FrameCounter'] - impact_frame

                # 파일 저장
                output_file = os.path.join(output_subject_dir, f"extracted_{file_name}")
                extracted_data.to_csv(output_file, index=False)

                # 요약 정보 저장
                extraction_summary.append({
                    'subject_id': subject_id,
                    'task_id': task_id,
                    'trial_id': trial_id,
                    'original_file': file_name,
                    'original_frames': len(sensor_data),
                    'extracted_frames': len(extracted_data),
                    'onset_frame': onset_frame,
                    'impact_frame': impact_frame,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_range': end_frame - start_frame + 1,
                    'output_file': os.path.basename(output_file)
                })

            except Exception as e:
                print(f"Error processing file: {sensor_file}, Error: {e}")
                failed_files.append({'subject_id': subject_id, 'file': os.path.basename(sensor_file),
                                    'reason': f'processing_error: {str(e)}'})
                continue

    # 결과 요약 저장
    summary_df = pd.DataFrame(extraction_summary)
    failed_df = pd.DataFrame(failed_files)

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'extraction_summary.csv'), index=False)

        # 추가 분석 수행
        task_stats = summary_df.groupby('task_id').agg({
            'original_frames': 'sum',
            'extracted_frames': 'sum',
            'subject_id': 'nunique',
            'original_file': 'count'
        }).reset_index()

        task_stats['extraction_ratio'] = task_stats['extracted_frames'] / task_stats['original_frames'] * 100
        task_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'task_statistics.csv'), index=False)

        subject_stats = summary_df.groupby('subject_id').agg({
            'original_frames': 'sum',
            'extracted_frames': 'sum',
            'task_id': 'nunique',
            'original_file': 'count'
        }).reset_index()

        subject_stats['extraction_ratio'] = subject_stats['extracted_frames'] / subject_stats['original_frames'] * 100
        subject_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'subject_statistics.csv'), index=False)

        # 통계 출력
        total_original_frames = summary_df['original_frames'].sum()
        total_extracted_frames = summary_df['extracted_frames'].sum()
        total_files = len(summary_df)

        print(f"\nExtraction Summary:")
        print(f"Total files processed successfully: {total_files}")
        print(f"Total original frames: {total_original_frames}")
        print(f"Total extracted frames: {total_extracted_frames}")
        print(f"Overall extraction ratio: {(total_extracted_frames / total_original_frames * 100):.2f}%")

    if not failed_df.empty:
        failed_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'failed_files.csv'), index=False)
        print(f"Failed files: {len(failed_df)}")

        # 실패 원인 분석
        reason_counts = failed_df['reason'].value_counts()
        print("Failure reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")

    return summary_df, failed_df

# 시각화 기능 추가
def create_visualizations(summary_df, output_dir=OUTPUT_BASE_DIR):
    """데이터 추출 결과에 대한 시각화 생성"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if summary_df is None or summary_df.empty:
        print("No data to visualize")
        return

    # 시각화 저장 폴더 생성
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 작업별(Task) 추출된 프레임 수
    plt.figure(figsize=(12, 6))
    task_frames = summary_df.groupby('task_id')['extracted_frames'].sum().sort_index()
    ax = sns.barplot(x=task_frames.index, y=task_frames.values)
    plt.title('Total Extracted Frames by Task')
    plt.xlabel('Task ID')
    plt.ylabel('Number of Frames')
    plt.xticks(rotation=0)

    # 바 위에 값 표시
    for i, v in enumerate(task_frames.values):
        ax.text(i, v + 0.5, str(int(v)), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'frames_by_task.png'))
    plt.close()

    # 2. 피험자별 추출된 프레임 수
    plt.figure(figsize=(16, 6))
    subject_frames = summary_df.groupby('subject_id')['extracted_frames'].sum().sort_index()
    ax = sns.barplot(x=subject_frames.index, y=subject_frames.values)
    plt.title('Total Extracted Frames by Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Number of Frames')
    plt.xticks(rotation=90)

    # 바 위에 값 표시
    for i, v in enumerate(subject_frames.values):
        ax.text(i, v + 0.5, str(int(v)), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'frames_by_subject.png'))
    plt.close()

    # 3. 추출된 프레임 길이 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data=summary_df, x='frame_range', bins=30, kde=True)
    plt.title('Distribution of Extracted Frame Ranges')
    plt.xlabel('Frame Range Length')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'frame_range_distribution.png'))
    plt.close()

    # 4. 히트맵: 피험자별 작업별 파일 수
    plt.figure(figsize=(16, 12))
    pivot_data = summary_df.pivot_table(
        index='subject_id',
        columns='task_id',
        values='original_file',
        aggfunc='count'
    ).fillna(0)

    sns.heatmap(pivot_data, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Number of Files by Subject and Task')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'subject_task_file_counts.png'))
    plt.close()

    print(f"Visualizations saved to {viz_dir}")

# 메인 실행 함수
def main():
    print("Starting data extraction process...")
    summary_df, failed_df = process_all_data()

    if summary_df is not None and not summary_df.empty:
        create_visualizations(summary_df)
        print("Data extraction and visualization complete!")
    else:
        print("No data was extracted. Check for errors.")

if __name__ == "__main__":
    main()