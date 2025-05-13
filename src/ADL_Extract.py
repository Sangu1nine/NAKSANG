import os
import pandas as pd
import numpy as np
import glob
import re
import tqdm
from pathlib import Path
import shutil

# 기본 디렉토리 경로
SENSOR_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/sensor_data_new'
OUTPUT_BASE_DIR = '/content/drive/MyDrive/KFall_dataset/data/selected_tasks_data'

# 파라미터 설정
TARGET_TASKS = [3, 4, 9, 10, 14, 15, 19, 36]  # 선택된 Task ID 목록

# 이전 출력 데이터 삭제 함수
def clean_previous_output(output_dir=OUTPUT_BASE_DIR, ask_confirmation=True):
    """이전에 생성된 출력 데이터 삭제"""
    if os.path.exists(output_dir):
        if ask_confirmation:
            confirm = input(f"기존 출력 폴더 '{output_dir}'가 존재합니다. 삭제하시겠습니까? (y/n): ")
            if confirm.lower() != 'y':
                print("기존 데이터 유지. 프로그램을 종료합니다.")
                return False

        print(f"기존 출력 폴더 '{output_dir}' 삭제 중...")
        try:
            shutil.rmtree(output_dir)
            print("기존 출력 폴더 삭제 완료")
        except Exception as e:
            print(f"폴더 삭제 중 오류 발생: {e}")
            return False

    # 출력 디렉토리 새로 생성
    os.makedirs(output_dir, exist_ok=True)
    return True

# 모든 센서 파일을 처리하는 함수
def extract_selected_tasks():
    # 결과 저장용 리스트
    extraction_summary = []
    failed_files = []

    # 피험자 폴더 처리 (SA06~SA38, SA34 제외)
    for subject_num in range(6, 39):
        if subject_num == 34:  # SA34 건너뛰기
            continue

        subject_id = f'SA{subject_num:02d}'
        sensor_dir = os.path.join(SENSOR_BASE_DIR, subject_id)

        # 센서 데이터 폴더 확인
        if not os.path.exists(sensor_dir):
            print(f"Sensor data folder not found: {sensor_dir}")
            continue

        # 출력 디렉토리 생성
        output_subject_dir = os.path.join(OUTPUT_BASE_DIR, subject_id)
        os.makedirs(output_subject_dir, exist_ok=True)

        # 센서 데이터 파일 목록 가져오기
        sensor_files = glob.glob(os.path.join(sensor_dir, "*.csv"))

        # 지정된 Task 파일만 필터링
        target_files = []
        for file_path in sensor_files:
            file_name = os.path.basename(file_path)
            t_match = re.search(r'T(\d+)', file_name)
            if t_match and int(t_match.group(1)) in TARGET_TASKS:
                target_files.append(file_path)

        if not target_files:
            print(f"{subject_id} - No files found for selected tasks")
            continue

        print(f"{subject_id} - Found {len(target_files)} files for selected tasks")

        # 각 센서 파일 처리
        for sensor_file in tqdm.tqdm(target_files, desc=f"Processing {subject_id}"):
            try:
                file_name = os.path.basename(sensor_file)

                # 파일명에서 정보 추출 (예: S06T03R01.csv)
                t_match = re.search(r'T(\d+)', file_name)
                r_match = re.search(r'R(\d+)', file_name)

                if not t_match or not r_match:
                    print(f"Cannot parse filename: {file_name}")
                    failed_files.append({'subject_id': subject_id, 'file': file_name, 'reason': 'filename_parse_error'})
                    continue

                task_id = int(t_match.group(1))
                trial_id = int(r_match.group(1))

                # 파일 복사
                output_file = os.path.join(output_subject_dir, file_name)
                shutil.copy2(sensor_file, output_file)

                # CSV 파일 정보 읽기 (행 수 확인용)
                try:
                    df = pd.read_csv(sensor_file)
                    num_rows = len(df)
                except:
                    num_rows = -1  # 파일 읽기 실패

                # 요약 정보 저장
                extraction_summary.append({
                    'subject_id': subject_id,
                    'task_id': task_id,
                    'trial_id': trial_id,
                    'original_file': file_name,
                    'num_rows': num_rows,
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

        # 통계 출력
        task_stats = summary_df.groupby('task_id').agg({
            'subject_id': 'nunique',
            'original_file': 'count'
        }).reset_index()
        task_stats.columns = ['task_id', 'num_subjects', 'num_files']
        task_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'task_statistics.csv'), index=False)

        subject_stats = summary_df.groupby('subject_id').agg({
            'task_id': 'nunique',
            'original_file': 'count'
        }).reset_index()
        subject_stats.columns = ['subject_id', 'num_task_types', 'num_files']
        subject_stats.to_csv(os.path.join(OUTPUT_BASE_DIR, 'subject_statistics.csv'), index=False)

        # 출력
        total_files = len(summary_df)
        print(f"\nExtraction Summary:")
        print(f"Total files extracted: {total_files}")

        # 작업별 파일 수
        print("\nFiles per task:")
        task_file_counts = summary_df['task_id'].value_counts().sort_index()
        for task, count in task_file_counts.items():
            print(f"  T{task}: {count} files")

    if not failed_df.empty:
        failed_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'failed_files.csv'), index=False)
        print(f"\nFailed files: {len(failed_df)}")

    return summary_df, failed_df

# 간단한 시각화 기능
def create_minimal_visualization(summary_df, output_dir=OUTPUT_BASE_DIR):
    """최소한의 시각화만 생성"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if summary_df is None or summary_df.empty:
        print("No data to visualize")
        return

    # 시각화 저장 폴더 생성
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # 작업별 파일 수 막대 그래프
    plt.figure(figsize=(10, 6))
    task_counts = summary_df['task_id'].value_counts().sort_index()
    sns.barplot(x=task_counts.index, y=task_counts.values)
    plt.title('Number of Files by Task')
    plt.xlabel('Task ID')
    plt.ylabel('Number of Files')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'files_by_task.png'))
    plt.close()

    print(f"Visualization saved to {viz_dir}")

# 메인 실행 함수
def main():
    print(f"Task Selection Settings:")
    print(f"- Selected tasks: {', '.join([f'T{t}' for t in TARGET_TASKS])}")
    print(f"- Output directory: {OUTPUT_BASE_DIR}")
    print("-" * 50)

    # 이전 출력 삭제
    if not clean_previous_output(OUTPUT_BASE_DIR):
        return

    print("Starting task extraction process...")
    summary_df, failed_df = extract_selected_tasks()

    if summary_df is not None and not summary_df.empty:
        create_minimal_visualization(summary_df)

        # 처리 완료된 파일 수 출력
        print("\nExtraction completed successfully!")
        print(f"Extracted {len(summary_df)} files for tasks: {', '.join([f'T{t}' for t in TARGET_TASKS])}")

        # 주요 통계 출력
        subject_count = summary_df['subject_id'].nunique()
        print(f"Data includes {subject_count} subjects")

        # 각 task별 파일 수 요약
        task_summary = summary_df.groupby('task_id')['original_file'].count()
        for task_id, count in task_summary.items():
            print(f"  T{task_id}: {count} files")
    else:
        print("No files were extracted. Check for errors.")

if __name__ == "__main__":
    main()