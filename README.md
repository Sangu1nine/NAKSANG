# IMU 데이터 수집 및 WiFi 전송 시스템

이 프로젝트는 MPU-6050 가속도/자이로 센서에서 데이터를 수집하고, WiFi를 통해 로컬 PC로 전송하는 시스템입니다.

## 요구사항

### 센서 장치 (라즈베리 파이 등)
- Python 3.x
- SMBus2 라이브러리 (`pip install smbus2`)
- Bitstring 라이브러리 (`pip install bitstring`)
- NumPy 및 Pandas (`pip install numpy pandas`)

### 수신 PC (윈도우, 맥, 리눅스 등)
- Python 3.x
- NumPy 및 Pandas (`pip install numpy pandas`)

## 설정 방법

1. 센서 장치에서 `get_data.py` 파일의 다음 부분을 수정합니다:
```python
# WiFi 통신 설정
WIFI_SERVER_IP = '192.168.0.100'  # 로컬 PC의 IP 주소 (변경 필요)
WIFI_SERVER_PORT = 5000           # 통신 포트
```
여기서 `WIFI_SERVER_IP`를 데이터를 수신할 PC의 IP 주소로 변경하세요.

2. 수신 PC에서 `client_receiver.py` 파일의 다음 부분을 필요에 따라 수정합니다:
```python
# 서버 설정
SERVER_IP = '0.0.0.0'   # 모든 네트워크 인터페이스에서 연결 수신
SERVER_PORT = 5000      # 통신 포트 (센서 장치와 동일하게 설정)
DATA_FOLDER = 'received_data'  # 수신 데이터가 저장될 폴더
```

## 사용 방법

### 수신 PC 측
1. 수신 PC에서 서버를 먼저 실행합니다:
```
python client_receiver.py
```
2. 서버가 시작되면 "클라이언트 연결 대기 중..." 메시지가 표시됩니다.

### 센서 장치 측
1. 센서 장치에서 데이터 수집 및 전송 프로그램을 실행합니다:
```
python get_data.py
```
2. 프로그램이 시작되면 센서 데이터를 수집하고 WiFi를 통해 수신 PC로 전송합니다.
3. 기본적으로 로컬에도 CSV 파일로 데이터가 저장됩니다.
4. 'Ctrl+C'를 눌러 데이터 수집을 종료할 수 있습니다.

## 데이터 형식

수집되는 IMU 데이터는 다음과 같은 형식으로 전송 및 저장됩니다:
- 타임스탬프 (초)
- 가속도 X, Y, Z (g 단위)
- 자이로스코프 X, Y, Z (°/s 단위)

## 문제 해결

1. WiFi 연결 실패:
   - 양쪽 장치의 IP 주소와 포트 설정이 올바른지 확인하세요.
   - 방화벽 설정에서 지정된 포트를 허용하세요.
   - 같은 네트워크에 연결되어 있는지 확인하세요.

2. 데이터 수신 오류:
   - 수신 PC에서 서버가 먼저 실행되고 있는지 확인하세요.
   - 네트워크 상태가 안정적인지 확인하세요.

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다.