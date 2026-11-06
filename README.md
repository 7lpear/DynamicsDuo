# DynamicsDuo
2026-2 Hongik Challenge

SO-101 로봇 제어 실험과 C920/C920e 기반 9×9 오목판 비전 프로젝트.

## 실행 명령어 모음

모든 명령은 **저장소 루트의 Windows PowerShell**에서 실행합니다.
카메라·비전은 Conda의 `python`, 로봇 연결은 LeRobot이 설치된 `.venv\Scripts\python.exe`를 사용합니다.
현재 연결은 리더 `COM6` / 팔로워 `COM10` 기준입니다. USB 재연결 후 포트를 확인하세요.
카메라 번호 `1`도 연결 환경에 맞춰 변경합니다.

| 지금 하려는 일 | 실행할 항목 |
|---|---|
| 새 터미널에서 시작 / 로봇 포트 찾기 | [기본 준비](#기본-준비) |
| 오목판의 돌 위치·흑백 확인 | [오목판 카메라 인식](#오목판-카메라-인식) |
| 전원·온도·보정 일치 확인 | [로봇 상태 한 번 확인](#로봇-상태-한-번-확인) |
| 실측 관절·전류·부하를 3D 화면에서 관측 | [실시간 관측](#실시간-관측) |
| 리더암으로 팔로워를 조작하며 3D 화면에 반영 | [리더암 조작과 3D 동기화](#리더암-조작과-3d-동기화) |
| 관절 보정값 불일치 / 다시 보정 | [관절 캘리브레이션](#관절-캘리브레이션) |
| 로봇 없이 3D 화면 검증 | [3D 화면 검증](#3d-화면-검증) |
| E5 등 교점을 로봇 XYZ로 변환 | [판과 로봇 좌표 보정](#판과-로봇-좌표-보정) |
| 학습 모델 탐지 경로를 비교·실험 | [선택 사항 DINO 탐지](#선택-사항-dino-탐지) |
| 코드를 수정한 뒤 확인 | [하드웨어 없는 검증](#하드웨어-없는-검증) |

### 기본 준비

새 터미널에서 작업을 시작할 때 실행합니다. `.venv` 명령은 별도 활성화가 필요 없습니다.

```powershell
Set-Location 'C:\Users\peter\Desktop\so-arm-101'
conda activate so-arm-101
```

로봇이 연결되지 않거나 COM 번호를 모를 때 포트 목록을 확인합니다.

```powershell
.venv\Scripts\python.exe -m serial.tools.list_ports
```

같은 로봇 COM 포트를 사용하는 다른 프로그램은 연결을 종료한 뒤 실행하세요.

### 오목판 카메라 인식

평소 돌 위치·흑백을 확인할 때 사용합니다. 격자를 확인한 뒤 **빈 판에서 `B`**를 눌러 기준을 수집합니다.

```powershell
python gomoku_vision/board_camera.py --camera 1 --size 9
```

카메라·판 위치가 바뀌었거나 처음 설치했다면 격자와 빈 판 기준을 다시 잡습니다.
아래 명령은 **판에 돌과 손이 없는 상태**에서 실행합니다.

```powershell
python gomoku_vision/board_camera.py --camera 1 --size 9 --reset --capture-empty
```

이전에 기준을 저장했고 카메라·판이 전혀 움직이지 않았다면 기준을 복원할 수 있습니다.

```powershell
python gomoku_vision/board_camera.py --camera 1 --size 9 --resume-baseline
```

`--size 9`는 8×8칸 판의 **9×9 교점**입니다. `N`: 카메라 전환, `A`: 격자 재검출,
`B`: 빈 판 기준 재수집, `S`: 저장, `D`: 진단 저장, `Q`/Esc: 종료.
`PAUSED`는 이전 판정이므로 새 착수 판단에 사용하지 않습니다.
설치·수동 보정·카메라 오류는 [비전 상세 안내](gomoku_vision/README.md)를 참고하세요.

### 로봇 상태 한 번 확인

전원을 연결한 뒤, 보정값·현재 각도·전압·온도·오류 상태를 확인할 때 사용합니다. **읽기 전용**입니다.

```powershell
.venv\Scripts\python.exe -m gomoku_robot read-state --port COM10 --robot-id follower
```

### 실시간 관측

장착물이나 자세가 바뀐 뒤 **팔을 움직이는 명령 없이** 실제 상태를 지켜볼 때 사용합니다.
3D 뷰어에 실제 각도·전류/부하 원시값·전압·온도를 5분간 표시합니다. Ctrl+C로 일찍 종료할 수 있습니다.
모터 토크가 꺼져 있으면 팔을 받쳐야 합니다.

```powershell
.venv\Scripts\python.exe monitor_follower.py --port COM10 --robot-id follower --duration 300
```

### 리더암 조작과 3D 동기화

LeRobot 기본 텔레오퍼레이션과 기존 3D 뷰어를 함께 실행합니다.
리더 COM6 / 팔로워 COM10 기준이며, 재연결 시 포트를 확인하세요.
저장소 루트에서 실행합니다. 같은 포트를 사용하는 다른 프로그램은 먼저 종료합니다.

```powershell
.venv\Scripts\python.exe -m lerobot.scripts.lerobot_teleoperate --robot.type=so101_follower --robot.port=COM10 --robot.id=follower --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=my_leader --teleop.calibration_dir=calibration/teleop_profiles
```

명령 실행 시 브라우저에서 `http://127.0.0.1:8765`가 열리고 바로 텔레오퍼레이션이 시작됩니다.
`t` 등의 시작 입력은 없습니다. 보정 파일이 맞지 않으면 LeRobot 자체 보정 절차가 표시될 수 있습니다.
제어는 LeRobot 기본 동작을 사용하므로, 실행 전 양팔을 비슷한 자세로 두고 경로를 비우세요.
3D 모델은 제어 루프가 이미 읽은 팔로워 실측값, 목표 열은 실제 전송 명령을 표시합니다.
별도 USB 읽기나 중복 포트 연결은 없으며 전류·부하 진단을 추가로 읽지는 않습니다.
Ctrl+C로 종료하면 로봇 연결과 뷰어 서버를 닫고 기본 설정에 따라 토크를 해제합니다. 팔을 받칠 준비를 하세요.

현재 환경에는 연결을 적용했습니다. 새 로봇 환경에는 먼저 아래 의존성을 설치합니다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-robot.txt
```

LeRobot을 재설치하거나 새 환경을 만들면 뷰어 연결을 한 번 적용합니다.

```powershell
.venv\Scripts\python.exe setup_lerobot_viewer.py
```

선택 옵션: `--follower_viewer=false`로 뷰어를 끄고, `--follower_viewer_browser=false`로 브라우저 자동 열기를 끄거나,
`--follower_viewer_port=8766`으로 포트를 바꿀 수 있습니다. `--display_data`는 LeRobot 자체 시각화 옵션입니다.

리더·팔로워 통신 읽기 실패는 첫 시도를 포함해 최대 3회까지 재시도합니다. 0.25초가 지나면
추가 시도를 시작하지 않으며, 개별 SDK 읽기의 시간 초과까지 강제로 단축하지는 않습니다.
이전 리더 값 재사용이나 모터 쓰기 재시도는 없습니다. 계속 실패하면 연결을 정리하고 중단합니다.
실패가 반복되면 `--fps=30`으로 확인하고 USB·허브·모터 연결 배선을 점검하세요.

### 관절 캘리브레이션

`Calibration missing or mismatched`가 나오거나 관절 영점·범위를 다시 측정할 때 사용합니다.
**매번 실행할 필요는 없습니다.** 기존 PC/장비 보정값은 자동 백업됩니다.

리더암(COM6)을 다시 보정할 때는 다음을 실행합니다. 텔레오퍼레이션이 사용하는
`calibration/teleop_profiles/my_leader.json`에 저장됩니다.

```powershell
.venv\Scripts\python.exe calibrate_leader.py --port COM6
```

팔로워암(COM10)을 다시 보정할 때는 다음을 실행합니다.

```powershell
.venv\Scripts\python.exe calibrate_follower.py --port COM10 --robot-id follower
```

팔을 받치고 모든 관절을 중간 자세에 둔 뒤 Enter → 손목 회전을 제외한 각 관절의 가동 범위를
천천히 기록 → Enter → `PASS: PC/device calibration match` 확인 순서입니다.
보정 완료 후 읽기 전용 상태 확인을 다시 실행합니다.
리더의 로컬 보정 파일은 위 LeRobot 명령의 `--teleop.calibration_dir`로 지정합니다.
쓰기 응답이 누락되어도 저장값을 다시 읽어 확인합니다. 최종 PASS 전에 중단됐다면 일부 보정값이
변경됐을 수 있으므로 보정을 끝낸 뒤 텔레오퍼레이션을 실행하세요.
[자세한 보정 절차](gomoku_robot/README.md#관절-보정을-다시-하는-경우)

### 3D 화면 검증

로봇 없이 모델 로딩·관절 표시·오류 상태를 자동 확인합니다. USB 연결이나 모터 명령은 없습니다.
Playwright와 로컬 Microsoft Edge가 필요합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe verify_viewer.py
```

### 판과 로봇 좌표 보정

관절 보정이 끝난 뒤 교점과 로봇 XYZ를 연결할 때 사용합니다. **계산·미리보기 전용**입니다.
측정 파일 생성은 처음 한 번만 하며, 이미 존재하면 덮어쓰지 않습니다.

```powershell
python -m gomoku_robot init
```

`calibration/robot_board.session.json`에 배치 이름, 네 모서리와 독립 검증점의 XYZ 실측값을 입력한 뒤 실행합니다.

```powershell
# 실측값의 보정 오차 확인
python -m gomoku_robot check

# 중앙 E5의 판 표면 위치와 30mm 위 목표 확인
python -m gomoku_robot target --coordinate E5 --hover-mm 30

# 실측 없이 명령 형식만 살펴보는 가상 예시
python -m gomoku_robot demo
```

현재 TCP 위치를 측정했다면 접근 경로도 계산할 수 있습니다.
**아래 `<현재X> <현재Y> <현재Z>`를 같은 로봇 좌표계의 실측 숫자(mm)로 바꾼 뒤** 실행하세요.

```powershell
python -m gomoku_robot plan-hover --coordinate E5 --hover-mm 30 --start-xyz-mm <현재X> <현재Y> <현재Z>
```

관절 상태를 파일로 기록하고 싶을 때는 다음 읽기 전용 명령을 사용합니다.
기존 출력 파일은 덮어쓰지 않으므로 다시 저장할 때는 파일명을 바꾸세요. 관절 각도는 XYZ 실측값이 아닙니다.

```powershell
.venv\Scripts\python.exe -m gomoku_robot read-state --port COM10 --robot-id follower --out calibration/robot_board.pose.json
```

보정 좌표계·입력 방법은 [판–로봇 좌표 상세 안내](gomoku_robot/README.md)를 참고하세요.

### 선택 사항 DINO 탐지

현재 OpenCV 돌 인식과 별도로 학습 모델 탐지를 비교할 때 사용합니다.
아래 의존성 설치는 DINO 환경을 처음 준비할 때 실행합니다.

```powershell
python -m pip install -r requirements-vision.txt
python live_dino.py --index 1
```

`S`: 저장, `P`: 분석 일시정지/재개, `Q`: 종료. CPU 탐지 박스는 수 초 간격으로 갱신될 수 있습니다.

카메라 미리보기와 빈 판 사진 저장이 필요할 때:

```powershell
python camera_test.py --index 1
```

DINO 경로의 판 보정은 아래 사진 경로를 **새로 저장한 빈 판 사진 경로로 바꿔서** 실행합니다.
이 보정은 `gomoku_vision/`의 자동 격자 보정과 별개입니다.

```powershell
python calibrate_board.py <새로저장한빈판사진경로>
```

동봉한 `calibration/board.json`은 샘플 사진용입니다. 실제 설치에 그대로 적용하지 마세요.
샘플 사진 한 장의 탐지 결과만 확인할 때는 다음을 사용합니다.

```powershell
python detect_grounding_dino.py captures/board_20261029_123855_712442.png --rectify --stone-filter
```

### 하드웨어 없는 검증

코드를 수정한 뒤 모의 테스트를 실행할 때 사용합니다. 로봇·카메라를 연결하지 않습니다.

```powershell
# 로봇 제어, 피드백, 보정과 공통 비전 테스트
.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py"

# OpenCV 격자·돌 인식 테스트: Conda 환경
python -m unittest discover -s gomoku_vision -p "test_*.py" -v

# 3D 화면의 자동 브라우저 검증: 가상 데이터만 사용
.venv\Scripts\python.exe verify_viewer.py
```

## 주요 파일

| 파일 | 기능 |
|---|---|
| camera_test.py | 카메라 미리보기와 원본 사진 저장 |
| calibrate_board.py | 네 교점을 클릭해 9×9 원근 보정 |
| live_dino.py / dino_detector.py | 실시간 영상과 비동기 DINO 탐지 |
| board_calibration.py / detection_utils.py | 공통 보정과 탐지 후처리 |
| detect_grounding_dino.py | 사진 한 장의 DINO 탐지 |
| gomoku_vision/ | OpenCV 격자 자동 검출, 실시간 돌 유무·흑백 구분 |
| gomoku_robot/ | 판–로봇 XYZ 보정 검사, 중앙 위 접근 경로 계산, 읽기 전용 관절 상태 저장 |
| experiments/ | YOLO·Moku 비교, 초기 방식, 파라미터 실험 |
| monitor_follower.py | 모터 쓰기 없이 실측 각도와 진단값을 3D 뷰어에 표시 |
| lerobot_viewer.py | LeRobot 기본 루프의 팔로워 실측값을 3D 뷰어에 전달 |
| setup_lerobot_viewer.py | 설치된 LeRobot 기본 CLI에 뷰어 연결 적용 |
| follower_feedback.py | 관절 이름과 읽기 전용 모터 진단 |

## 문서와 현재 진행 상태

- [OpenCV 오목판·흑백 돌 인식 사용법](gomoku_vision/README.md)
- [판–로봇 좌표 보정과 접근 경로 준비](gomoku_robot/README.md)
- [실시간 탐지 사용법](LIVE_DINO.md)
- [탐지 모델 비교 결과](detector_benchmark/REPORT.md)
- [검증 이미지](detector_benchmark/validation_contact_sheet.png)
- [로봇 동작과 제한 사항](MOTION_VIEWER.md)
- [선택 실험 도구](experiments/README.md)

실시간 실행과 사진 실행은 동일한 DinoDetector 엔진과 보정/후처리를 공유합니다.
YOLO 비교에만 필요한 ultralytics는 experiments/requirements.txt로 분리했습니다.

기존 DINO 경로에서 원통형 백돌은 `bottle cap.` 프롬프트와 돌 크기/중복 필터로 실험했습니다.
개수 3개 제한은 없습니다. DINO 경로의 흑백 분류, 교점 배열 확정, 가림 처리는 후속 작업입니다.

별도 `gomoku_vision/` 도구는 학습 모델 없이 격자를 찾고 빈 판 기준과 원형 경계를
이용해 돌 유무·흑백을 판별합니다. Windows/macOS 모두 Conda `so-arm-101`에서 실행할 수 있으며,
기본 카메라 입력 방식은 OS에 따라 자동 선택됩니다. 설정은 [사용법](gomoku_vision/README.md)을 참고하세요.
판·카메라 이동 후에는 격자와 빈 판 기준을 다시 설정해야 합니다.
`gomoku_robot/`에서 네 모서리와 독립 검증점의 실측 XYZ로 판–로봇 좌표를 보정하고
목표 교점 위 접근 경로를 계산할 수 있습니다. 실측값은 직접 입력해야 하며,
역기구학·충돌 검사·모터 실행 및 실제 착수 연결은 아직 구현하지 않았습니다.
강화학습 에이전트도 아직 연결하지 않았습니다.
동일 배치에서 파생한 이미지의 성공은 새 배치나 실제 대국 성능을 보증하지 않습니다.

## 샘플과 로컬 파일

빈 판과 백돌 배치 샘플 2장을 captures/에 포함했습니다. 얼굴 테스트 사진,
새 촬영 이미지, 모델 가중치, 가상환경, 실행 결과는 Git에서 제외합니다.
다운로드 모델의 이용 조건은 각 모델 제공자의 라이선스를 따릅니다.
뷰어에 포함된 외부 자료는 viewer/의 라이선스 파일을 참고하세요.

## Git 작업

```powershell
git pull --ff-only
git status
git add 수정한파일
git commit -m "변경 내용"
git push origin main
```
