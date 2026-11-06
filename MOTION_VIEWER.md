# SO-101 스크립트 + 실시간 3D 화면

## 리더암 텔레오퍼레이션과 3D 동기화

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

현재 환경에는 연결을 적용했습니다. LeRobot을 재설치하거나 새 환경을 만들면 한 번 실행합니다.

```powershell
.venv\Scripts\python.exe setup_lerobot_viewer.py
```

선택 옵션: `--follower_viewer=false`로 뷰어를 끄고, `--follower_viewer_browser=false`로 브라우저 자동 열기를 끄거나,
`--follower_viewer_port=8766`으로 포트를 바꿀 수 있습니다. `--display_data`는 LeRobot 자체 시각화 옵션입니다.

뷰어는 각 루프의 `get_observation()`에서 읽은 엔코더 값을 사용합니다.
명령 전송 직전 관측이며, 목표값을 실제 위치로 대신 표시하지 않습니다.
몸체 관절은 도, 그리퍼는 %입니다. `use_degrees=false`일 때도 표시만 도 단위로 변환하고 제어 명령은 그대로 둡니다.

리더·팔로워 읽기의 `ConnectionError`에 한해 최대 3회(첫 시도 포함)까지 재시도합니다.
0.25초가 지나면 추가 시도를 시작하지 않으며 개별 SDK 읽기 시간은 별도입니다.
이전 관측으로 움직임을 계속하거나 쓰기 명령을 재전송하지 않습니다. 재시도 중에는 새 움직임 명령을 보내지 않습니다.
계속 실패하면 기본 종료 처리를 수행합니다. 반복 오류 시 `--fps=30`과 USB·모터 연결 상태를 확인하세요.

관절 보정에는 기존 `calibrate_leader.py --port COM6` 및
`calibrate_follower.py --port COM10 --robot-id follower` 도구를 사용합니다.
기존 캘리브레이션과 장비 백업은 유지하며 별도 자세 대응 오프셋 파일은 사용하지 않습니다.
설치 스크립트는 알려진 소스 구조에만 연결을 적용하고, 원본은
`logs/lerobot_teleoperate.original.py`에 보관합니다. SDK 재설치 시 다시 적용하세요.

## 읽기 전용 실시간 관측

텔레오퍼레이션을 종료한 뒤 실행합니다. 동일 COM 포트를 동시에 열 수는 없습니다.

```powershell
.venv\Scripts\python.exe monitor_follower.py --port COM10 --robot-id follower --duration 300
```

팔로워 각도와 전압·온도·전류/부하 원시값·상태를 표시합니다. 현재 토크 상태를 바꾸지 않으므로
토크가 꺼져 있다면 팔을 받쳐 주세요. `--fps` 기본 10Hz, 진단값은 약 0.5초마다 읽습니다.
`--no-browser`로 자동 브라우저 열기를 끄고 `--viewer-port`로 HTTP 포트를 바꿀 수 있습니다.
`--duration` 또는 Ctrl+C로 끝나며, 연결 해제 시에도 토크 상태를 유지합니다.
관절 엔코더 위치를 시각화하는 도구로, 카메라 질량 추정이나 물리적 처짐 계산은 하지 않습니다.

한 번만 상태를 읽어 JSON으로 출력하려면 다음을 사용합니다.

```powershell
.venv\Scripts\python.exe -m gomoku_robot read-state --port COM10 --robot-id follower
```

## 로봇 없이 뷰어 검사

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe verify_viewer.py
```

Microsoft Edge를 headless로 열어 가상 관측값의 위치·목표·진단 표시와 데이터 정지 상태를 검사합니다.
USB와 LeRobot 모터 제어를 사용하지 않습니다. 결과 화면은 `logs/viewer-verified.png`에 저장합니다.

## 외부 자료

URDF와 관절 매핑은 [LeLab](https://github.com/huggingface/leLab), 렌더링은
[Three.js](https://github.com/mrdoob/three.js)를 사용합니다. 다운로드 버전은 `viewer/assets-version.json`,
라이선스는 `viewer/LELAB-LICENSE.txt`와 `viewer/vendor/LICENSE.txt`에 있습니다.
모델과 JavaScript는 로컬에서 제공됩니다. 파일이 없는 경우 `python setup_viewer.py`로 준비합니다.
