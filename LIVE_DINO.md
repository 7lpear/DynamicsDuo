# 웹캠 DINO 탐지 사용

Conda `so-arm-101`에서 실행한다. 로봇 제어와 보드 확정은 포함하지 않는다.
기존 calibration/board.json과 동일한 카메라/판 위치 및 해상도가 필요하다.

```powershell
conda activate so-arm-101
python live_dino.py --index 1
```

Windows 카메라 앱과 camera_test.py는 먼저 종료한다.
모델이 로딩되는 동안 현재 카메라 영상이 먼저 보이고, 첫 분석이 끝나면 영상 위에 박스를 표시한다.
CPU에서 분석은 수 초마다 갱신된다. 모델은 한 번만 로딩한다.

- 한 화면에 원본 카메라 영상과 탐지 박스를 겹쳐 표시한다. 보정 좌표의 박스를 원본 좌표로 되돌린다.
- 영상은 계속 움직이고, 박스는 분석이 끝날 때마다 갱신된다.
- detection age: 분석에 사용한 사진이 촬영된 뒤 지난 초. 돌이 움직인 직후에는 이전 위치에 박스가 남을 수 있다.
- P: 분석 일시정지/재개. 이미 진행 중인 분석은 완료 후 표시될 수 있다.
- S: 현재 보정 영상, 마지막 분석 입력/박스, 시간 정보와 JSON 저장.
- Q 또는 Esc: 종료. 진행 중인 분석/모델 로딩 종료를 기다릴 수 있다.
- 저장 폴더: live_dino_results/실행시각/. live_board.png와 analyzed_input.png는 다른 시점일 수 있다.
- live_overlay.png: 저장 시점에 표시한 카메라 영상과 탐지 박스 화면.

처음에는 빈 판으로 시작해 탐지가 0개인지 확인한다.
백돌 하나를 놓고 손을 뺀 뒤 1~2회 결과 갱신을 기다린다.
이후 3개를 놓거나 위치를 바꿔 결과를 확인한다.
문제가 있으면 S로 저장하고 당시 실제 돌 개수/배치를 기록한다.

기본 탐지 설정은 bottle cap., box threshold 0.25, text threshold 0.15.
대상 수를 3개로 고정하거나 상위 3개만 남기는 제한은 없다.
터미널과 저장 JSON의 counts에서 신뢰도 통과 후보 수, 크기 필터 후 개수, 중복 제거 후 개수를 확인할 수 있다.
보정 좌표의 돌 크기 필터와 중복 제거를 적용한다. 흑백 분류는 하지 않는다.
카메라나 판이 움직이면 종료 후 새 빈 판 사진으로 calibrate_board.py를 다시 실행한다.
실제 해상도가 보정과 다르면 잘못된 좌표를 사용하지 않고 중단한다.

카메라가 열리지 않을 때:

```powershell
python live_dino.py --index 1 --backend msmf
```

USB 카메라를 열지 않는 반복 추론 검사:

```powershell
python live_dino.py --replay captures/board_20261029_123855_712442.png --no-window --stop-after-results 2 --duration 120
```

현재 단계의 결과는 돌 후보다. 손/로봇팔 가림 감지, 9x9 교점 대응,
시간에 따른 보드 확정과 착수 검증은 후속 단계에서 구현한다.
