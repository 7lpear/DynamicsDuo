# SO-101 흰 돌 시연 녹화·재생과 SmolVLA 학습

Mac에서 리더암으로 흰 돌 집기 동작을 녹화하고, 팔로워암으로 저장된 동작을 재생하는 안내입니다.
로봇 연결·보정, 카메라 원격 조작, 녹화·재생과 데이터 준비 이후에 SmolVLA를 미세조정하는 순서를 설명합니다.

기존 로컬 데이터는 성공 시연 18개, 10,794프레임, 30fps, 상단 카메라 640×480입니다.
원본 20개 중 로그상 에피소드 **6, 18**을 제외했습니다(0부터 시작하는 번호).
흰 돌을 공급 위치에서 집어 **중앙 교차점 한 곳**에 놓는 작업입니다.
시연 영상과 행동 데이터는 별도로 복사하며 GitHub에 포함하지 않습니다.

## 1. 브랜치 받기

```sh
git clone --branch feat/smolvla-portable-training https://github.com/7lpear/DynamicsDuo.git
cd DynamicsDuo
```

이미 이 저장소를 받은 컴퓨터에서는 로컬 변경을 먼저 정리한 후:

```sh
git fetch origin
git switch feat/smolvla-portable-training
git pull --ff-only
```

## Mac 환경 준비

기존에 만든 환경이 있다면 활성화합니다.

```sh
conda activate gomoku-vla
```

새 Mac에서 환경을 준비할 경우 저장소 루트에서 실행합니다.

```sh
conda create -n gomoku-vla python=3.12 -y
conda activate gomoku-vla
conda install ffmpeg -c conda-forge -y
python -m pip install -r smolvla/requirements-mac.txt
```

`requirements-mac.txt`는 기존 녹화 컴퓨터에서 확인한 버전입니다. Mac의 OS/아키텍처에 따라 설치 가능 여부가 달라질 수 있습니다.
`lerobot-record` 같은 명령을 찾지 못하면 현재 터미널의 Python 환경을 먼저 확인합니다.

```sh
python --version
python -m pip show lerobot
```

## 시연 녹화와 그대로 재생

녹화·재생은 실제 로봇과 카메라가 연결된 컴퓨터에서 수행합니다.
`lerobot-replay`는 저장된 관절 명령을 순서대로 재생하고,
`lerobot-rollout`은 모델이 영상과 상태를 보고 새로운 동작을 예측합니다.
녹화 직후 동작을 그대로 확인할 때는 미세조정 없이 `lerobot-replay`를 사용합니다.

### 환경, 포트, 카메라 확인

기존 Mac에서는 `conda activate gomoku-vla`를 실행합니다.
새 Mac 환경은 위 [Mac 환경 준비](#mac-환경-준비)의 설치 명령으로 준비합니다.
기존 Python 환경에서 하드웨어 도구가 빠져 있다면 추가합니다.

```sh
python -m pip install "lerobot[core_scripts,feetech]==0.7.0"
lerobot-find-port
lerobot-find-cameras opencv
```

`lerobot-find-port` 안내가 나오면 확인하려는 암의 USB만 뽑고 Enter를 누릅니다.
출력된 포트를 기록하고 USB를 다시 연결한 뒤 다른 암도 확인합니다.
기존 Mac의 마지막 확인값은 팔로워 `/dev/tty.usbserial-120`, 리더 `/dev/tty.usbserial-130`, 상단 카메라 `0`입니다.
아래 명령은 그 값을 사용합니다. 다른 컴퓨터나 USB 재연결 후에는 확인된 값으로 바꾸세요.
Windows 포트는 보통 `COM3` 같은 형식이고 Linux는 `/dev/ttyACM0` 같은 형식입니다.

기존 보정 ID는 팔로워 `gomoku_follower`, 리더 `gomoku_leader`입니다.
새 제어 컴퓨터에서는 해당 암의 보정 파일을 옮기거나 다음을 각각 실행하고 터미널 안내에 따라 보정합니다.
이미 같은 암의 유효한 보정 파일이 있다면 재사용할 수 있습니다.

```sh
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbserial-120 --robot.id=gomoku_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbserial-130 --teleop.id=gomoku_leader
```

카메라는 공급 위치, 보드 중앙 교차점과 그리퍼가 모두 보이도록 고정합니다.
Mac에서 iPhone 카메라가 잡히면 iPhone의 연속성 카메라 설정을 끄고 다시 카메라 번호를 확인합니다.

### 카메라와 함께 원격 조작

아래 명령은 Windows PowerShell에서도 붙여 넣기 쉽도록 한 줄로 작성했습니다.

```sh
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/tty.usbserial-120 --robot.id=gomoku_follower --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=/dev/tty.usbserial-130 --teleop.id=gomoku_leader --display_data=true
```

리더암으로 흰 돌을 집어 중앙에 놓고 그리퍼를 위로 빼 봅니다.
영상과 동작이 정상인지 확인한 후 `Ctrl+C`로 종료하고 녹화로 넘어갑니다.
같은 로봇 포트를 원격 조작과 녹화 프로그램이 동시에 사용하지 않도록 합니다.

### 한 번 시험 녹화

돌을 공급 위치에 놓고 로봇을 시작 자세로 준비한 뒤 실행합니다.

```sh
lerobot-record --robot.type=so101_follower --robot.port=/dev/tty.usbserial-120 --robot.id=gomoku_follower --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=/dev/tty.usbserial-130 --teleop.id=gomoku_leader --dataset.repo_id=local/gomoku_white_single_demo --dataset.single_task="Pick up the white stone from the supply and place it at the center intersection of the board." --dataset.num_episodes=1 --dataset.episode_time_s=20 --dataset.reset_time_s=20 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

녹화 안내가 나오면 **흰 돌 집기 → 중앙에 놓기 → 그리퍼 위로 빼기**를 수행합니다.
20초를 채우려고 기다리거나 동작을 서두를 필요는 없습니다.

- `→` 또는 `n`: 현재 녹화/준비 시간을 일찍 종료합니다.
- `←` 또는 `r`: 현재 에피소드를 취소하고 다시 녹화합니다.
- `Esc` 또는 `q`: 세션을 종료하고 저장 처리를 진행합니다.

Mac에서 키가 반응하지 않으면 녹화 터미널을 활성화하고 문자 키 `n`, `r`, `q`를 사용합니다.
종료 후 영상 인코딩과 연결 해제가 끝나 터미널 프롬프트로 돌아올 때까지 기다립니다.

학습용 20회를 수집하려면 위 명령의 `--dataset.num_episodes=1`을 `20`으로,
`--dataset.repo_id=local/gomoku_white_single_demo`를 `local/gomoku_white_center_20`으로 바꿉니다.
매회 준비 시간에는 돌을 공급 위치로 돌려놓고 리더암으로 팔로워를 시작 자세에 복귀시킵니다.
실수한 회차는 `r`로 다시 녹화하거나 로그의 에피소드 번호를 기록해 나중에 제외합니다.
에피소드 번호는 0부터 시작합니다.

### 저장 폴더 확인

기본 저장 경로는 `~/.cache/huggingface/lerobot/local/`입니다.
`HF_LEROBOT_HOME`을 설정한 경우에는 그 경로 아래 `local/`에 저장됩니다.
LeRobot 0.7.0은 데이터셋 이름 뒤에 날짜·시간을 붙일 수 있으므로 실제 폴더명을 확인해야 합니다.
다음은 OS에 관계없이 유효한 기본/설정 경로를 출력합니다.

```sh
python -c "from pathlib import Path; from lerobot.utils.constants import HF_LEROBOT_HOME; p=Path(HF_LEROBOT_HOME)/'local'; print(p); print('\n'.join(x.name for x in sorted(p.glob('gomoku_white*')) if x.is_dir()))"
```

예를 들어 `gomoku_white_single_demo_20261118_120000`이 출력되면 재생할 ID는
`local/gomoku_white_single_demo_20261118_120000`입니다.
저장 폴더의 `meta/info.json`에서 `total_episodes`와 `total_frames`를 확인할 수 있습니다.

### 방금 녹화한 동작을 그대로 재생

리더암과 카메라 설정은 필요하지 않습니다. 팔로워와 녹화 데이터가 필요합니다.
돌과 보드를 녹화 때 위치에 놓고, 팔로워를 녹화 시작 자세와 가깝게 준비합니다.
아래 `local/ACTUAL_DATASET_FOLDER`를 위에서 확인한 실제 데이터셋 ID로 바꾸세요.

```sh
lerobot-replay --robot.type=so101_follower --robot.port=/dev/tty.usbserial-120 --robot.id=gomoku_follower --dataset.repo_id=local/ACTUAL_DATASET_FOLDER --dataset.episode=0
```

`--dataset.episode=0`은 첫 번째 녹화입니다. 다른 회차는 저장된 0부터 시작하는 번호로 바꿉니다.
데이터를 기본 캐시가 아닌 다른 폴더에 옮겼다면 `--dataset.root="DATASET_DIR"`도 지정합니다.
이 저장소의 18개 데이터셋을 `prepare`로 풀어 놓은 경우에는 저장소 루트에서 다음처럼 재생할 수 있습니다.

```sh
lerobot-replay --robot.type=so101_follower --robot.port=/dev/tty.usbserial-120 --robot.id=gomoku_follower --dataset.repo_id=local/gomoku_white_center_18_clean --dataset.root=smolvla/data/gomoku_white_center_18_clean --dataset.episode=0
```

실행하면 실제 로봇이 움직입니다. 예상과 다르면 `Ctrl+C`로 중단합니다.
재생은 관절 명령을 따라가므로 실제 궤적과 돌 배치가 녹화 때와 완전히 같다고 보장할 수는 없습니다.
돌이 다른 곳에 있어도 영상으로 찾아 경로를 수정하지 않습니다.
정리한 18개 데이터셋에서는 원래 로그 번호 6, 18을 뺀 뒤 나머지를 0~17로 다시 매겼습니다.

## 기존 녹화 데이터 준비와 검사

시연 영상은 GitHub에 포함하지 않습니다. 원래 Mac의 데이터를 별도로 복사해야 합니다.
Mac에서 이전에 만든 `gomoku_white_center_18_clean.zip`에는 macOS 메타데이터가 들어갈 수 있으므로,
이번 작업 폴더에서 준비한 다음 ZIP을 옮기세요.

```text
/Users/choiyunchang/Desktop/rlproject/gomoku_white_center_18_clean.zip
```

옮긴 컴퓨터의 `DynamicsDuo/smolvla/datasets/`에 이 파일을 넣은 뒤 아래 명령을 실행합니다.
데이터 원본 폴더는 Mac의 `~/.cache/huggingface/lerobot/local/gomoku_white_center_18_clean`에 있습니다.

```sh
python smolvla/run.py prepare
python smolvla/run.py check --device cpu
```

`prepare`는 별도로 옮긴 ZIP의 SHA-256을 확인하고 `smolvla/data/gomoku_white_center_18_clean/`에 압축을 풉니다.
`check`는 지정한 장치의 행렬 계산, 에피소드/프레임 수, 작업 설명과 비디오 디코딩을 확인합니다.
시연 ZIP, 압축 해제 데이터와 학습 결과는 모두 Git에서 제외됩니다.

## SmolVLA 미세조정

녹화 파일을 만들고 그대로 재생하는 것만으로는 SmolVLA가 학습되지 않습니다.
기본 모델을 우리 환경에 맞추기 위해, 성공 시연의 영상·현재 관절 상태·작업 명령을 입력하고
시연에서 기록한 행동을 학습 목표로 삼아 모델 가중치를 업데이트하는 미세조정이 필요합니다.
이는 시연을 따라 배우는 모방학습이며, AlphaZero 오목 전략 학습과는 별도입니다.

현재 준비한 것은 흰 돌을 중앙 교차점에 놓는 성공 시연 **18개**입니다.
먼저 이 데이터로 미세조정한 뒤, 새로 준비한 시작 상태에서 집기와 정확한 배치 성공률을 확인합니다.
18개만으로 안정적인 성능이 보장되지는 않으므로 실패가 반복되는 상황의 시연을 추가합니다.
중앙 한 곳에 놓는 데이터로 임의의 행·열 명령까지 학습했다고 판단하지 않습니다.

저장소 루트에서 먼저 10스텝으로 데이터 로딩·역전파·저장 여부를 점검합니다.
기존 `gomoku-vla` 환경과 앞 절에서 준비한 데이터셋을 사용합니다.

```sh
conda activate gomoku-vla
python smolvla/run.py train --device auto --steps 10 --batch-size 1 --output smolvla/outputs/smoke
```

정상 종료하면 별도 출력 폴더에 첫 미세조정을 진행합니다.
다음 5,000스텝은 첫 실험 설정이며, 최적 학습량이나 성공률을 보장하는 값은 아닙니다.

```sh
python smolvla/run.py train --device auto --steps 5000 --batch-size 4 --output smolvla/outputs/white_center_v1
```

- `--device auto`는 사용 가능한 가속 장치를 선택하고, 없으면 CPU를 사용합니다. CPU에서는 학습이 오래 걸릴 수 있습니다.
- 메모리가 부족하면 새 출력 폴더를 지정하고 `--batch-size`를 `2` 또는 `1`로 줄입니다.
- 명령만 먼저 확인하려면 `--dry-run`을 추가합니다. 이 옵션에서는 실제 장치 점검이나 학습을 하지 않습니다.
- 원본 `lerobot/smolvla_base` 가중치를 불러오고, 녹화 데이터의 정규화 통계를 적용합니다.
- 모든 18개 에피소드를 학습에 사용하며 별도 검증 분할은 하지 않습니다. 학습 손실만으로 실제 성공을 판단하지 마세요.
- 학습 컴퓨터에 로봇이나 카메라를 연결할 필요는 없습니다. 학습 중 로봇은 움직이지 않습니다.

학습 결과는 `smolvla/outputs/white_center_v1/checkpoints/`에 저장됩니다.
다음으로 최신 완료 체크포인트를 내보냅니다.

```sh
python smolvla/run.py export --run smolvla/outputs/white_center_v1
```

`smolvla/exports/gomoku_smolvla.zip`에는 `pretrained_model` 폴더 전체가 들어갑니다.
가중치와 함께 모델 설정·전처리·후처리·정규화 파일을 사용해야 합니다.
이 폴더를 `lerobot-rollout`의 `--policy.path`로 지정하면, 학습한 모델의 예측으로 로봇을 제어할 수 있습니다.
이는 앞 절의 `lerobot-replay`와 다르며, 실제 로봇에서 짧게 실행해 집기·배치 성공 여부를 평가해야 합니다.

## 참고와 검증 범위

- [LeRobot 실제 로봇 모방학습 안내](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot SO-101 안내](https://huggingface.co/docs/lerobot/so101)
- [LeRobot 설치 안내](https://huggingface.co/docs/lerobot/installation)

로컬 Mac에서 데이터 준비·18개 에피소드와 비디오 검사를 확인했습니다.
실제 로봇을 움직일 때는 현재 연결된 포트와 해당 암의 보정 파일을 사용합니다.
