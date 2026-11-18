# SO-101 흰 돌 집기 SmolVLA 학습

Mac에서 수집한 시연을 NVIDIA GPU 컴퓨터에서 미세조정하고, 결과를 Mac에 가져와 실행하는 도구입니다.
Python 명령은 Windows의 Anaconda Prompt/PowerShell, Linux, macOS에서 같은 형태로 사용합니다.
GPU 드라이버와 가용 메모리는 각 컴퓨터에서 확인해야 합니다. 학습 컴퓨터에 로봇이나 카메라를 연결할 필요는 없습니다.

포함 데이터: 성공 시연 18개, 10,794프레임, 30fps, 상단 카메라 640×480.
원본 20개 중 로그상 에피소드 **6, 18**을 제외했습니다(0부터 시작하는 번호).
흰 돌을 공급 위치에서 집어 **중앙 교차점 한 곳**에 놓는 작업입니다.
여러 좌표에 놓기, 돌 색상 선택, AlphaZero 연동은 이 데이터의 학습 범위에 포함되지 않습니다.

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

## 2. RTX 5070 Ti 학습 환경

Conda가 설치된 터미널에서 실행합니다. `nvidia-smi`가 실패하면 NVIDIA 드라이버 설치/업데이트부터 진행합니다.

```sh
nvidia-smi
conda create -n gomoku-vla python=3.12 -y
conda activate gomoku-vla
conda install ffmpeg -c conda-forge -y
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r smolvla/requirements-cuda.txt
python -m pip check
```

Mac과 동일한 LeRobot **0.7.0**을 사용합니다. CUDA 학습용 PyTorch는 **2.9.1 + CUDA 12.8**,
영상 디코더 TorchCodec은 호환되는 **0.8.x**로 맞춥니다. NVIDIA 드라이버는 CUDA 12.8 실행을 지원해야 합니다.
위 PyTorch 배포에는 CUDA 런타임이 포함되므로 이 작업에 CUDA Toolkit을 따로 설치할 필요는 없습니다.
처음 모델과 토크나이저를 받을 때 인터넷 연결이 필요합니다. 학습 중 Hub 업로드와 외부 실험 로깅은 기본으로 꺼져 있습니다.

## 3. 데이터 준비와 검사

시연 영상은 GitHub에 포함하지 않습니다. 원래 Mac의 데이터를 별도로 복사해야 합니다.
Mac에서 이전에 만든 `gomoku_white_center_18_clean.zip`에는 macOS 메타데이터가 들어갈 수 있으므로,
이번 작업 폴더에서 준비한 다음 ZIP을 옮기세요.

```text
/Users/choiyunchang/Desktop/rlproject/DynamicsDuo-smolvla/smolvla/datasets/gomoku_white_center_18_clean.zip
```

옮긴 컴퓨터의 `DynamicsDuo/smolvla/datasets/`에 이 파일을 넣은 뒤 아래 명령을 실행합니다.
데이터 원본 폴더는 Mac의 `~/.cache/huggingface/lerobot/local/gomoku_white_center_18_clean`에 있습니다.

```sh
python smolvla/run.py prepare
python smolvla/run.py check --device cuda
```

`prepare`는 별도로 옮긴 ZIP의 SHA-256을 확인하고 `smolvla/data/gomoku_white_center_18_clean/`에 압축을 풉니다.
`check`는 실제 GPU 행렬 계산, 에피소드/프레임 수, 작업 설명과 비디오 디코딩을 확인합니다.
시연 ZIP, 압축 해제 데이터와 학습 결과는 모두 Git에서 제외됩니다.

## 4. 학습 실행

먼저 10스텝으로 모델 로딩, 역전파, 체크포인트 저장까지 되는지 점검합니다.

```sh
python smolvla/run.py train --device cuda --steps 10 --batch-size 2 --output smolvla/outputs/smoke
```

정상 완료 후 첫 실험을 시작합니다.

```sh
python smolvla/run.py train --device cuda --steps 5000 --batch-size 4 --output smolvla/outputs/white_center_v1
```

- 실행 전에 명령만 보려면 `--dry-run`을 붙입니다.
- 메모리 부족이면 새 출력 폴더를 지정하고 `--batch-size 2` 또는 `1`로 줄입니다.
- Windows에서도 시작하기 쉽도록 DataLoader 작업자 수는 0, 영상 디코더는 PyAV를 사용합니다.
- `top` 영상을 `camera1`으로 연결합니다. 모델 입력 특징은 데이터셋에서 다시 구성하여 상단 카메라 한 대만 사용합니다.
- 학습은 원본 `lerobot/smolvla_base`에서 시작합니다. 로컬 시험용 `smolvla_live_runtime`은 필요하지 않습니다.
- 학습 데이터의 관절 정규화 통계는 LeRobot 학습 파이프라인이 적용하고 체크포인트에 저장합니다.
- 저장 주기는 1,000스텝(그보다 짧은 실험은 마지막 스텝)입니다. 기존 출력 폴더는 덮어쓰지 않습니다.
- 5,000스텝은 첫 실험 설정입니다. 18개 시연만으로 성공률이 보장되지는 않습니다.
  이 설정은 전체 18개를 학습에 쓰며, 별도 검증 분할/로봇 성공률 평가는 수행하지 않습니다.

## 5. 학습 결과를 Mac으로 가져오기

```sh
python smolvla/run.py export --run smolvla/outputs/white_center_v1
```

최신 숫자 체크포인트의 `pretrained_model` 전체를 `smolvla/exports/gomoku_smolvla.zip`으로 묶습니다.
이 ZIP을 Mac으로 옮기고 압축을 풉니다. `config.json`, `model.safetensors`, 전처리/후처리 JSON과
정규화 `.safetensors`를 함께 사용해야 하므로 가중치 파일만 복사하지 마세요.
컴퓨터별 경로를 포함하는 심볼릭 링크가 아니라 실제 파일을 ZIP에 넣습니다.

## 6. Mac에서 실행

기존 `gomoku-vla` 환경을 사용합니다. 새 Mac에서 같은 환경을 만들려면:

```sh
conda create -n gomoku-vla python=3.12 -y
conda activate gomoku-vla
conda install ffmpeg -c conda-forge -y
python -m pip install -r smolvla/requirements-mac.txt
```

`requirements-mac.txt`는 녹화 컴퓨터에서 확인한 버전입니다. Mac의 OS/아키텍처에 따라 설치 가능 여부가 달라질 수 있습니다.

GPU 사용 여부를 확인합니다.

```sh
python -c "import torch; print(torch.backends.mps.is_available())"
```

로봇 보정은 물리적인 암에 대응합니다. 기존 Mac에서는 `gomoku_follower` ID의 보정을 사용하고,
새 제어 컴퓨터에서는 보정 파일을 옮기거나 다시 보정합니다. USB 포트는 재연결 시 바뀔 수 있습니다.
카메라와 돌/보드는 녹화 때와 같은 구도로 준비합니다.

아래를 한 줄로 실행하고 `MODEL_DIR`를 압축을 푼 `pretrained_model`의 절대 경로로 바꿉니다.
`FOLLOWER_PORT`도 현재 연결된 실제 포트로 바꿉니다(기존 Mac에서는 `/dev/tty.usbserial-120`).

```sh
lerobot-rollout --strategy.type=base --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=gomoku_follower --robot.max_relative_target=1.0 --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --policy.path="MODEL_DIR" --device=mps --task="Pick up the white stone from the supply and place it at the center intersection of the board." --duration=10 --interactive=true --return_to_initial_position=false
```

MPS가 없으면 `--device=cpu`로 바꿉니다. 로딩 후 `/start`를 입력하면 실제 로봇이 움직입니다.
처음에는 짧게 확인하고 잘못 움직이면 `Ctrl+C`로 중단합니다. 이동 제한은 충돌 방지를 보장하지 않습니다.

## 참고와 검증 범위

- [LeRobot SmolVLA 공식 안내](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot 설치 안내](https://huggingface.co/docs/lerobot/installation)
- [PyTorch CUDA 배포](https://pytorch.org/get-started/previous-versions/#v291)
- [TorchCodec 호환표](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)

원본 모델 가중치는 GitHub에 포함하지 않고 Hugging Face에서 받습니다.
로컬 Mac에서 데이터 준비·18개 에피소드와 비디오 검사·실제 CPU 학습 1스텝·체크포인트 저장을 확인했습니다.
모델 내보내기 ZIP에는 설정, 가중치, 전처리/후처리와 정규화 파일을 함께 넣습니다.
RTX 5070 Ti의 CUDA 실행은 해당 컴퓨터에서 확인해야 합니다. 로컬 1스텝은 실행 점검이며 작업 성공률 검증이 아닙니다.
