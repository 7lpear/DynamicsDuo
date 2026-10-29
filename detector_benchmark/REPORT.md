# 실제 사진 기반 탐지 비교

환경: Conda so-arm-101, CPU. 대상: 흰 출력물 위 원통형 백돌 3개.

## 실험 방법

- 수동 표시한 백돌 3개 박스와 IoU >= 0.5이면 정탐으로 판정.
- 프롬프트 6개 × 보정/자동대비 2개 × 전체/고정 겹침 타일 2개 비교.
- 신뢰도 기준 7개(0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6).
- DINO 텍스트 기준 2개(0.15, 0.25), YOLO 입력 크기 960.
- 같은 모델 출력에서 여러 신뢰도 기준을 평가했으며 각 기준마다 재추론한 것은 아님.
- YOLO 평가 조합 168개, DINO 평가 조합 336개.
- 합성한 새 장면이 아니라 실제 사진에서 자르기/밝기/대비/노이즈로 파생 입력을 생성.

## 결과

YOLO-World 최상: 정탐 1/3, 오탐 0, 누락 2.
DINO는 bottle cap 프롬프트로 실제 백돌 3개를 탐지.
빈 판에는 판 전체와 위쪽 가장자리 박스가 오탐되므로 돌 크기 필터를 함께 사용.
크기 필터: 보정 좌표에서 가로/세로 25~120px, 종횡비 0.5~2.
격자 간격은 100px이며 실제 돌 지름은 약 65~80px. 이 기하 조건은 학습 모델과 별도 규칙.
중복 박스는 IoU 0.45 기준으로 억제.

## 선택 설정의 변형 이미지 검사

| 입력 | 변형 | 정탐 | 오탐 | 누락 |
|---|---|---:|---:|---:|
| stones | original | 3 | 0 | 0 |
| stones | dim_075 | 3 | 0 | 0 |
| stones | bright_125 | 3 | 0 | 0 |
| stones | contrast_070 | 3 | 0 | 0 |
| stones | noise_sigma3 | 3 | 0 | 0 |
| empty | original | 0 | 0 | 0 |
| empty | dim_075 | 0 | 0 | 0 |
| empty | bright_125 | 0 | 0 | 0 |
| empty | contrast_070 | 0 | 0 | 0 |
| empty | noise_sigma3 | 0 | 0 | 0 |

## 실행

```powershell
python detect_grounding_dino.py captures/board_20261029_123855_712442.png --rectify --prompt "bottle cap." --threshold 0.25 --text-threshold 0.15 --stone-filter
```

## 한계

같은 돌 배치 한 장에 맞춰 선택한 설정이며 새 배치에서의 독립 평가가 아니다.
밝기 변형도 실제 조명 방향 변화나 손/로봇 그림자를 대신하지 않는다.
흑돌 분류, 가려짐, 실시간 대국은 검증하지 않았다. bottle cap 라벨은 흑백 분류 결과가 아니다.
실제 대국 도입 전 다른 위치/개수의 사진과 손이 들어온 사진으로 추가 검증해야 한다.
CPU 한 장 추론은 수 초이며 모델 로딩/전처리 시간은 별도다. 영상 30fps 처리가 확인된 것은 아니다.

## 재현

python -m experiments.benchmark_detectors --backend yolo 또는 --backend dino: 전체 비교.
python -m experiments.validate_dino: 파생 이미지와 원시 검증 출력 생성.
python -m experiments.summarize_detector_tests: 저장된 출력에 크기 필터 적용 및 보고서 생성.
