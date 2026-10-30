# DynamicsDuo
2026-2 Hongik Challenge

SO-101 로봇 제어 실험과 C920/C920e 기반 9×9 오목판 비전 프로젝트.

## 환경과 실행

Windows PowerShell / Conda `so-arm-101` 환경을 사용합니다.

```powershell
conda activate so-arm-101
python -m pip install -r requirements-vision.txt
python live_dino.py --index 1
```

실행하면 현재 카메라 영상 위에 Grounding DINO 탐지 박스를 표시합니다.
모델은 최초 한 번 로딩하며 분석 중에도 영상은 갱신됩니다. CPU 박스 갱신은 수 초 간격입니다.
`S`: 저장, `P`: 분석 일시정지/재개, `Q`: 종료.

카메라/판 위치가 달라지면 반드시 재보정합니다.

```powershell
python camera_test.py --index 1
python calibrate_board.py captures/새로저장한빈판사진.png
```

동봉한 calibration/board.json은 샘플 사진의 보정값입니다. 다른 설치에 그대로 적용하지 마세요.

## 주요 파일

| 파일 | 기능 |
|---|---|
| camera_test.py | 카메라 미리보기와 원본 사진 저장 |
| calibrate_board.py | 네 교점을 클릭해 9×9 원근 보정 |
| live_dino.py / dino_detector.py | 실시간 영상과 비동기 DINO 탐지 |
| board_calibration.py / detection_utils.py | 공통 보정과 탐지 후처리 |
| detect_grounding_dino.py | 사진 한 장의 DINO 탐지 |
| gomoku_vision/ | OpenCV 격자 자동 검출, 실시간 돌 유무·흑백 구분 |
| experiments/ | YOLO·Moku 비교, 초기 방식, 파라미터 실험 |
| test_follower_motion.py | SO-101 관절 확인과 동작 실험 |

## 문서와 현재 진행 상태

- [OpenCV 오목판·흑백 돌 인식 사용법](gomoku_vision/README.md)
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
이용해 돌 유무·흑백을 판별합니다. 해당 폴더의 별도 가상환경에서 실행하세요.
판·카메라 이동 후에는 격자와 빈 판 기준을 다시 설정해야 합니다.
강화학습 에이전트 및 실제 로봇 착수 연결은 아직 구현하지 않았습니다.
동일 배치에서 파생한 이미지의 성공은 새 배치나 실제 대국 성능을 보증하지 않습니다.

## 샘플과 로컬 파일

빈 판과 백돌 배치 샘플 2장을 captures/에 포함했습니다. 얼굴 테스트 사진,
새 촬영 이미지, 모델 가중치, 가상환경, 실행 결과는 Git에서 제외합니다.
다운로드 모델의 이용 조건은 각 모델 제공자의 라이선스를 따릅니다.
뷰어에 포함된 외부 자료는 viewer/의 라이선스 파일을 참고하세요.

모델을 다운로드한 뒤 샘플 사진으로 확인할 수 있습니다.

```powershell
python detect_grounding_dino.py captures/board_20261029_123855_712442.png --rectify --stone-filter
```

## Git 작업

하드웨어 없이 회귀 테스트를 실행할 수 있습니다.

```powershell
python -m unittest test_vision test_follower_demo test_follower_load test_motion_viewer
```

```powershell
git pull --ff-only
git status
git add 수정한파일
git commit -m "변경 내용"
git push origin main
```
