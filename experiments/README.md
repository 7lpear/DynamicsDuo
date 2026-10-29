# 선택 실험 도구

일상적인 카메라/DINO 실행에는 이 폴더와 ultralytics 설치가 필요하지 않습니다.
실패 모델 비교와 초기 방식은 재현을 위해 남겨둡니다. 모든 명령은 저장소 루트에서 실행합니다.

```powershell
python -m pip install -r experiments/requirements.txt
python -m experiments.detect_yolo_world captures/board_20261029_123855_712442.png --rectify
python -m experiments.detect_moku captures/board_20261029_123855_712442.png --rectify
python -m experiments.read_board captures/board_20261029_123855_712442.png
python -m experiments.benchmark_detectors --backend dino
python -m experiments.benchmark_detectors --backend yolo
python -m experiments.validate_dino
python -m experiments.summarize_detector_tests
```

benchmark/validate는 포함된 특정 샘플 사진과 수동 정답을 사용한 실험입니다.
일반 대국 성능 측정 도구가 아니며, 일부 명령은 기존 실험 출력 또는 캐시된 모델을 요구합니다.
summarize_detector_tests는 benchmark 두 모델과 validate 출력이 모두 준비되어야 실행됩니다.
