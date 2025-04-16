# CompressLLM

CompressLLM은 LLaMA 모델을 기반으로 ARC(Abstraction and Reasoning Corpus) 문제를 해결하기 위한 프로젝트입니다. CompressARC 스타일의 다중 텐서 잠재 벡터 방식을 사용합니다.

## 주요 기능

- 다중 텐서 잠재 벡터 최적화 기반 ARC 문제 해결
- 다양한 차원의 잠재 표현을 통한 지식 압축
- 방향성 처리 모듈을 통한 공간적 패턴 인식
- LLaMA 모델을 활용한 효과적인 추론

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/CompressLLM.git
cd CompressLLM

# 필요 패키지 설치
pip install -r requirements.txt

# 데이터셋 다운로드 (ARC-AGI)
# 데이터셋을 'dataset' 디렉토리에 위치시켜야 합니다.
```

## 사용 방법

### 단일 문제 분석

```bash
# 예제 태스크 분석 (examples 폴더의 태스크 중 하나 사용)
python scripts/analyze_example.py --task-id your_task_id
```

### 모든 문제 실행

```bash
# 모든 태스크에 대해 실행
python scripts/run_all.py --split training --output-dir results/ --use-multi-latent
```

### 다중 GPU 병렬 실행

```bash
# 병렬 처리 방식으로 실행
python scripts/run_all.py --split training --output-dir results/ --parallel --gpu-ids 0,1,2,3 --use-multi-latent
```

## 프로젝트 구조

- `models/`: 모델 구현 (LlamaARCSolver, 다중 텐서 잠재 벡터 등)
- `utils/`: 유틸리티 함수 (그리드 처리, 텐서 연산, 시각화 등)
- `training/`: 훈련 관련 모듈
- `evaluation/`: 평가 관련 모듈
- `scripts/`: 실행 스크립트

## 확장 기능

이 구현은 추가 확장 가능합니다:
- `models/multitensor_latent_dir/`: 다중 텐서 잠재 벡터 확장 모듈
- `utils/multitensor_operations_dir/`: 텐서 연산 확장 모듈

## 주의사항

- 이 코드는 ARC-AGI 데이터셋을 사용합니다. 데이터셋을 'dataset' 디렉토리에 위치시켜야 합니다.
- LLaMA 모델을 사용하기 때문에 적절한 권한과 모델 접근 설정이 필요합니다.

## 라이센스

MIT
