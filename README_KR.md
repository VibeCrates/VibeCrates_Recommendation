# VibeCrates Recommendation System

추천 시스템 리포지토리입니다. 사용자에게 개인화된 추천을 제공합니다.

## 프로젝트 구조

```
VibeCrates_Recommendation/
├── data/                    # 학습 및 테스트 데이터
│   ├── raw/                # 원본 데이터
│   └── processed/          # 전처리된 데이터
├── models/                 # 학습된 모델 파라미터
│   ├── checkpoints/        # 학습 중간 체크포인트
│   └── final/              # 최종 모델
├── src/                    # 소스 코드
│   ├── models/             # 추천 모델 구현
│   ├── data/               # 데이터 로딩 및 전처리
│   ├── training/           # 모델 학습 코드
│   └── api/                # FastAPI 서버 구현
├── scripts/                # 학습, 평가, 추론 스크립트
├── tests/                  # 유닛 테스트
├── notebooks/              # Jupyter 노트북
├── docker/                 # Docker 설정
└── requirements.txt        # Python 의존성
```

## 설치

### 1. 환경 설정

```bash
# 리포지토리 클론
git clone <repository-url>
cd VibeCrates_Recommendation

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

- `data/raw/` 디렉토리에 원본 데이터 추가
- 데이터 구조:
  - `users.csv`: 사용자 정보
  - `items.csv`: 아이템 정보
  - `interactions.csv`: 사용자-아이템 상호작용 데이터

## 사용 방법

### 모델 학습

```bash
python scripts/train.py \
    --model_type collaborative_filtering \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.001
```

### 모델 평가

```bash
python scripts/evaluate.py \
    --model_path models/final/model.pkl
```

### 추론 (단일 사용자 추천)

```bash
python scripts/inference.py \
    --user_id 123 \
    --top_k 10
```

### API 서버 실행

```bash
# 로컬 개발 서버
python -m uvicorn src.api.main:app --reload --port 8000

# 프로덕션 서버
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API 문서는 http://localhost:8000/docs 에서 확인할 수 있습니다.

### Docker로 실행

```bash
# 이미지 빌드
docker-compose -f docker/docker-compose.yml build

# 서비스 시작
docker-compose -f docker/docker-compose.yml up

# 서비스 중지
docker-compose -f docker/docker-compose.yml down
```

## API 엔드포인트

### Health Check
```
GET /api/v1/health
```

### 추천 생성
```
POST /api/v1/recommend
Content-Type: application/json

{
    "user_id": 123,
    "num_recommendations": 10,
    "filters": {}
}
```

또는

```
GET /api/v1/recommend/{user_id}?top_k=10
```

### 사용자 프로필 조회
```
GET /api/v1/user-profile/{user_id}
```

### 아이템 정보 조회
```
GET /api/v1/item-info/{item_id}
```

### 피드백 제출
```
POST /api/v1/feedback?user_id=123&item_id=456&rating=4.5
```

## 주요 파일 설명

### 모델 구현
- `src/models/base.py`: 기본 추천 모델 추상 클래스
- `src/models/recommender.py`: 다양한 추천 모델 구현
- `src/models/utils.py`: 모델 유틸리티 함수

### 데이터 처리
- `src/data/loader.py`: 데이터 로딩
- `src/data/preprocessing.py`: 데이터 전처리 및 특성 공학

### 학습
- `src/training/trainer.py`: 모델 학습 루프
- `src/training/config.py`: 학습 설정 및 하이퍼파라미터

### API
- `src/api/main.py`: FastAPI 앱 메인
- `src/api/routes.py`: API 엔드포인트
- `src/api/schemas.py`: 요청/응답 스키마
- `src/api/dependencies.py`: 의존성 주입 (모델 로딩 등)

## 테스트

```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 파일 실행
pytest tests/test_models.py

# 상세 출력
pytest tests/ -v
```

## 개발 가이드

### 새로운 모델 추가

1. `src/models/recommender.py`에서 `BaseRecommender` 상속
2. 필수 메서드 구현: `fit()`, `predict()`, `evaluate()`
3. `scripts/train.py`에서 모델 설정

### API 엔드포인트 추가

1. `src/api/schemas.py`에서 요청/응답 스키마 정의
2. `src/api/routes.py`에서 라우트 함수 구현
3. `tests/test_api.py`에서 테스트 작성

### 전처리 로직 추가

1. `src/data/preprocessing.py`의 `DataPreprocessor` 클래스에 메서드 추가
2. `scripts/train.py`에서 전처리 적용

## 성능 최적화

- 모델 캐싱: 서버 시작 시 모델 로드 (핫 스타트)
- 배치 처리: 대량의 예측 요청에 대해 배치 처리
- 모니터링: 예측 지연시간 및 정확도 모니터링

## 알려진 이슈

TODO: 추가될 예정

## 라이선스

MIT License

## 연락처

email address: maxsky9018@gmail.com
