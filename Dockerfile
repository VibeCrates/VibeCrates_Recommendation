# VibeCrates 추천 API 컨테이너.
#
# 설계 의도: **모델을 이미지에 굽지 않는다.**
#   모델 2.2GB + 인덱스 586MB를 이미지에 넣으면 빌드·전송이 느려지고, 재학습할 때마다
#   이미지를 다시 만들어야 한다. 실행 시 볼륨으로 마운트한다(아래 사용법 참조).
#   마운트하지 않아도 앱은 뜨고 /api/v1/ping은 동작한다 — 통신 확인이 모델 적재와
#   분리되어야 실패 지점을 가릴 수 있기 때문이다.
FROM python:3.12-slim

WORKDIR /app

# torch는 용량이 커서(약 800MB) 레이어를 나눠 캐시가 살아 있게 한다.
# 추론만 하므로 CPU 빌드로 충분하다 — GPU가 필요한 것은 학습이고, 쿼리 1건은
# CLIP 텍스트 인코더 통과 + 인덱스와의 내적뿐이라 CPU에서 수십 ms다.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# 모델·인덱스 경로는 dependencies.py가 환경변수로 읽는다.
ENV MODEL_PATH=/app/models/trained_model.pt \
    INDEX_DIR=/app/indexes \
    IMAGE_DIR=/app/data/images \
    DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 0.0.0.0 바인딩이어야 컨테이너 밖에서 접근된다(127.0.0.1이면 컨테이너 안에서만 보인다).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
