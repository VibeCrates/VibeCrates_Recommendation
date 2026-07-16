# 설계 노트 — poet/추상 스타일 쿼리 정렬 개선

작성: 2026-07-16 (세션 16) · 근거: `experiments/eval_lang_20260618_report.txt`, `reports/report_session_16.txt`
관련: [진단 메모리 project-music-text-diagnosis], `docs/design_audio_feature_integration.md`

## 문제 정의
6월 언어/스타일 평가에서 poet 스타일이 최하(avg 0.86, x율 52.5%). 실패는 **추상·감각 이미지**에
집중(“A voice whispering under moonlight” 0.00, “Like the excitement of a spring day” 0.15),
성공은 **명확한 감정 키워드**에 한정(“In the name of love” 1.90).

중요: **도메인 무관 전역 문제**다. 스타일×도메인 교차표에서 poet은 all 0.85 / book 0.85 /
movie 0.90 / music 0.85로 균일하게 실패. “Action movie set in space”(영화 direct 쿼리)도
전멸. 즉 음악만의 문제가 아니라 추상/은유 쿼리 전반의 정렬 실패다.

## 근본 원인 (3겹)
1. **쿼리 인코더가 CLIP text + 완전 동결.** `src/models/recommender.py::QueryBlock`은
   CLIPTextModel을 freeze하고 MLP 헤드만 학습. CLIP text는 이미지-캡션으로 학습돼 구체·시각
   언어에 강하고 은유에 약하며, 동결이라 poet 언어에 적응할 capacity가 없다. (콘텐츠 쪽
   `TextBlock`은 SBERT+LoRA로 적응 가능 → **인코더 비대칭**.)
2. **학습 타깃이 3 페르소나 평균.** `encode_query`가 아이템당 DSV 3개(Poet/Space/Philosopher)를
   mean-pooling → 콘텐츠는 세 방향의 “타협 centroid”에 정렬. 개별 poet 쿼리는 한 번도 검색
   앵커로 학습되지 않는다. 추론 시 순수 poet 쿼리는 centroid에서 벗어나 실패.
3. **콘텐츠 텍스트에 mood 어휘 부재.** content_text는 사실/화제 서술(overview·블러브·메타·가사)
   뿐. poet 쿼리의 mood(친밀·야간·나직함)와 SBERT 공간에서 겹칠 표적이 없다.

---

## 개선안 1 — 쿼리 인코더 대칭화  · 도메인 무관
CLIP text 동결 해제 후 **LoRA 부착**(콘텐츠 TextBlock과 대칭), 또는 **SBERT 쿼리 경로 추가**
후 CLIP(구체·시각)⊕SBERT(추상·의미) 융합.
- SBERT 경로를 넣으면 콘텐츠 텍스트와 **동일 인코더 공간**이 생겨 추상 쿼리 정렬이 크게 개선.
- 최소 변경안: `QueryBlock`에 `LoraConfig(target_modules=["q","v"])` 적용, `text_encoder`
  requires_grad 해제. 비용 소폭↑.
```python
# QueryBlock.__init__ (스케치)
self.text_encoder = CLIPTextModel.from_pretrained(model_name)
self.text_encoder = get_peft_model(self.text_encoder, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
# (선택) SBERT 병렬 경로 추가 후 concat → MLP
```

---

## 개선안 2 — 페르소나 평균 제거  · 도메인 무관
mean-pooling으로 개별 스타일이 앵커가 되지 못하는 문제. 각 `(아이템, 쿼리)`를 개별 positive로.

- **방법 A (배치 확장)**: B 아이템 → 3B 쌍. 단, InfoNCE 대각선 positive 규칙상 같은 아이템의
  다른 페르소나가 false negative가 됨 → 같은-아이템 쌍 마스킹 필요.
- **방법 B (확률적 샘플링, 권장)**: 매 스텝 아이템당 쿼리 1개 랜덤 선택. epoch을 거치며 콘텐츠가
  세 페르소나를 각각 직접 앵커로 경험, false negative 없음, batch/비용 그대로.
```python
# dataset.py MultiModalDataset.__getitem__ (방법 B 스케치)
raw = self.queries[idx]
qs = [q.strip() for q in str(raw).split("|") if q.strip()]
query = [random.choice(qs)] if qs else []   # 학습: 스텝마다 1개 샘플
# 추론 경로(단일 쿼리 입력)는 기존 encode_query 그대로.
```
- `encode_query`의 mean-pooling은 추론(멀티 쿼리 입력)용으로만 남긴다.

효과: poet 쿼리가 1급 검색 키로 학습돼 “moonlight whisper” 류가 실제로 아이템을 변별.

---

## 개선안 3 — 3도메인 vibe description 통일  · 도메인별 구현
원리는 도메인 무관(“콘텐츠에 mood 어휘를 SBERT 공간에 심어 poet의 표적을 만든다”). mood 신호
출처만 도메인마다 다르다. 이는 세션 16 진단 (A) “텍스트 타입 3도메인 통일”과 동일한 작업이다.

| 도메인 | 현 content_text | mood 결핍 | mood 신호 출처 |
|---|---|---|---|
| Movie | Title/Genre/Overview/Director/Cast | overview=줄거리, 톤 어휘 적음 | overview 재서술 + 장르 + **포스터(강한 시각 mood)** |
| Book  | Title/Author/Category/Description  | 블러브=주제 위주 | 블러브 재서술 + 카테고리 + 커버(약함) |
| Music | 메타 + 가사/설명 | 사실 나열, mood 無 | 메타 + **오디오 피처** + 가사요약 + 커버 |

- movie/book용 합성기도 `scripts/generate_music_descriptions.py`와 **동일 계약**(2~3문장 3인칭
  mood/톤/내용 설명, movie overview register)으로 필요. 소스만 교체.
- **이중 이득**: `scripts/generate_queries.py::build_synopsis`가 콘텐츠 원문으로 쿼리(라벨)를
  생성하므로, mood 담은 콘텐츠 → poet 학습쌍 품질도 동반 개선(라벨 오염 D 해소).
- **함정(세분성)**: mood를 coarse bin에서만 뽑으면 비슷한 곡/작품이 동일 mood 텍스트로 뭉쳐
  변별력 붕괴. 오디오 bin/장르는 객관 앵커로만, 트랙/작품 고유 디테일을 LLM이 얹어 구별성 유지.
- **도메인 우선순위**: 음악(고유 오디오 활용+커버 약함)>영화(overview 톤 보강+포스터 경로 활용)
  >책(블러브 재서술 주 수단).

---

## 상보성 — 함께 가야 하는 이유
- 개선안 3은 콘텐츠 쪽에 poet가 매칭할 **표적(mood 어휘)** 을 만들고,
- 개선안 2는 학습이 그 표적을 **poet 방향에서 직접** 조준하게 하며,
- 개선안 1은 쿼리 인코더가 **추상 언어를 표현**하도록 capacity를 준다.

3만 하면 표적은 생겨도 학습이 centroid를 겨냥해 절반만, 2만 하면 앵커는 생겨도 매칭할 mood가
없어 헛돌고, 1만 하면 인코더는 좋아져도 타깃·표적이 그대로다. **세 개를 한 개선 사이클로 묶고**
6월 평가를 baseline 삼아 poet avg_score로 측정한다.

## 후속 액션(구현 시)
- [ ] QueryBlock LoRA/ SBERT 병렬 경로 (개선안 1)
- [ ] 학습 시 아이템당 쿼리 1개 확률 샘플, mean-pool은 추론용만 (개선안 2)
- [ ] movie/book vibe description 합성기(음악 스크립트와 동일 계약), 3도메인 description_synth
- [ ] `_build_content_text` 3도메인 모두 description_synth 우선 사용으로 통일
- [ ] 6월 eval 재실행(poet 중심) → baseline 대비 poet/atmosphere avg_score 비교
