# 설계 노트 — 오디오 피처의 모델 투입

작성: 2026-07-16 (세션 16) · 관련 진단: `reports/report_session_16.txt`

## 배경
`src/data/preprocessing.py::create_item_features`가 Spotify 오디오 피처
(danceability/energy/loudness/speechiness/acousticness/instrumentalness/liveness/valence/tempo, 9개)를
정규화하도록 정의돼 있으나 **모델·트레이너 어디서도 호출되지 않는다**(grep 0건). 음악에서 가장
객관적·고유한 신호가 통째로 버려지는 상태. 이를 실제로 투입하는 두 경로를 설계한다.

---

## 경로 A — 텍스트 경유 (암묵적)  · **PoC로 이미 구현됨**

`scripts/generate_music_descriptions.py::verbalize_audio`가 오디오 피처를 자연어구
("high-energy and intense, upbeat and positive, ~120 BPM, minor key")로 바꿔 합성 description에
녹인다. 이 description이 `content_text`가 되어 SBERT로 들어가므로, 오디오 신호가 텍스트 경로로
간접 투입된다.

- 장점: **아키텍처 변경 0**, 3도메인 fusion 구조 그대로. 텍스트 타입 통일과 동시 달성.
- 한계: 연속값이 자연어 bin(3단계 등) + SBERT 토크나이즈를 거치며 정밀도 손실. 신호가 "soft"해짐.
- 커버리지 검증: 전체 40,036 트랙 100%가 오디오 프로필 생성(트랙당 평균 5.9개 서술구), 기존 무텍스트
  3,686건 전량 포함.

경로 A는 즉시 효과(타입 통일 + instrumental 커버)를 주지만 오디오 정밀도는 경로 B가 보완한다.
**둘은 배타적이지 않고 병행 권장**.

---

## 경로 B — 명시적 numeric 브랜치  · 아키텍처 변경안

오디오 벡터를 SBERT/CLIP과 나란한 제3 모달리티로 dual-encoder에 직접 넣는다.

### B-1. 표준 스키마에 컬럼 추가
`prepare_domain_df` 출력에 두 컬럼 추가:
- `audio_features`: 길이 D(=9) float 리스트(JSON 문자열). movie/book은 영벡터.
- `has_audio`: 0/1 presence flag. movie/book·오디오결측 트랙은 0.

```python
# preprocessing.py prepare_domain_df 내부
audio_vec = _extract_audio_vector(domain, row)   # (D,) or None
records.append({
    ...,
    "audio_features": json.dumps(audio_vec if audio_vec is not None else [0.0]*AUDIO_DIM),
    "has_audio": int(audio_vec is not None),
})
```

정규화 주의:
- `create_item_features`는 StandardScaler를 **매 호출 새로 fit** 한다 → train/val/test·추론 간
  분포가 어긋난다. **scaler를 train 음악 행에만 fit 후 저장**(`indexes/audio_scaler.pkl`)하고
  이후 transform만. 배포·추론에서 동일 scaler 로드.
- **결측 sentinel 버그**: 0~1 피처의 `-1` 결측값(실측 1행)과 tempo/mode NaN(4,092행)이 섞여 있다.
  `-1`은 NaN이 아니라 `np.nanmean`이 못 거른다 → **`<0` 값을 먼저 NaN으로 치환한 뒤 nanmean 대치**.
  loudness는 dB라 음수가 정상이므로 0~1 피처에만 적용.

### B-2. Dataset / collate 확장
```python
# dataset.py MultiModalDataset.__init__ 인자에 audio_features(List[List[float]]), has_audio(List[int]) 추가
# __getitem__ 반환에 추가:
"audio_features": torch.tensor(self.audio_features[idx], dtype=torch.float32),  # (D,)
"has_audio": torch.tensor(self.has_audio[idx], dtype=torch.float32),            # scalar

# collate_fn:
"audio_features": torch.stack([s["audio_features"] for s in batch]),  # (B, D)
"has_audio": torch.stack([s["has_audio"] for s in batch]),            # (B,)
```
loader.py의 `required` 컬럼 목록과 `MultiModalDataset(...)` 생성 호출부 2곳(get_dataloaders_from_df,
load_data_from_csv)도 새 컬럼 전달하도록 수정.

### B-3. 모델 — AudioBlock 신설 + ContentBlock 3-모달 융합
```python
class AudioBlock(nn.Module):
    """오디오 피처 벡터 → z_audio(768). presence flag로 게이팅해 무오디오 도메인엔 0 기여."""
    def __init__(self, input_dim: int = 9, output_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )
    def forward(self, x, has_audio):            # x:(B,D)  has_audio:(B,)
        z = self.net(x)
        return z * has_audio.unsqueeze(-1)      # 무오디오→영벡터(상수 bias 주입 방지)

# ContentBlock: 입력 1536→2304 (z_text ⊕ z_image ⊕ z_audio)
class ContentBlock(nn.Module):
    def __init__(self, input_dim: int = 2304, output_dim: int = 768): ...
    def forward(self, z_image, z_text, z_audio):
        combined = torch.cat((z_image, z_text, z_audio), dim=1)
        ...
```
게이팅이 핵심: movie/book은 `has_audio=0`이라 z_audio=0 → 기존 동작 보존, 회귀 없음.

```python
# DualEncoderModel
self.audio_block = AudioBlock()
def encode_content(self, text_list, images, audio, has_audio):
    z_text  = self.text_block(text_list)
    z_image = self.image_block(images)
    z_audio = self.audio_block(audio, has_audio)
    z_content = self.content_block(z_image, z_text, z_audio)
    return z_content, z_text, z_image, z_audio
def forward(self, batch):
    z_content, z_text, z_image, z_audio = self.encode_content(
        batch['content_text'], batch['content_image'],
        batch['audio_features'], batch['has_audio'])
    z_query = self.encode_query(batch['query'])
    return {"z_content":z_content,"z_text":z_text,"z_image":z_image,"z_audio":z_audio,"z_query":z_query}
```

### B-4. 손실
- Stage1(InfoNCE text↔query, text↔image, image↔query)은 **그대로 유지**.
- 오디오는 Stage2의 z_content(→ z_query KL 증류)로 흘려보내는 것이 최소 변경.
- (선택) 음악 배치 한정으로 `InfoNCE(z_audio, z_query)` 항 추가 가능. `has_audio` 마스크로
  음악 샘플만 골라 계산. 오디오→쿼리 정렬을 직접 학습시키고 싶을 때만.

---

## 트레이드오프 · 권장

| | 경로 A (텍스트) | 경로 B (numeric) |
|---|---|---|
| 아키텍처 변경 | 없음 | AudioBlock+ContentBlock+Dataset |
| 오디오 정밀도 | soft(bin+SBERT) | 연속값 보존 |
| instrumental 커버 | ○ | ○ |
| 타입 통일 부수효과 | ○(핵심) | ✗ |
| 위험 | 낮음 | 회귀 방지 위해 게이팅 필수 |

**권장 순서**: ① 경로 A(합성 description)를 먼저 적용해 타입 통일 + 커버리지 확보(즉효, 저위험) →
학습 재실행으로 baseline 개선 확인 → ② 경로 B를 얹어 오디오 정밀도 회수. B는 게이팅으로
movie/book 회귀가 없으므로 A 위에 안전하게 증분 적용 가능.

## 후속 액션(구현 시)
- [ ] `create_item_features` 결측 `-1`→NaN 치환 + scaler 저장/로드 분리
- [ ] `prepare_domain_df`에 audio_features/has_audio 방출
- [ ] Dataset/collate/loader required 컬럼 갱신
- [ ] AudioBlock 추가, ContentBlock 1536→2304, forward 배선
- [ ] 추론 스크립트(infer.py/inference.py/build_index.py)의 encode_content 호출부 동기화
