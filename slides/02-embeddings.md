---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 2회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 2회차: 임베딩과 벡터 표현

## 텍스트를 숫자로 바꾸면 왜 검색이 되는가?


이미지 파일: [s02-cover.png](image-prompts.md#s02-cover-png)

---

## 오늘의 목표

- 임베딩(embedding)의 개념과 의미 공간을 이해한다
- 코사인 유사도로 텍스트 유사성을 계산할 수 있다
- 한국어에 적합한 임베딩 모델을 선택할 수 있다
- sentence-transformers로 의미 기반 검색을 직접 구현한다

---

## 임베딩이란

> 텍스트(단어, 문장, 문서)를 **고정 길이의 실수 벡터**로 변환하는 것

```
"강아지가 공원에서 뛰어놀아요"  →  [0.12, -0.85, 0.33, ..., 0.07]
                                    (예: 768차원 또는 1024차원)
```

핵심 성질: **의미가 비슷한 텍스트는 벡터 공간에서 가깝다**

```
"강아지가 공원에서 뛰어놀아요"   ●─┐
"개가 잔디밭에서 놀고 있어요"    ●─┘ 가까움
"주식 시장이 급락했습니다"          ●   멂
```

---

## 의미 공간 (Semantic Space)

- 각 차원이 어떤 "의미의 축"을 암묵적으로 표현
- 단어 임베딩의 고전적 예시:

```
vector("왕") - vector("남자") + vector("여자") ≈ vector("여왕")
```

- 현대 문장 임베딩(BERT 계열)은 **문맥**까지 반영
  - "배를 타고 갔다" vs "배가 고프다" → 다른 벡터

---

## 유사도 계산: 코사인 유사도

두 벡터 사이의 **각도**로 유사성 측정 (크기 무시)

$$\text{cosine similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

- 범위: -1 ~ 1 (1에 가까울수록 유사)
- 그 외: 유클리드 거리(L2), 내적(dot product)
- 대부분의 벡터 DB가 세 가지 모두 지원 — **임베딩 모델이 권장하는 것**을 쓰면 됨 (정규화된 벡터는 코사인 ≈ 내적)

---

## 임베딩 모델의 종류

| 구분 | 예시 | 특징 |
|---|---|---|
| API 방식 | Voyage AI, OpenAI embeddings | 관리 불필요, 호출당 과금 |
| 로컬 오픈소스 | BAAI/bge-m3, multilingual-e5 | 무료, GPU 있으면 빠름 |
| 한국어 특화 | jhgan/ko-sroberta-multitask | 한국어 성능 우수, 가벼움 |

**이 강의의 선택**: `BAAI/bge-m3`
- 다국어(한국어 포함) 성능 우수, 무료, 로컬 실행
- 최대 8192 토큰 입력, 1024차원 출력
- 참고: Anthropic은 자체 임베딩 API가 없으며 Voyage AI를 권장

---

## 임베딩 모델 선택 기준

1. **언어**: 한국어 문서라면 반드시 다국어/한국어 모델
2. **입력 길이**: 청크 크기보다 큰 max token (bge-m3: 8192)
3. **차원 수**: 클수록 표현력↑, 저장 공간·검색 속도 비용↑
4. **벤치마크**: MTEB 리더보드 (특히 Retrieval 태스크 점수)
5. **라이선스와 비용**: 상용 이용 가능 여부

⚠️ 중요: **인덱싱과 질의에 반드시 같은 모델**을 써야 함
모델을 바꾸면 전체 재인덱싱 필요

---

<!-- _class: lead -->

# 실습: 임베딩 직접 만져보기

---

## 실습 1: 첫 임베딩 생성

```python
# embed_basic.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")  # 최초 실행 시 다운로드 (~2GB)

sentences = [
    "강아지가 공원에서 뛰어놀아요",
    "개가 잔디밭에서 놀고 있어요",
    "주식 시장이 급락했습니다",
]

embeddings = model.encode(sentences)
print(embeddings.shape)   # (3, 1024) — 문장 3개, 각 1024차원
print(embeddings[0][:5])  # 첫 문장 벡터의 앞 5개 값
```

---

## 실습 2: 코사인 유사도 계산

```python
from sentence_transformers import util

# 모든 쌍의 유사도 행렬
sim = util.cos_sim(embeddings, embeddings)
print(sim)
# tensor([[1.0000, 0.8721, 0.3105],
#         [0.8721, 1.0000, 0.2987],
#         [0.3105, 0.2987, 1.0000]])
```

- 문장 1↔2 (강아지/개): **0.87** — 단어는 달라도 의미가 유사
- 문장 1↔3 (강아지/주식): **0.31** — 의미가 멂
- 키워드 매칭으로는 불가능한 **의미 기반 매칭**

---

## 실습 3: 미니 의미 검색 엔진

```python
# semantic_search.py
documents = [
    "연차 휴가는 입사일 기준으로 매년 15일 부여됩니다.",
    "법인카드 사용 내역은 매월 5일까지 제출해야 합니다.",
    "재택근무는 주 2회까지 팀장 승인 하에 가능합니다.",
    "회의실 예약은 그룹웨어에서 2주 전부터 가능합니다.",
    "경조사 지원금은 인사팀에 증빙 서류와 함께 신청합니다.",
]

doc_embeddings = model.encode(documents)

query = "집에서 일해도 되나요?"           # '재택'이라는 단어가 없음!
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, doc_embeddings)[0]
best = scores.argmax()
print(f"질문: {query}")
print(f"최적 문서: {documents[best]} (유사도 {scores[best]:.3f})")
# → "재택근무는 주 2회까지..." 가 검색됨
```

---

## 실습 3에서 확인할 것

- 질문에 "재택"이라는 단어가 **없어도** 재택근무 문서가 검색됨
- 이것이 **키워드 검색(BM25)과의 결정적 차이**
- 단, 의미 검색도 만능이 아님:
  - 고유명사, 제품 코드, 오탈자에 약함 → 6회차 하이브리드 검색에서 보완

```python
# 반례 실험: 코드명 검색
query2 = "XK-2000 모델 스펙"
# → 의미 임베딩은 "XK-2000"을 잘 구분 못할 수 있음
```

---

## 과제

1. 본인 도메인의 문서 10개로 실습 3을 재구성하고, 잘 되는 질문 / 안 되는 질문 각 3개씩 찾기
2. `jhgan/ko-sroberta-multitask` 모델로 바꿔서 같은 실험 → 결과 비교
3. (심화) MTEB 리더보드에서 한국어 retrieval 상위 모델 3개 조사

## 다음 시간

**3회차: 벡터 데이터베이스** — 문서가 100만 개면 어떻게 검색하나?
