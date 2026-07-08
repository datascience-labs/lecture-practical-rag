---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 6회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 6회차: 검색 고도화 I — BM25와 하이브리드 검색

## 의미 검색이 놓치는 것을 키워드로 잡는다


이미지 파일: [s06-cover.png](image-prompts.md#s06-cover-png)

---

## 오늘의 목표

- 벡터 검색이 실패하는 패턴을 파악한다
- BM25 키워드 검색의 원리를 이해하고 구현한다
- RRF(Reciprocal Rank Fusion)로 두 검색을 결합한다
- 5회차 파이프라인에 하이브리드 검색을 통합한다

---

## 벡터 검색이 실패하는 경우

| 실패 패턴 | 예시 질문 | 이유 |
|---|---|---|
| 고유명사/코드 | "XK-2000 스펙 알려줘" | 임베딩 공간에서 코드명은 변별력 낮음 |
| 정확한 용어 | "제23조 2항 내용은?" | "제23조"와 "제24조"가 벡터상 거의 동일 |
| 신조어/약어 | "TF트리거 절차" | 학습 데이터에 없는 사내 용어 |
| 짧은 질문 | "환불" | 벡터가 정보 부족으로 모호 |

→ 이런 경우는 오히려 **전통적 키워드 검색**이 정확함

---

## BM25: 키워드 검색의 표준

TF-IDF의 개선판 — 문서 검색 엔진(Elasticsearch 등)의 기본 알고리즘

$$\text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

직관적으로 3가지 요소:

1. **TF** — 질문의 단어가 문서에 많이 나올수록 점수↑ (단, 포화됨: 10번이나 100번이나 비슷)
2. **IDF** — 흔한 단어("있다", "하다")는 가중치↓, 희귀 단어("XK-2000")는 가중치↑
3. **문서 길이 정규화** — 긴 문서가 우연히 매칭될 확률 보정

---

## 한국어 BM25: 토큰화가 관건

영어는 공백 분리로 충분하지만 한국어는 조사·어미 때문에 **형태소 분석** 필요

```
"재택근무를 신청하려면"
  공백 분리:   ["재택근무를", "신청하려면"]        ← "재택근무" 불일치!
  형태소 분석: ["재택근무", "를", "신청", "하", "려면"]  ← 매칭 가능
```

```python
# kiwipiepy: 빠르고 정확한 한국어 형태소 분석기
# pip install kiwipiepy rank_bm25
from kiwipiepy import Kiwi

kiwi = Kiwi()

def tokenize(text: str) -> list[str]:
    # 명사, 동사, 형용사, 외국어(영문코드) 등 내용어만 추출
    return [t.form for t in kiwi.tokenize(text)
            if t.tag[0] in ("N", "V", "S") or t.tag == "SL"]
```

---

## 실습 1: BM25 검색 구현

```python
# bm25_search.py
from rank_bm25 import BM25Okapi

documents = [
    "XK-2000 모델의 최대 출력은 500W이며 무게는 2.3kg입니다.",
    "재택근무는 주 2회까지 팀장 승인 하에 가능합니다.",
    "XK-3000은 XK-2000의 후속 모델로 출력이 30% 향상되었습니다.",
    "연차 휴가는 입사일 기준으로 매년 15일 부여됩니다.",
]

tokenized = [tokenize(d) for d in documents]
bm25 = BM25Okapi(tokenized)

query = "XK-2000 스펙"
scores = bm25.get_scores(tokenize(query))
for doc, s in sorted(zip(documents, scores), key=lambda x: -x[1]):
    print(f"[{s:.2f}] {doc[:50]}")
# XK-2000이 포함된 문서가 정확히 상위에 랭크됨
```

같은 질문을 벡터 검색으로 해보고 순위를 비교해보자!

---

## 하이브리드 검색: 두 세계의 결합

```
        질문
      ┌──┴──┐
      ▼     ▼
  벡터 검색  BM25 검색
  (의미)    (키워드)
      │     │
      ▼     ▼
   순위 리스트 A  순위 리스트 B
      └──┬──┘
         ▼
    RRF로 융합 (Rank Fusion)
         ▼
     최종 top-k
```

점수 체계가 다른 두 검색을 어떻게 합칠까? → **점수가 아닌 순위**로 융합

---

## RRF (Reciprocal Rank Fusion)

각 검색기에서의 **순위**만 사용 — 점수 정규화 불필요, 강건함

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)} \qquad (k = 60 \text{ 관례})$$

```python
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings: 각 검색기가 반환한 문서 ID 리스트 (순위순)"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

- 두 검색 모두에서 상위인 문서 → 최상위
- 한쪽에서만 1위인 문서 → 중상위 (안전망)

---

## 실습 2: 하이브리드 검색기 클래스

```python
# hybrid_search.py
class HybridRetriever:
    def __init__(self, collection, documents: dict[str, str]):
        self.collection = collection            # ChromaDB (벡터)
        self.doc_ids = list(documents.keys())
        self.doc_texts = documents
        self.bm25 = BM25Okapi([tokenize(t) for t in documents.values()])

    def search(self, query: str, k: int = 4) -> list[dict]:
        # 1) 벡터 검색 — 넉넉히 뽑는다
        vec = self.collection.query(query_texts=[query], n_results=k * 3)
        vec_ranking = vec["ids"][0]

        # 2) BM25 검색
        scores = self.bm25.get_scores(tokenize(query))
        bm25_ranking = [self.doc_ids[i]
                        for i in sorted(range(len(scores)),
                                        key=lambda i: -scores[i])[:k * 3]]

        # 3) RRF 융합
        fused = rrf_fuse([vec_ranking, bm25_ranking])[:k]
        return [{"id": i, "text": self.doc_texts[i]} for i in fused]
```

---

## 실습 3: 벡터 vs BM25 vs 하이브리드 비교

```python
test_cases = [
    ("XK-2000 스펙 알려줘",       "키워드형 질문"),
    ("집에서 일해도 되나요?",       "의미형 질문"),
    ("XK 시리즈 신형은 뭐가 좋아?", "혼합형 질문"),
]

for query, qtype in test_cases:
    print(f"\n=== [{qtype}] {query}")
    print("  벡터:",   vector_only(query, k=1))
    print("  BM25:",   bm25_only(query, k=1))
    print("  하이브리드:", hybrid.search(query, k=1)[0]["text"][:50])
```

**기대 결과**:
- 키워드형 → BM25 승, 의미형 → 벡터 승
- **하이브리드는 양쪽 모두에서 준수** ← 이것이 핵심 가치

---

## 실무 참고

- Qdrant, Weaviate, Milvus 등은 하이브리드 검색을 **내장 지원**
- Elasticsearch/OpenSearch: BM25 + kNN 벡터 검색 결합 가능
- ChromaDB는 미지원 → 오늘처럼 직접 구현 (원리를 알면 어디서든 응용 가능)
- 가중치 조정: RRF 대신 `α·벡터점수 + (1-α)·BM25점수`(정규화 후)를 쓰는 방식도 있음
  - α 튜닝 필요 — 데이터셋별 평가(9회차)로 결정

---

## 과제

1. 5회차 중간 프로젝트에 `HybridRetriever` 통합
2. 5회차에서 "검색 실패"로 분류한 질문들을 하이브리드로 재실험 → 개선율 기록
3. (심화) RRF의 k값(10, 60, 200)에 따른 결과 변화 관찰

## 다음 시간

**7회차: 리랭킹** — 후보는 찾았다, 이제 순서를 제대로 매기자
