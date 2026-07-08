---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 7회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 7회차: 검색 고도화 II — 리랭킹

## 후보는 찾았다, 이제 순서를 제대로 매기자


이미지 파일: [s07-cover.png](image-prompts.md#s07-cover-png)

---

## 오늘의 목표

- Bi-encoder와 Cross-encoder의 구조적 차이를 이해한다
- 2단계 검색(Retrieve → Rerank) 아키텍처를 구현한다
- 리랭커 도입 전후의 정확도를 비교 측정한다
- LLM 기반 리랭킹의 장단점을 파악한다

---

## 왜 리랭킹이 필요한가

벡터 검색(bi-encoder)의 구조적 한계:

```
Bi-encoder (지금까지 사용):
  질문  → [인코더] → 벡터 A ─┐
                              ├─ 코사인 유사도  ← 상호작용은 마지막 내적 한 번!
  문서  → [인코더] → 벡터 B ─┘
```

- 문서 벡터는 **질문을 모른 채** 미리 계산됨
- 질문-문서 간 세밀한 상호작용(단어 대응, 부정어 등)을 못 봄

```
질문: "연차를 이월할 수 없는 경우는?"
청크A: "연차는 다음 해로 이월할 수 있다"        ← 벡터상 매우 유사 (오답!)
청크B: "단, 퇴직 예정자는 연차 이월이 불가하다"  ← 이게 정답
```

---

## Cross-encoder: 질문과 문서를 함께 본다

```
Cross-encoder (리랭커):
  [질문 + 문서] → 하나의 인코더에 함께 입력 → 관련도 점수 (0~1)
```

| | Bi-encoder | Cross-encoder |
|---|---|---|
| 정확도 | 보통 | **높음** (토큰 수준 상호작용) |
| 속도 | 빠름 (사전 계산) | 느림 (쌍마다 추론) |
| 100만 문서 검색 | 가능 | 불가능 (100만 번 추론) |
| 용도 | 1차 검색 (recall) | 2차 재정렬 (precision) |

→ 둘을 **조합**: 빠른 검색으로 후보 20~50개 → 정확한 리랭커로 top 3~5 선별

---

## 2단계 검색 아키텍처

```
질문
 │
 ▼
1단계: Retrieval (하이브리드 검색)          ← 목표: Recall
 │    "정답이 후보 안에 들어만 있으면 됨"
 │    top 20~50 (빠르게, 넓게)
 ▼
2단계: Reranking (Cross-encoder)           ← 목표: Precision
 │    "정답을 최상위로 끌어올림"
 │    top 3~5 (느리지만 정확하게)
 ▼
LLM 프롬프트에 투입
```

**실무 효과**: 검색 정확도 개선 수단 중 **가성비 최고**로 꼽힘

---

## 리랭커 모델 선택

| 모델 | 유형 | 특징 |
|---|---|---|
| **BAAI/bge-reranker-v2-m3** | 로컬 | 다국어(한국어 OK), bge-m3와 짝, 무료 |
| Cohere Rerank | API | 성능 우수, 호출당 과금 |
| ms-marco-MiniLM | 로컬 | 가볍고 빠름, 영어 위주 |
| LLM-as-reranker | API | 유연함, 비싸고 느림 (뒤에서 다룸) |

이 강의: `bge-reranker-v2-m3` — 임베딩 모델(bge-m3)과 같은 계열

---

## 실습 1: 리랭커 기본 사용

```python
# rerank_basic.py
# pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "연차를 이월할 수 없는 경우는?"
candidates = [
    "연차는 다음 해로 이월할 수 있다.",
    "단, 퇴직 예정자는 연차 이월이 불가하다.",
    "재택근무는 주 2회까지 가능하다.",
]

pairs = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)

for doc, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
    print(f"[{s:.4f}] {doc}")
# [0.87] 단, 퇴직 예정자는 연차 이월이 불가하다.   ← 정답이 1위로!
# [0.42] 연차는 다음 해로 이월할 수 있다.
# [0.01] 재택근무는 주 2회까지 가능하다.
```

---

## 실습 2: 파이프라인에 리랭킹 통합

```python
# rag_pipeline_v2.py — 5·6회차 파이프라인 확장
class RerankedRAG(SimpleRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def retrieve(self, question: str, k: int = 4,
                 candidates: int = 20) -> list[dict]:
        # 1단계: 넓게 검색 (하이브리드 또는 벡터)
        r = self.collection.query(query_texts=[question],
                                  n_results=candidates)
        chunks = [{"text": d, "source": m["source"]}
                  for d, m in zip(r["documents"][0], r["metadatas"][0])]

        # 2단계: 리랭킹
        pairs = [(question, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        chunks.sort(key=lambda c: -c["rerank_score"])
        # 점수 임계값: 관련 없으면 과감히 버린다
        return [c for c in chunks[:k] if c["rerank_score"] > 0.1]
```

---

## 실습 3: 리랭킹 전후 비교 실험

```python
# 정답 청크 ID를 알고 있는 평가셋
eval_set = [
    {"q": "연차 이월이 안 되는 경우?",  "answer_id": "hr-017"},
    {"q": "XK-2000과 XK-3000 차이는?", "answer_id": "prod-003"},
    # ... 10개 이상
]

def hit_at_k(retrieve_fn, k: int = 3) -> float:
    hits = sum(
        1 for case in eval_set
        if case["answer_id"] in [c["id"] for c in retrieve_fn(case["q"], k=k)]
    )
    return hits / len(eval_set)

print(f"리랭킹 전 Hit@3: {hit_at_k(baseline.retrieve):.0%}")
print(f"리랭킹 후 Hit@3: {hit_at_k(reranked.retrieve):.0%}")
```

**Hit@k**: 정답 청크가 상위 k개 안에 들어간 비율 — 9회차에서 정식 평가 지표로 확장

---

## LLM 기반 리랭킹

Cross-encoder 대신 LLM에게 직접 관련도를 물을 수도 있다:

```python
def llm_rerank(client, query: str, chunks: list[str]) -> list[int]:
    listing = "\n".join(f"[{i}] {c[:200]}" for i, c in enumerate(chunks))
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        messages=[{"role": "user", "content":
            f"질문: {query}\n\n다음 문서들을 질문과의 관련도 순으로 정렬해, "
            f"인덱스만 JSON 배열로 출력해줘 (예: [2,0,1]).\n\n{listing}"}],
    )
    import json
    return json.loads(response.content[0].text)
```

- **장점**: 복잡한 조건("최신 규정 우선") 반영 가능, 별도 모델 불필요
- **단점**: 느리고 비쌈 → 후보가 적고 정확도가 극히 중요할 때만

---

## 지연시간(latency) 예산 관리

```
사용자 체감 목표: 첫 토큰까지 2초 이내 (예시)

임베딩(질문 1개)     ~50ms
벡터 검색(HNSW)      ~20ms
BM25                 ~10ms
리랭킹(20 pairs)     ~200ms (GPU) / ~1.5s (CPU) ← 병목!
LLM 첫 토큰          ~500ms~
```

- 리랭킹 후보 수(20 → 10)를 줄이면 선형으로 단축
- CPU 환경이면 경량 리랭커 또는 후보 축소 고려
- **원칙**: 품질 지표와 지연시간을 함께 측정하고 트레이드오프 결정

---

## 과제

1. 중간 프로젝트에 리랭킹 통합, Hit@3 전/후 비교표 작성
2. `candidates` 수(10/20/50)에 따른 정확도와 지연시간 측정
3. (심화) LLM 리랭킹 vs Cross-encoder 리랭킹 정확도 비교

## 다음 시간

**8회차: 쿼리 변환과 고급 RAG 패턴** — 검색 전에 질문 자체를 개선한다
