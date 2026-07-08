---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 3회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 3회차: 벡터 데이터베이스

## 100만 개 문서에서 0.1초 만에 검색하기


이미지 파일: [s03-cover.png](image-prompts.md#s03-cover-png)

---

## 오늘의 목표

- 벡터 DB가 왜 필요한지, 어떻게 빠른지(ANN) 이해한다
- 주요 벡터 DB를 비교하고 선택 기준을 세운다
- ChromaDB로 컬렉션 생성 → 저장 → 검색 → 필터링을 실습한다
- 메타데이터 필터링으로 검색 품질을 높인다

---

## 왜 벡터 DB가 필요한가

2회차 방식(brute-force)의 문제:

```python
# 문서가 N개면 질문마다 N번 코사인 유사도 계산
scores = util.cos_sim(query_embedding, doc_embeddings)  # O(N)
```

| 문서 수 | brute-force 검색 시간 (대략) |
|---|---|
| 1,000 | 수 ms — 문제없음 |
| 100만 | 수백 ms ~ 수 초 |
| 1억 | 수십 초 — 서비스 불가 |

→ **근사 최근접 이웃(ANN, Approximate Nearest Neighbor)** 인덱스 필요

---

## ANN: 정확도를 조금 내주고 속도를 얻는다

**HNSW (Hierarchical Navigable Small World)** — 가장 널리 쓰이는 알고리즘

- 벡터들을 다층 그래프로 연결
- 상위 층(성긴 연결)에서 대략적 위치 → 하위 층에서 정밀 탐색
- 고속도로 → 국도 → 골목길로 목적지를 찾는 것과 유사

```
Layer 2:  ●───────────●          (빠른 이동)
Layer 1:  ●───●───●───●───●      (중간 탐색)
Layer 0:  ●─●─●─●─●─●─●─●─●─●    (전체 벡터, 정밀 탐색)
```

- 검색 복잡도: 대략 O(log N) — 1억 개도 밀리초 단위
- 트레이드오프: recall 98~99% (완벽하지 않지만 실용적으로 충분)

---

## 주요 벡터 DB 비교

| 이름 | 유형 | 특징 | 적합한 경우 |
|---|---|---|---|
| **ChromaDB** | 임베디드 | pip 설치만으로 시작, 학습·프로토타입 최적 | 이 강의, PoC |
| **FAISS** | 라이브러리 | Meta제, 인덱스 알고리즘 풍부, DB 기능 없음 | 연구, 커스텀 |
| **Qdrant** | 서버형 | Rust, 필터링 강력, 무료 셀프호스팅 | 중규모 서비스 |
| **Milvus** | 서버형 | 대규모 분산, 운영 복잡도 높음 | 대규모 |
| **pgvector** | 확장 | 기존 PostgreSQL에 추가 | RDB 병행 팀 |
| **Pinecone** | 클라우드 | 완전관리형, 과금 | 운영 부담 최소화 |

**원칙**: 프로토타입은 ChromaDB → 성능/규모 요구가 명확해지면 이전

---

## 벡터 DB의 핵심 개념

- **Collection**: RDB의 테이블에 해당. 하나의 임베딩 모델 = 하나의 컬렉션
- **Document**: 원문 텍스트 (청크)
- **Embedding**: 문서의 벡터
- **Metadata**: 출처, 날짜, 부서 등 부가 정보 → **필터링에 사용**
- **ID**: 각 항목의 고유 식별자 (업데이트/삭제에 필요)

```
Collection "company_docs"
├── id: "hr-001"
│   ├── document: "연차 휴가는 매년 15일..."
│   ├── embedding: [0.12, -0.85, ...]
│   └── metadata: {"source": "인사규정.pdf", "dept": "HR", "year": 2026}
```

---

<!-- _class: lead -->

# 실습: ChromaDB

---

## 실습 1: 컬렉션 생성과 데이터 저장

```python
# chroma_basic.py
import chromadb
from chromadb.utils import embedding_functions

# 디스크에 저장되는 클라이언트 (재시작해도 유지)
client = chromadb.PersistentClient(path="./chroma_db")

# 임베딩 함수 지정 — 컬렉션에 넣을 때 자동으로 임베딩됨
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3"
)

collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
)
```

---

## 실습 1 (계속): 문서 추가

```python
collection.add(
    ids=["hr-001", "hr-002", "fin-001", "it-001", "hr-003"],
    documents=[
        "연차 휴가는 입사일 기준으로 매년 15일 부여됩니다.",
        "재택근무는 주 2회까지 팀장 승인 하에 가능합니다.",
        "법인카드 사용 내역은 매월 5일까지 제출해야 합니다.",
        "VPN 접속 문제는 IT헬프데스크(#1234)로 문의하세요.",
        "경조사 지원금은 인사팀에 증빙 서류와 함께 신청합니다.",
    ],
    metadatas=[
        {"dept": "HR", "year": 2026},
        {"dept": "HR", "year": 2026},
        {"dept": "FIN", "year": 2025},
        {"dept": "IT", "year": 2026},
        {"dept": "HR", "year": 2025},
    ],
)
print(collection.count())  # 5
```

---

## 실습 2: 유사도 검색

```python
results = collection.query(
    query_texts=["집에서 근무하려면 어떻게 해야 하나요?"],
    n_results=2,
)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"[거리 {dist:.3f}] {doc}")

# [거리 0.31] 재택근무는 주 2회까지 팀장 승인 하에 가능합니다.
# [거리 0.62] 연차 휴가는 입사일 기준으로 매년 15일 부여됩니다.
```

⚠️ ChromaDB는 **거리(distance)**를 반환 — 작을수록 유사
(코사인 거리 = 1 - 코사인 유사도)

---

## 실습 3: 메타데이터 필터링

```python
# HR 부서의 2026년 문서에서만 검색
results = collection.query(
    query_texts=["휴가 관련 규정"],
    n_results=3,
    where={"$and": [
        {"dept": {"$eq": "HR"}},
        {"year": {"$eq": 2026}},
    ]},
)
```

실무에서 필터링이 중요한 이유:

- **권한 제어**: 사용자가 접근 가능한 문서만 검색
- **최신성**: 폐기된 구버전 규정 제외
- **정확도**: 검색 공간을 좁혀 노이즈 감소

---

## 실습 4: 업데이트와 삭제

```python
# 문서 수정 (규정 개정 시)
collection.update(
    ids=["hr-002"],
    documents=["재택근무는 주 3회까지 팀장 승인 하에 가능합니다."],
    metadatas=[{"dept": "HR", "year": 2026}],
)

# 문서 삭제 (규정 폐기 시)
collection.delete(ids=["hr-003"])

# ID로 직접 조회
item = collection.get(ids=["hr-002"])
print(item["documents"])
```

→ 실무 RAG는 "한 번 넣고 끝"이 아니라 **문서 수명주기 관리**가 필수

---

## 설계 팁

1. **컬렉션 분리 기준**: 임베딩 모델별, 도메인별로 분리
2. **ID 설계**: `{출처}-{문서ID}-{청크번호}` 형태 권장 → 재인덱싱 시 덮어쓰기 용이
3. **메타데이터는 미리 설계**: 나중에 필터 조건이 필요해지면 전체 재삽입
4. **거리 임계값(threshold)**: "관련 문서 없음" 판단에 사용
   - 예: 거리 > 0.6이면 검색 결과를 버리고 "모른다"고 답변

---

## 과제

1. 본인 도메인 문서 30개 이상으로 컬렉션 구축 (메타데이터 2종 이상 포함)
2. 필터링 있음/없음 검색 결과를 비교하는 실험 3가지
3. 거리 임계값을 바꿔가며 "관련 없음" 판정 정확도 관찰

## 다음 시간

**4회차: 문서 로딩과 청킹 전략** — 긴 문서를 어떻게 자를 것인가?
