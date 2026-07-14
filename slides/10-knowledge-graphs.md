---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 10회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 10회차: 지식 그래프 기초

## 벡터가 못 담는 "관계"를 담는 자료구조


---

![bg](generated/s10-cover.png)

---

## 오늘의 목표

- 지식 그래프(트리플, 노드, 엣지)의 구조를 이해한다
- LLM으로 텍스트에서 개체와 관계를 추출한다
- NetworkX로 그래프를 구축하고 multi-hop 탐색을 실행한다
- 벡터 검색이 실패하는 질문을 그래프로 해결해본다

---

## 9회차 실패 사례 복기: 벡터 검색의 구조적 한계

```
질문: "김철수 팀장의 상사가 결재할 수 있는 최대 금액은?"

필요한 추론 사슬:
  ① 김철수의 상사는 누구인가?        → 문서 A: "김철수는 박영희 본부장 산하..."
  ② 박영희의 직급은?                → 문서 B: "박영희 본부장은..."
  ③ 본부장의 결재 한도는?            → 문서 C: "본부장 전결 한도는 5천만 원"

벡터 검색: 질문 전체와 유사한 청크를 찾음
  → 문서 A, B, C 중 어느 것도 질문 전체와 비슷하지 않음 → 검색 실패!
```

- 8회차의 쿼리 분해로 일부 해결 가능하지만: 느리고, LLM이 분해를 잘해야 하고, 매번 반복
- **관계를 미리 구조화**해두면? → 지식 그래프

---

## 지식 그래프란

**개체(Entity)**를 노드로, **관계(Relation)**를 엣지로 갖는 그래프

기본 단위: **트리플 (주어, 술어, 목적어)**

```
(김철수, 소속, 영업1팀)
(영업1팀, 상위조직, 영업본부)
(박영희, 본부장, 영업본부)
(본부장, 결재한도, 5천만원)
```

```
[김철수] ──소속──▶ [영업1팀] ──상위조직──▶ [영업본부]
                                              ▲
[박영희] ─────────본부장───────────────────────┘
```

→ "김철수의 상사" = 그래프에서 **2-hop 탐색**으로 즉시 도달

---

## 벡터 vs 그래프: 상호보완 관계

| | 벡터 검색 | 지식 그래프 |
|---|---|---|
| 잘하는 것 | 의미적 유사성, 모호한 질문 | 정확한 관계, multi-hop, 집계 |
| 못하는 것 | 관계 추론, 전역 질문 | 유사어 매칭, 비정형 서술 |
| 구축 비용 | 낮음 (임베딩만) | 높음 (개체·관계 추출) |
| 데이터 형태 | 비정형 텍스트 그대로 | 구조화 필요 |

**결론은 대체가 아니라 결합** — 11회차 GraphRAG의 핵심 아이디어

---

## 그래프 저장소 선택지

| 도구 | 유형 | 특징 |
|---|---|---|
| **NetworkX** | 파이썬 라이브러리 | 인메모리, 학습·프로토타입 최적 (이 강의) |
| Neo4j | 그래프 DB | Cypher 쿼리, 시각화, 운영급 |
| Memgraph | 그래프 DB | 인메모리 고성능, Cypher 호환 |
| RDF 스토어 (Jena, GraphDB) | 트리플 스토어 | SPARQL, 온톨로지(13회차)와 연계 |

```python
# pip install networkx
import networkx as nx
G = nx.DiGraph()                       # 방향 그래프
G.add_edge("김철수", "영업1팀", relation="소속")
```

---

## LLM 기반 개체·관계 추출

전통적으로 NLP의 난제였지만, LLM이 판도를 바꿈:

```python
# extract_triples.py
import anthropic, json

EXTRACT_PROMPT = """텍스트에서 (주어, 관계, 목적어) 트리플을 추출하라.
규칙:
- 개체명은 원문 표기 그대로 사용 (약어 임의 확장 금지)
- 관계는 간결한 한국어 동사/명사로 (소속, 담당, 승인권자, 상위조직 등)
- JSON 배열로만 출력: [{"s": ..., "r": ..., "o": ...}, ...]"""

def extract_triples(client, text: str) -> list[dict]:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,
        system=EXTRACT_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return json.loads(response.content[0].text)
```

---

## 실습 1: 문서에서 그래프 구축

```python
# build_graph.py
import networkx as nx

documents = [
    "김철수 과장은 영업1팀 소속으로 XK-2000 제품 판매를 담당한다.",
    "영업1팀과 영업2팀은 영업본부 산하 조직이다.",
    "박영희 본부장이 영업본부를 총괄하며, 5천만 원까지 전결 가능하다.",
    "XK-2000은 산업용 전원장치로 제조1공장에서 생산된다.",
]

client = anthropic.Anthropic()
G = nx.DiGraph()

for doc_id, doc in enumerate(documents):
    for t in extract_triples(client, doc):
        G.add_edge(t["s"], t["o"], relation=t["r"], source=f"doc-{doc_id}")

print(f"노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
for u, v, d in G.edges(data=True):
    print(f"  ({u}) --[{d['relation']}]--> ({v})")
```

⚠️ 엣지에 `source`(출처 문서)를 저장 — 답변 근거 추적에 필수

---

## 실습 2: Multi-hop 탐색

```python
# graph_query.py
def find_neighbors(G, entity: str, depth: int = 2) -> list[tuple]:
    """entity에서 depth-hop 이내의 모든 관계를 수집"""
    facts = []
    visited, frontier = {entity}, [entity]
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            # 나가는 엣지와 들어오는 엣지 모두 수집
            for u, v, d in list(G.out_edges(node, data=True)) + \
                           list(G.in_edges(node, data=True)):
                facts.append((u, d["relation"], v))
                for n in (u, v):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
        frontier = next_frontier
    return facts

facts = find_neighbors(G, "김철수", depth=2)
# [('김철수', '소속', '영업1팀'), ('영업1팀', '상위조직', '영업본부'),
#  ('박영희', '총괄', '영업본부'), ...]
```

---

## 실습 3: 그래프 기반 질의응답

```python
def graph_qa(client, G, question: str, entity: str) -> str:
    facts = find_neighbors(G, entity, depth=2)
    facts_text = "\n".join(f"({s}, {r}, {o})" for s, r, o in facts)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system="주어진 지식 그래프 사실(트리플)에만 근거해 답변하라. "
               "사실들을 연결한 추론 과정을 보여라.",
        messages=[{"role": "user",
                   "content": f"<사실>\n{facts_text}\n</사실>\n\n질문: {question}"}],
    )
    return response.content[0].text

print(graph_qa(client, G,
    "김철수의 상사가 결재할 수 있는 최대 금액은?", entity="김철수"))
# → 김철수 → 영업1팀 → 영업본부 → 박영희(본부장) → 5천만 원  ✓
```

벡터 검색으로는 실패하던 질문이 **관계 탐색으로 해결됨**

---

## 남은 문제들 (다음 회차 예고)

1. **질문에서 시작 개체를 어떻게 찾나?** (위에서는 `entity="김철수"`를 수동 지정)
   → 개체 인식 + 개체명 임베딩 매칭
2. **"우리 회사 전체 조직 구조를 요약해줘"** 같은 전역 질문은?
   → 커뮤니티 탐지 + 사전 요약 = **GraphRAG(11회차)**
3. **동일 개체의 다른 표기** ("김철수" vs "김 과장" vs "철수")
   → 개체 해소(entity resolution)
4. **관계 어휘의 난립** ("소속" vs "속함" vs "재직")
   → 스키마/온톨로지 = **OntoRAG(13회차)**

---

## 과제

1. 본인 도메인 문서 10개에서 트리플 추출 → 그래프 구축 → 시각화
   (`nx.draw` 또는 pyvis 사용)
2. 추출 품질 검수: 잘못 추출된 트리플 비율과 유형 기록
3. multi-hop 질문 3개를 만들어 벡터 RAG vs 그래프 QA 비교

## 다음 시간

**11회차: GraphRAG 개념과 아키텍처** — Microsoft GraphRAG가 전역 질문을 푸는 방법
