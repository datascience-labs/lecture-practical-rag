---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 14회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 14회차: NodeRAG — 이종 노드 그래프 RAG

## 그래프 노드의 "종류"를 설계하다


---

![bg](generated/s14-cover.png)

---

## 오늘의 목표

- 동종(homogeneous) 그래프의 한계와 이종(heterogeneous) 그래프의 이점을 이해한다
- NodeRAG의 노드 타입 체계와 그래프 구성을 파악한다
- Personalized PageRank 기반 검색을 구현한다
- GraphRAG / OntoRAG / NodeRAG를 종합 비교하고 선택 기준을 정리한다

---

## 지금까지 그래프의 아쉬움

GraphRAG(12회차)의 그래프를 다시 보면 — 노드가 사실상 **개체 한 종류**:

```
[김철수] ── [영업1팀] ── [영업본부] ...
```

그런데 RAG 파이프라인에는 성격이 다른 정보 조각들이 공존:

- 원문 청크 (텍스트 덩어리)
- 개체 (이름 + 설명)
- 관계 (개체 쌍의 연결)
- **개별 사실/사건** ("2026년 3월, PD-2000 단종 결정") — 개체도 청크도 아님
- 커뮤니티 요약 (고수준 인사이트)

이들을 전부 같은 취급하거나(뭉개짐), 그래프 밖에 두면(연결 단절) 손해
→ **각각을 서로 다른 타입의 노드로 그래프 안에 통합하자** = NodeRAG

---

## NodeRAG 개요

> "NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes" (2025)

핵심 주장:

1. 그래프 구조 설계가 RAG 성능의 관건인데, 기존 방법은 노드 설계가 조악하다
2. 정보의 **역할별로 노드 타입을 분리**하고
3. 검색은 그래프 알고리즘(**Personalized PageRank**)으로 관련 노드를 확산 탐색
4. 최종 컨텍스트는 **검색에 적합한 타입의 노드만** 선별해 조립

효과: GraphRAG 대비 검색 정밀도·비용 효율 개선 보고 (특히 multi-hop 벤치마크)

---

## NodeRAG의 노드 타입 체계

| 노드 타입 | 내용 | 검색 대상? | 비유 |
|---|---|---|---|
| **T** (Text) | 원문 청크 전체 | ✅ (최종 근거) | 본문 |
| **S** (Semantic unit) | 청크에서 추출한 독립적 사건/사실 요약 | ✅ | 문장 카드 |
| **N** (Entity name) | 개체 이름 (가벼운 심볼) | 진입점 | 색인어 |
| **R** (Relationship) | 관계를 노드로 승격시킨 것 | 중계 | 연결 카드 |
| **A** (Attribute) | 중요 개체의 속성 요약 (LLM 생성) | ✅ | 인물 프로필 |
| **H** (High-level) | 커뮤니티 요약 인사이트 | ✅ | 챕터 요약 |
| **O** (Overview) | H의 제목/키워드 | 진입점 | 목차 |

- **진입점 노드**(N, O): 검색의 시작점 — 가볍고 매칭 전용
- **내용 노드**(T, S, A, H): 실제 컨텍스트로 투입되는 노드

---

## 왜 관계(R)를 노드로 만드나?

동종 그래프: 관계는 엣지 → 엣지는 검색·랭킹의 대상이 되기 어려움

```
동종:   [김철수] ──담당──▶ [PD-2000]           (엣지 = 부가정보)

NodeRAG: [김철수] ─── [R: "김철수는 PD-2000    ─── [PD-2000]
                       판매를 담당한다"]
                        ▲ 관계 자체가 노드 = 임베딩·검색·랭킹 가능
```

- "누가 PD-2000을 담당하지?"라는 질문이 **R 노드와 직접 매칭**될 수 있음
- 그래프 이론에서 이런 변환을 통해 관계에도 "일급 시민" 지위 부여

---

## NodeRAG 인덱싱 파이프라인

```
문서 → 청킹
  │
  ├─ T 노드: 청크 그대로
  │
  ├─ [LLM] 분해(decomposition):
  │    S 노드 (의미 단위) + N 노드 (개체명) + R 노드 (관계)
  │    S ↔ 관련 N, R과 연결 / T ↔ 파생된 S와 연결
  │
  ├─ [LLM] 중요 개체 선별 → A 노드 (속성 요약) 생성
  │
  └─ 커뮤니티 탐지 → [LLM] H 노드 (인사이트) + O 노드 (제목)
```

이후: 내용 노드들(T, S, A, H) + 진입점 임베딩 → HNSW 인덱스에 저장
(그래프 구조와 벡터 인덱스의 **이중 인덱싱**)

---

## NodeRAG 질의: 진입점 + 그래프 확산

```
질문: "PD-2000 단종의 배경은?"

① 이중 진입 (dual entry):
   - 정확 매칭: 질문에서 개체 추출 → N 노드 "PD-2000" 히트
   - 벡터 매칭: 질문 임베딩 → 유사한 S/A/H 노드 히트

② Personalized PageRank (PPR):
   진입점들에서 그래프를 따라 "관련성 확산"
   → 진입점과 가깝게 연결된 노드일수록 높은 점수

③ 필터링: 점수 상위 노드 중 내용 노드(T, S, A, H)만 선별

④ 컨텍스트 조립 → LLM 답변
```

PPR이 핵심: **명시적 multi-hop 추론 없이** 그래프 구조가 관련 정보를 모아줌

---

## Personalized PageRank 직관

일반 PageRank: "전체 그래프에서 중요한 노드" (구글 검색의 원조)
**Personalized** PageRank: "**특정 시작점들에서 볼 때** 중요한 노드"

```
랜덤 서퍼 모델:
  - 시작점(진입점 노드)에서 출발해 엣지를 따라 무작위 이동
  - 확률 α로 시작점으로 순간이동 (reset)
  - 오래 머무는 노드 = 시작점과 구조적으로 밀접한 노드
```

```python
import networkx as nx

def ppr_search(G, entry_nodes: list[str], top_k: int = 10) -> list[str]:
    personalization = {n: 1.0 for n in entry_nodes}     # 시작점에 질량 집중
    scores = nx.pagerank(G, alpha=0.85, personalization=personalization)
    ranked = sorted(scores, key=scores.get, reverse=True)
    # 내용 노드만 필터 (진입점·관계 노드는 컨텍스트에서 제외)
    return [n for n in ranked
            if G.nodes[n].get("node_type") in ("T", "S", "A", "H")][:top_k]
```

---

## 실습: 미니 NodeRAG (12회차 코드 확장)

```python
# mini_noderag.py — 분해 프롬프트
DECOMPOSE_PROMPT = """청크를 분해해 JSON으로 출력하라:
{"semantic_units": [   # 독립적으로 이해 가능한 사실/사건 단위 요약
   {"summary": ..., "entities": [언급된 개체명들]}],
 "relations": [{"s": ..., "desc": "관계 서술문", "o": ...}]}"""

def build_hetero_graph(chunks: list[str]) -> nx.Graph:
    G = nx.Graph()
    for ci, chunk in enumerate(chunks):
        t_id = f"T{ci}"
        G.add_node(t_id, node_type="T", text=chunk)
        data = llm_json(DECOMPOSE_PROMPT, chunk)          # LLM 호출 (12회차 방식)
        for si, su in enumerate(data["semantic_units"]):
            s_id = f"S{ci}-{si}"
            G.add_node(s_id, node_type="S", text=su["summary"])
            G.add_edge(t_id, s_id)                        # T ↔ S
            for ent in su["entities"]:
                G.add_node(ent, node_type="N")
                G.add_edge(s_id, ent)                     # S ↔ N
        for ri, r in enumerate(data["relations"]):
            r_id = f"R{ci}-{ri}"
            G.add_node(r_id, node_type="R", text=r["desc"])
            G.add_edge(r["s"], r_id); G.add_edge(r_id, r["o"])   # N ↔ R ↔ N
    return G
```

---

## 실습: 질의 흐름

```python
class NodeRAGSearch:
    def __init__(self, G):
        self.G = G
        self.model = SentenceTransformer("BAAI/bge-m3")
        # 내용 노드(text 보유)만 임베딩
        self.content = [n for n, d in G.nodes(data=True) if "text" in d]
        self.embs = self.model.encode([G.nodes[n]["text"] for n in self.content])

    def search(self, question: str, k: int = 8) -> str:
        # ① 이중 진입: 벡터 상위 + 질문에 등장한 개체명(N 노드)
        q = self.model.encode(question)
        sims = util.cos_sim(q, self.embs)[0]
        vector_entries = [self.content[i] for i in sims.argsort(descending=True)[:5]]
        name_entries = [n for n, d in self.G.nodes(data=True)
                        if d.get("node_type") == "N" and n in question]

        # ② PPR 확산 → ③ 내용 노드 필터
        picked = ppr_search(self.G, vector_entries + name_entries, top_k=k)

        # ④ 컨텍스트 조립
        return "\n\n".join(f"[{self.G.nodes[n]['node_type']}] "
                           f"{self.G.nodes[n]['text']}" for n in picked)
```

→ 조립된 컨텍스트를 5회차 `ask()`의 프롬프트에 투입하면 완성

---

## 3대 Graph RAG 종합 비교

| | GraphRAG (MS) | OntoRAG | NodeRAG |
|---|---|---|---|
| 핵심 기여 | 커뮤니티 요약으로 **전역 질문** 해결 | **스키마 제약**으로 정합성·추론 | **이종 노드 + PPR**로 검색 정밀도 |
| 그래프 | 동종(개체 중심) | 스키마 종속 | 이종(7종 노드) |
| 검색 | Local/Global 이원화 | 구조 질의 + 추론 | PPR 확산 (단일 메커니즘) |
| 강한 질문 | 요약·조망형 | 규칙·계층 질의 | multi-hop 세부 질문 |
| 인덱싱 비용 | 높음 | 중간 + 설계 비용 | 높음 (분해 호출) |
| 적합 도메인 | 보고서·서사 코퍼스 | 규정·법률·의료 | 복잡한 세부 사실 QA |

**실무 조합 예**: 온톨로지 제약 추출(13) + 이종 노드 구조(14) + 커뮤니티 요약(11-12)
— 세 기법은 파이프라인의 서로 다른 단계를 개선하므로 공존 가능

---

## 과제

1. 12회차 코퍼스로 미니 NodeRAG 구축, multi-hop 질문 5개에서 GraphRAG Local Search와 비교
2. PPR의 alpha(0.5 / 0.85 / 0.95) 변화가 검색 결과에 미치는 영향 관찰
3. 9회차 평가 하네스로 3대 방법 + 기본 RAG 최종 비교표 작성 → 17회차에 지참

## 다음 시간

**15회차: 오픈소스 VLM과 멀티모달 RAG** — 텍스트를 넘어 도면과 사진을 검색하고 이해한다
