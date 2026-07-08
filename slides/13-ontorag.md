---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 13회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 13회차: OntoRAG — 온톨로지 기반 RAG

## 관계 어휘의 난립을 스키마로 다스린다


이미지 파일: [s13-cover.png](image-prompts.md#s13-cover-png)

---

## 오늘의 목표

- 온톨로지(클래스, 속성, 제약)의 개념을 이해한다
- 자유 추출 그래프의 문제를 온톨로지로 해결하는 원리를 배운다
- 온톨로지 제약 기반 추출과 검증을 구현한다
- 온톨로지의 추론(reasoning) 능력을 RAG에 활용한다

---

## 12회차 그래프의 문제점 복기

LLM 자유 추출로 만든 그래프를 열어보면:

```
(김철수, 소속, 영업1팀)
(박영희, 재직, 영업본부)      ← "소속"과 "재직"이 다른 관계로 취급됨
(이민수, 속해있음, 개발팀)     ← 또 다른 표기
(영업1팀, 소속, XK-2000)      ← 팀이 제품에 "소속"?? (의미 오류)
(XK-2000, 담당, 김철수)       ← 방향이 거꾸로
```

문제 정리:

1. **관계 어휘 난립** — 같은 의미, 다른 표기 → 그래프 탐색이 깨짐
2. **의미적 오류** — 타입에 맞지 않는 관계
3. **방향 불일치** — 주어/목적어 뒤바뀜

→ 필요한 것: **"무엇이 어떤 관계를 가질 수 있는가"의 규격** = 온톨로지

---

## 온톨로지란

> 도메인의 개념(클래스), 속성(관계), 제약을 **형식적으로 정의한 스키마**

```
클래스 (Class):          사람, 조직, 제품, 시설
클래스 계층 (Hierarchy):  본부장 ⊂ 관리자 ⊂ 직원 ⊂ 사람
속성 (Property):         소속(사람 → 조직)
                         담당(사람 → 제품)
                         생산처(제품 → 시설)
제약 (Constraint):       "소속"의 주어(domain)는 사람, 목적어(range)는 조직
카디널리티:               한 사람의 소속 조직은 최대 1개 (예시 규칙)
```

- RDB의 스키마와 비슷하지만: **계층 + 추론 규칙**이 추가됨
- 표준 기술: RDF(트리플), RDFS/OWL(스키마 언어), SPARQL(질의어)

---

## 온톨로지의 힘: 추론 (Reasoning)

명시되지 않은 사실을 규칙으로 유도:

```
정의:  본부장 ⊂ 관리자          (클래스 계층)
       결재권보유(관리자 → 예산)  (관리자 전체에 적용되는 속성)
사실:  (박영희, 직급, 본부장)

추론:  박영희는 본부장 → 본부장은 관리자 → 관리자는 결재권 보유
       ∴ (박영희, 결재권보유, ...)   ← 명시적으로 쓴 적 없는 사실!
```

- **transitive 속성**: `상위조직`이 이행적이면
  (영업1팀→영업본부), (영업본부→본사) ⇒ (영업1팀→본사) 자동 유도
- **inverse 속성**: `담당`의 역은 `담당자` — 한쪽만 추출돼도 양방향 질의 가능

---

## OntoRAG: 온톨로지 + RAG 결합 아키텍처

```
┌── 설계 (사람 + LLM 보조) ──────────────────────┐
│ 도메인 온톨로지 정의 (클래스/속성/제약)          │
└────────────────────────┬───────────────────────┘
                         ▼
┌── 인덱싱 ──────────────────────────────────────┐
│ ① 온톨로지를 프롬프트에 넣어 "제약된 추출"       │
│ ② 추출 결과를 온톨로지로 검증 (오류 자동 필터)   │
│ ③ 정합성 있는 지식 그래프 구축                   │
└────────────────────────┬───────────────────────┘
                         ▼
┌── 질의 ────────────────────────────────────────┐
│ 질문 → 구조 질의(SPARQL/탐색) + 추론 + 벡터 검색 │
└─────────────────────────────────────────────────┘
```

GraphRAG와의 차이: 그래프가 **스키마에 종속** → 일관성·검증가능성·추론 확보

---

## Step 1: 온톨로지 정의 (경량 버전)

정식 OWL 대신, 실용적인 Python dict로 시작:

```python
# ontology.py
ONTOLOGY = {
    "classes": {
        "사람":  {"parent": None},
        "직원":  {"parent": "사람"},
        "관리자": {"parent": "직원"},
        "조직":  {"parent": None},
        "제품":  {"parent": None},
        "시설":  {"parent": None},
    },
    "properties": {
        "소속":   {"domain": "사람", "range": "조직"},
        "상위조직": {"domain": "조직", "range": "조직", "transitive": True},
        "담당":   {"domain": "직원", "range": "제품", "inverse": "담당자"},
        "생산처":  {"domain": "제품", "range": "시설"},
        "총괄":   {"domain": "관리자", "range": "조직"},
    },
}
```

**설계 팁**: 문서 샘플을 LLM에게 주고 온톨로지 초안을 뽑게 한 뒤 사람이 다듬기

---

## Step 2: 온톨로지 제약 추출

추출 프롬프트에 온톨로지를 명시 — LLM이 규격에 맞게 추출:

```python
def ontology_prompt(onto: dict) -> str:
    classes = ", ".join(onto["classes"].keys())
    props = "\n".join(
        f'- {name}: {p["domain"]} → {p["range"]}'
        for name, p in onto["properties"].items())
    return f"""텍스트에서 트리플을 추출하라. 반드시 아래 스키마만 사용하라.

허용 클래스: {classes}
허용 관계 (주어타입 → 목적어타입):
{props}

규칙:
- 위에 없는 관계명은 절대 만들지 마라. 의미가 같으면 허용 관계로 정규화하라.
  (예: "재직", "속해있음" → "소속")
- 개체마다 클래스를 지정하라.
- JSON: {{"triples": [{{"s": ..., "s_type": ..., "r": ..., "o": ..., "o_type": ...}}]}}"""
```

→ 12회차의 "재직/속해있음/소속" 난립이 **추출 단계에서 차단됨**

---

## Step 3: 추출 결과 검증 (Validation)

LLM이 규칙을 어겨도 **코드가 걸러낸다** — 이중 방어:

```python
def is_subclass(onto, cls, ancestor) -> bool:
    while cls is not None:
        if cls == ancestor:
            return True
        cls = onto["classes"].get(cls, {}).get("parent")
    return False

def validate_triple(onto: dict, t: dict) -> tuple[bool, str]:
    prop = onto["properties"].get(t["r"])
    if prop is None:
        return False, f"미정의 관계: {t['r']}"
    if not is_subclass(onto, t["s_type"], prop["domain"]):
        return False, f"주어 타입 오류: {t['s_type']} (기대: {prop['domain']})"
    if not is_subclass(onto, t["o_type"], prop["range"]):
        return False, f"목적어 타입 오류: {t['o_type']} (기대: {prop['range']})"
    return True, "OK"

# (영업1팀[조직], 소속, XK-2000[제품])
# → False, "주어 타입 오류: 조직 (기대: 사람)"   ← 12회차의 오류가 자동 검출!
```

---

## Step 4: 추론 구현 (transitive / inverse)

```python
def apply_inference(onto: dict, G: nx.DiGraph) -> nx.DiGraph:
    # transitive: (A,r,B), (B,r,C) ⇒ (A,r,C)
    for name, p in onto["properties"].items():
        if p.get("transitive"):
            edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d["relation"] == name]
            closure = nx.transitive_closure(nx.DiGraph(edges))
            for u, v in closure.edges():
                if not G.has_edge(u, v):
                    G.add_edge(u, v, relation=name, inferred=True)

        if p.get("inverse"):   # inverse: (A,r,B) ⇒ (B,r⁻¹,A)
            for u, v, d in list(G.edges(data=True)):
                if d["relation"] == name:
                    G.add_edge(v, u, relation=p["inverse"], inferred=True)
    return G

# 이제 "영업1팀은 본사 산하인가?" → 명시 안 된 사실도 그래프에 존재
```

`inferred=True` 표시 유지 — 답변 시 "추론된 사실"임을 명시 가능 (신뢰성)

---

## 정식 시맨틱 웹 스택 (실무 확장)

우리의 경량 구현 ↔ 표준 기술 대응:

| 경량 구현 (오늘) | 표준 스택 |
|---|---|
| Python dict 온톨로지 | OWL / RDFS (Protégé로 편집) |
| NetworkX 그래프 | RDF 트리플 스토어 (GraphDB, Jena) |
| validate_triple() | SHACL (제약 검증 표준) |
| apply_inference() | OWL Reasoner (HermiT, Pellet) |
| 그래프 탐색 | SPARQL 질의 |

```python
# rdflib 맛보기 (pip install rdflib)
from rdflib import Graph
g = Graph().parse("company.ttl")
rows = g.query("""SELECT ?person WHERE {
    ?person :소속 ?org . ?org :상위조직* :본사 . }""")   # *= transitive 경로!
```

의료·법률·금융처럼 **정합성이 중요한 도메인**은 표준 스택 채택 가치가 큼

---

## OntoRAG 도입 판단

| 신호 | 판단 |
|---|---|
| 추출 그래프의 관계 어휘가 수십 종으로 난립 | 도입 고려 ↑ |
| 도메인에 명확한 규칙 존재 (직급, 법조문, 제품 계층) | 도입 고려 ↑↑ |
| "X의 모든 하위 것들" 류의 이행적 질의가 많음 | 도입 고려 ↑↑ |
| 도메인이 자유롭고 개념이 유동적 (뉴스, 소셜) | 무리한 스키마는 역효과 |
| 온톨로지 설계·유지 인력 부재 | 경량 버전부터 |

**GraphRAG vs OntoRAG는 배타적이지 않다**: GraphRAG의 추출 단계에 온톨로지 제약을 넣는 하이브리드가 실무 스위트스팟

---

## 과제

1. 본인 도메인의 온톨로지 설계 (클래스 5개+, 속성 5개+, LLM 초안 → 수동 개선)
2. 12회차 코퍼스를 온톨로지 제약으로 재추출 → 검증 통과율과 관계 어휘 수를 자유 추출과 비교
3. transitive 속성 1개를 정의하고, 추론 전에는 답 못하던 질문이 추론 후 답해지는 사례 1개 만들기

## 다음 시간

**14회차: NodeRAG** — 그래프 노드의 이질성(heterogeneity)을 설계하다
