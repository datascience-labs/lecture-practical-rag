---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 12회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 12회차: GraphRAG II — 실습

## 나만의 GraphRAG 파이프라인 완성하기


이미지 파일: [s12-cover.png](image-prompts.md#s12-cover-png)

---

## 오늘의 목표

- 11회차 개념을 코드로: 추출 → 그래프 → 커뮤니티 → 리포트 → 질의
- Local Search와 Global Search를 직접 구현한다
- Microsoft GraphRAG 라이브러리의 구조를 파악한다
- 기본 RAG와 GraphRAG의 결과를 나란히 비교한다

---

## 오늘의 전략: 직접 구현 + 라이브러리 병행

- **전반부**: 미니 GraphRAG를 **직접 구현** (~150줄)
  - 모든 단계를 이해하고 커스터마이즈할 수 있게
- **후반부**: Microsoft `graphrag` 라이브러리 소개
  - 실무에서 바로 쓸 수 있는 완성품

> 직접 구현을 먼저 하는 이유: 라이브러리는 블랙박스가 되기 쉽다.
> 원리를 알아야 실패했을 때 어디를 고칠지 안다. (5회차 디버깅 트리와 같은 철학)

---

## Step 1: 개체·관계 추출 (풍부한 버전)

```python
# mini_graphrag.py
import anthropic, json
import networkx as nx

EXTRACT_PROMPT = """텍스트에서 개체와 관계를 추출해 JSON으로 출력하라.
{"entities": [{"name": ..., "type": "PERSON|ORG|PRODUCT|CONCEPT",
               "description": "개체에 대한 1-2문장 설명"}],
 "relations": [{"source": ..., "target": ...,
                "description": "관계에 대한 1문장 설명"}]}"""

client = anthropic.Anthropic()

def extract(text: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-8", max_tokens=4096,
        system=EXTRACT_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return json.loads(response.content[0].text)
```

---

## Step 2: 그래프 구축 + 개체 설명 병합

```python
def build_graph(chunks: list[str]) -> nx.Graph:
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        data = extract(chunk)
        for e in data["entities"]:
            if G.has_node(e["name"]):
                # 이미 있는 개체 → 설명을 누적 (나중에 LLM으로 통합 가능)
                G.nodes[e["name"]]["description"] += " " + e["description"]
            else:
                G.add_node(e["name"], type=e["type"],
                           description=e["description"], chunks=[i])
            G.nodes[e["name"]].setdefault("chunks", []).append(i)
        for r in data["relations"]:
            G.add_edge(r["source"], r["target"],
                       description=r["description"], chunk=i)
    return G
```

⚠️ 실무 포인트: "김철수"와 "김 과장" 같은 표기 변형은 여기서 병합 안 됨
→ 개체명 임베딩 유사도 + LLM 판정으로 별도 해소 단계 필요 (과제)

---

## Step 3: 커뮤니티 탐지와 리포트 생성

```python
import networkx.algorithms.community as nx_comm

def build_community_reports(G: nx.Graph) -> list[dict]:
    communities = nx_comm.louvain_communities(G, seed=42)
    reports = []
    for i, comm in enumerate(communities):
        if len(comm) < 2:
            continue
        # 커뮤니티 내부의 개체 설명 + 관계 설명 수집
        entities = "\n".join(f"- {n}: {G.nodes[n]['description']}" for n in comm)
        edges = "\n".join(f"- {u} ↔ {v}: {d['description']}"
                          for u, v, d in G.edges(comm, data=True)
                          if u in comm and v in comm)
        response = client.messages.create(
            model="claude-opus-4-8", max_tokens=1024,
            system="개체와 관계 정보로 이 주제 군집의 제목과 종합 요약을 작성하라. "
                   'JSON 출력: {"title": ..., "summary": ...}',
            messages=[{"role": "user",
                       "content": f"<개체>\n{entities}\n</개체>\n<관계>\n{edges}\n</관계>"}],
        )
        report = json.loads(response.content[0].text)
        reports.append({**report, "community_id": i, "members": list(comm)})
    return reports
```

---

## Step 4: Local Search 구현

```python
from sentence_transformers import SentenceTransformer, util

class LocalSearch:
    def __init__(self, G, chunks, embed_model="BAAI/bge-m3"):
        self.G, self.chunks = G, chunks
        self.model = SentenceTransformer(embed_model)
        self.names = list(G.nodes)
        # 개체 이름+설명을 임베딩 → 질문과 매칭할 인덱스
        self.entity_embs = self.model.encode(
            [f"{n}: {G.nodes[n]['description']}" for n in self.names])

    def search(self, question: str, k_entities: int = 3) -> str:
        q_emb = self.model.encode(question)
        scores = util.cos_sim(q_emb, self.entity_embs)[0]
        top = [self.names[i] for i in scores.argsort(descending=True)[:k_entities]]

        context = []
        for name in top:                          # 개체 설명 + 이웃 관계 + 원문 청크
            context.append(f"[개체] {name}: {self.G.nodes[name]['description']}")
            for u, v, d in self.G.edges(name, data=True):
                context.append(f"[관계] {u} ↔ {v}: {d['description']}")
            for ci in self.G.nodes[name].get("chunks", [])[:2]:
                context.append(f"[원문] {self.chunks[ci]}")
        return "\n".join(dict.fromkeys(context))   # 중복 제거
```

---

## Step 5: Global Search 구현 (Map-Reduce)

```python
class GlobalSearch:
    def __init__(self, reports: list[dict]):
        self.reports = reports

    def search(self, question: str) -> str:
        # Map: 각 커뮤니티 리포트에서 부분 답변 + 관련도 점수
        partials = []
        for r in self.reports:
            resp = client.messages.create(
                model="claude-opus-4-8", max_tokens=512,
                system='리포트에서 질문과 관련된 내용을 추출하고 관련도(0~100)를 매겨라. '
                       '관련 없으면 score를 0으로. JSON: {"score": ..., "answer": ...}',
                messages=[{"role": "user",
                           "content": f"<리포트: {r['title']}>\n{r['summary']}\n"
                                      f"</리포트>\n\n질문: {question}"}],
            )
            p = json.loads(resp.content[0].text)
            if p["score"] > 20:
                partials.append(p)

        # Reduce: 부분 답변들을 종합
        partials.sort(key=lambda p: -p["score"])
        merged = "\n\n".join(p["answer"] for p in partials[:5])
        final = client.messages.create(
            model="claude-opus-4-8", max_tokens=2048,
            system="부분 답변들을 종합해 완결된 최종 답변을 작성하라.",
            messages=[{"role": "user", "content": f"질문: {question}\n\n{merged}"}],
        )
        return final.content[0].text
```

---

## Step 6: 통합 실행과 비교

```python
chunks = load_and_chunk("my_corpus/")            # 4회차 청킹 재사용

G = build_graph(chunks)                          # 인덱싱 (시간·비용 소요!)
reports = build_community_reports(G)
local = LocalSearch(G, chunks)
global_ = GlobalSearch(reports)

# 비교 실험
questions = {
    "local":  "김철수 과장이 담당하는 제품의 생산 공장은?",   # multi-hop
    "global": "이 회사의 사업 구조를 전체적으로 요약해줘",     # 전역
}

# 기본 RAG (5회차) vs GraphRAG를 같은 질문으로
print(basic_rag.ask(questions["global"]))        # 산발적 청크 나열
print(global_.search(questions["global"]))       # 구조화된 전체 요약
```

---

## Microsoft GraphRAG 라이브러리

직접 구현의 개념이 그대로 대응됨:

```bash
pip install graphrag
mkdir -p ./ragproj/input && cp docs/*.txt ./ragproj/input/
graphrag init --root ./ragproj      # settings.yaml, .env 생성
# .env에 API 키, settings.yaml에 모델 설정
graphrag index --root ./ragproj     # 인덱싱 (추출→그래프→커뮤니티→리포트)

graphrag query --root ./ragproj --method local  "김철수가 담당하는 제품은?"
graphrag query --root ./ragproj --method global "전체 주제를 요약해줘"
```

| settings.yaml 주요 항목 | 대응 개념 |
|---|---|
| `entity_extraction.prompt` | Step 1의 추출 프롬프트 (도메인 맞춤 필수!) |
| `community_reports` | Step 3의 리포트 생성 |
| `local_search` / `global_search` | Step 4 / Step 5 |

그 외 구현체: LlamaIndex `PropertyGraphIndex`, Neo4j GraphRAG 패키지, LightRAG 등

---

## 인덱싱 비용 관리 실전 팁

- **작게 시작**: 문서 10~50개로 파이프라인 검증 후 확장
- **추출 모델 분리**: 추출은 저비용 모델(Haiku), 질의 응답은 고성능 모델
  — 품질 저하 여부는 9회차 평가로 확인
- **프롬프트 튜닝이 최우선**: 기본 추출 프롬프트는 범용 → 도메인 개체 유형
  (예: "법조문", "제품코드")을 명시하면 품질 급상승
- **증분 인덱싱**: 새 문서만 추출하고 그래프에 병합 (전체 재구축 회피)
- 중간 산출물(추출 결과 JSON)을 **파일로 캐시** — 재실행 시 LLM 호출 절약

---

## 과제

1. 본인 코퍼스로 미니 GraphRAG 전체 파이프라인 실행 (추출 캐시 필수!)
2. 9회차 평가셋에 전역 질문 5개를 추가하고, 기본 RAG vs GraphRAG 평가표 갱신
3. (심화) 개체 해소: 임베딩 유사도 상위 개체쌍을 LLM에게 "같은 개체인가?" 판정시켜 병합
4. (심화) Microsoft graphrag 라이브러리로 같은 코퍼스 인덱싱 → 직접 구현과 결과 비교

## 다음 시간

**13회차: OntoRAG** — 관계 어휘의 난립을 온톨로지로 다스린다
