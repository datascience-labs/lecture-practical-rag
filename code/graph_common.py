# graph_common.py — 10~14회차 과제 공용: 슬라이드의 그래프 RAG 구현 모음
#
# 10회차 extract_triples / find_neighbors / graph_qa,
# 12회차 미니 GraphRAG(풍부한 추출·커뮤니티 리포트·Local/Global Search),
# 14회차 미니 NodeRAG(이종 노드 그래프 + PPR)를 슬라이드 코드 그대로 옮기되,
# LLM 추출 결과를 hw_cache/에 캐시한다 (12회차 "추출 캐시 필수!" 원칙).
#
# 캐시가 이미 있으면 API 키 없이도 그래프를 재구성할 수 있다.

import networkx as nx

from hw_common import llm_json, llm_text, load_cache, save_cache

# ---------------------------------------------------------------
# 10회차: 트리플 추출과 그래프 질의
# ---------------------------------------------------------------

EXTRACT_PROMPT = """텍스트에서 (주어, 관계, 목적어) 트리플을 추출하라.
규칙:
- 개체명은 원문 표기 그대로 사용 (약어 임의 확장 금지)
- 관계는 간결한 한국어 동사/명사로 (소속, 담당, 승인권자, 상위조직 등)
- JSON 배열로만 출력: [{"s": ..., "r": ..., "o": ...}, ...]"""


def extract_triples(client, text: str) -> list[dict]:
    return llm_json(client, EXTRACT_PROMPT, text, expect="array")


def build_triple_graph(client, docs: list[dict], cache_name: str) -> nx.DiGraph:
    """docs: [{"id", "text"}]. 문서별 트리플 추출(캐시) → DiGraph (10회차 실습 1)."""
    cache = load_cache(cache_name) or {}
    missing = [d for d in docs if d["id"] not in cache]
    if missing and client is None:
        raise RuntimeError(
            f"추출 캐시(hw_cache/{cache_name}.json)에 문서 {len(missing)}건이 없어 "
            "LLM 추출이 필요하다 — ANTHROPIC_API_KEY를 설정하고 다시 실행하라.")
    for d in missing:
        cache[d["id"]] = extract_triples(client, d["text"])
        save_cache(cache_name, cache)                # 문서마다 저장 — 중단돼도 재사용
        print(f"  추출: {d['id']} → 트리플 {len(cache[d['id']])}개")

    G = nx.DiGraph()
    for doc_id, triples in cache.items():
        for t in triples:
            G.add_edge(t["s"], t["o"], relation=t["r"], source=doc_id)
    return G


def find_neighbors(G, entity: str, depth: int = 2) -> list[tuple]:
    """entity에서 depth-hop 이내의 모든 관계를 수집 (10회차 실습 2)."""
    facts = []
    visited, frontier = {entity}, [entity]
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for u, v, d in list(G.out_edges(node, data=True)) + \
                           list(G.in_edges(node, data=True)):
                facts.append((u, d["relation"], v))
                for n in (u, v):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
        frontier = next_frontier
    return list(dict.fromkeys(facts))                # 중복 제거, 순서 유지


def guess_entity(G, question: str) -> str | None:
    """질문 문자열에 등장하는 노드 중 가장 긴 것을 시작 개체로 추정.

    10회차 "시작 개체를 어떻게 찾나?" 문제의 최소 해법 — 정식 해법은 12회차
    LocalSearch의 개체 임베딩 매칭이다.
    """
    hits = [n for n in G.nodes if n in question]
    return max(hits, key=len) if hits else None


def graph_qa(client, G, question: str, entity: str) -> str:
    """그래프 사실(트리플)에만 근거한 답변 생성 (10회차 실습 3)."""
    facts = find_neighbors(G, entity, depth=2)
    facts_text = "\n".join(f"({s}, {r}, {o})" for s, r, o in facts)
    return llm_text(
        client,
        "주어진 지식 그래프 사실(트리플)에만 근거해 답변하라. "
        "사실들을 연결한 추론 과정을 보여라.",
        f"<사실>\n{facts_text}\n</사실>\n\n질문: {question}", max_tokens=1024)


# ---------------------------------------------------------------
# 12회차: 미니 GraphRAG
# ---------------------------------------------------------------

RICH_EXTRACT_PROMPT = """텍스트에서 개체와 관계를 추출해 JSON으로 출력하라.
{"entities": [{"name": ..., "type": "PERSON|ORG|PRODUCT|CONCEPT",
               "description": "개체에 대한 1-2문장 설명"}],
 "relations": [{"source": ..., "target": ...,
                "description": "관계에 대한 1문장 설명"}]}"""


def build_rich_graph(client, chunks: list[str], cache_name: str) -> nx.Graph:
    """청크별 풍부한 추출(캐시) → 개체 설명이 누적된 무방향 그래프 (12회차 Step 1~2)."""
    cache = load_cache(cache_name) or {}
    missing = [i for i in range(len(chunks)) if str(i) not in cache]
    if missing and client is None:
        raise RuntimeError(
            f"추출 캐시(hw_cache/{cache_name}.json)가 없어 LLM 추출이 필요하다 — "
            "ANTHROPIC_API_KEY를 설정하고 다시 실행하라.")
    for i in missing:
        cache[str(i)] = llm_json(client, RICH_EXTRACT_PROMPT, chunks[i],
                                 max_tokens=4096)
        save_cache(cache_name, cache)
        print(f"  추출: 청크 {i} → 개체 {len(cache[str(i)]['entities'])}, "
              f"관계 {len(cache[str(i)]['relations'])}")

    G = nx.Graph()
    for i in range(len(chunks)):
        data = cache[str(i)]
        for e in data["entities"]:
            if G.has_node(e["name"]):
                G.nodes[e["name"]]["description"] += " " + e["description"]
            else:
                G.add_node(e["name"], type=e.get("type", "CONCEPT"),
                           description=e["description"])
            chunk_ids = G.nodes[e["name"]].setdefault("chunks", [])
            if i not in chunk_ids:
                chunk_ids.append(i)
        for r in data["relations"]:
            G.add_edge(r["source"], r["target"],
                       description=r["description"], chunk=i)
    for n in G.nodes:                        # 관계에만 등장한 개체도 최소 속성 보장
        G.nodes[n].setdefault("description", n)
        G.nodes[n].setdefault("chunks", [])
    return G


def build_community_reports(client, G: nx.Graph, cache_name: str) -> list[dict]:
    """Louvain 커뮤니티 탐지 → 커뮤니티별 LLM 리포트 (12회차 Step 3, 캐시)."""
    import networkx.algorithms.community as nx_comm

    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    if client is None:
        raise RuntimeError(
            f"리포트 캐시(hw_cache/{cache_name}.json)가 없어 LLM 요약이 필요하다 — "
            "ANTHROPIC_API_KEY를 설정하고 다시 실행하라.")

    communities = nx_comm.louvain_communities(G, seed=42)
    reports = []
    for i, comm in enumerate(communities):
        if len(comm) < 2:
            continue
        entities = "\n".join(f"- {n}: {G.nodes[n]['description']}" for n in comm)
        edges = "\n".join(f"- {u} ↔ {v}: {d['description']}"
                          for u, v, d in G.edges(comm, data=True)
                          if u in comm and v in comm)
        report = llm_json(
            client,
            "개체와 관계 정보로 이 주제 군집의 제목과 종합 요약을 작성하라. "
            'JSON 출력: {"title": ..., "summary": ...}',
            f"<개체>\n{entities}\n</개체>\n<관계>\n{edges}\n</관계>",
            max_tokens=1024)
        reports.append({**report, "community_id": i, "members": sorted(comm)})
        print(f"  커뮤니티 {i} 리포트: {report['title']}")
    save_cache(cache_name, reports)
    return reports


class LocalSearch:
    """개체 임베딩 매칭 → 개체 설명 + 이웃 관계 + 원문 청크 조립 (12회차 Step 4)."""

    def __init__(self, G, chunks, embed_model="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.G, self.chunks = G, chunks
        self.model = SentenceTransformer(embed_model)
        self.names = list(G.nodes)
        self.entity_embs = self.model.encode(
            [f"{n}: {G.nodes[n]['description']}" for n in self.names])

    def search(self, question: str, k_entities: int = 3) -> str:
        from sentence_transformers import util

        q_emb = self.model.encode(question)
        scores = util.cos_sim(q_emb, self.entity_embs)[0]
        top = [self.names[i] for i in scores.argsort(descending=True)[:k_entities]]

        context = []
        for name in top:
            context.append(f"[개체] {name}: {self.G.nodes[name]['description']}")
            for u, v, d in self.G.edges(name, data=True):
                context.append(f"[관계] {u} ↔ {v}: {d['description']}")
            for ci in self.G.nodes[name].get("chunks", [])[:2]:
                context.append(f"[원문] {self.chunks[ci]}")
        return "\n".join(dict.fromkeys(context))


class GlobalSearch:
    """커뮤니티 리포트 map-reduce로 전역 질문 답변 (12회차 Step 5)."""

    def __init__(self, client, reports: list[dict]):
        self.client = client
        self.reports = reports

    def search(self, question: str) -> str:
        partials = []
        for r in self.reports:                       # Map: 리포트별 부분 답변
            p = llm_json(
                self.client,
                "리포트에서 질문과 관련된 내용을 추출하고 관련도(0~100)를 매겨라. "
                '관련 없으면 score를 0으로. JSON: {"score": ..., "answer": ...}',
                f"<리포트: {r['title']}>\n{r['summary']}\n</리포트>\n\n질문: {question}",
                max_tokens=512)
            if p["score"] > 20:
                partials.append(p)

        partials.sort(key=lambda p: -p["score"])     # Reduce: 종합
        merged = "\n\n".join(p["answer"] for p in partials[:5])
        return llm_text(self.client,
                        "부분 답변들을 종합해 완결된 최종 답변을 작성하라.",
                        f"질문: {question}\n\n{merged}")


# ---------------------------------------------------------------
# 14회차: 미니 NodeRAG
# ---------------------------------------------------------------

DECOMPOSE_PROMPT = """청크를 분해해 JSON으로 출력하라:
{"semantic_units": [
   {"summary": "독립적으로 이해 가능한 사실/사건 단위 요약", "entities": ["언급된 개체명들"]}],
 "relations": [{"s": ..., "desc": "관계 서술문", "o": ...}]}"""


def build_hetero_graph(client, chunks: list[str], cache_name: str) -> nx.Graph:
    """T/S/N/R 이종 노드 그래프 구축 (14회차 실습, 분해 결과 캐시)."""
    cache = load_cache(cache_name) or {}
    missing = [i for i in range(len(chunks)) if str(i) not in cache]
    if missing and client is None:
        raise RuntimeError(
            f"분해 캐시(hw_cache/{cache_name}.json)가 없어 LLM 호출이 필요하다 — "
            "ANTHROPIC_API_KEY를 설정하고 다시 실행하라.")
    for i in missing:
        cache[str(i)] = llm_json(client, DECOMPOSE_PROMPT, chunks[i],
                                 max_tokens=2048)
        save_cache(cache_name, cache)
        print(f"  분해: 청크 {i} → 의미단위 {len(cache[str(i)]['semantic_units'])}, "
              f"관계 {len(cache[str(i)]['relations'])}")

    G = nx.Graph()
    for ci, chunk in enumerate(chunks):
        t_id = f"T{ci}"
        G.add_node(t_id, node_type="T", text=chunk)
        data = cache[str(ci)]
        for si, su in enumerate(data["semantic_units"]):
            s_id = f"S{ci}-{si}"
            G.add_node(s_id, node_type="S", text=su["summary"])
            G.add_edge(t_id, s_id)                    # T ↔ S
            for ent in su["entities"]:
                G.add_node(ent, node_type="N")
                G.add_edge(s_id, ent)                 # S ↔ N
        for ri, r in enumerate(data["relations"]):
            r_id = f"R{ci}-{ri}"
            G.add_node(r_id, node_type="R", text=r["desc"])
            for ent in (r["s"], r["o"]):              # 관계 끝점도 N 노드로 보장
                if not G.has_node(ent):
                    G.add_node(ent, node_type="N")
            G.add_edge(r["s"], r_id)
            G.add_edge(r_id, r["o"])                  # N ↔ R ↔ N
    return G


def ppr_search(G, entry_nodes: list[str], top_k: int = 10,
               alpha: float = 0.85) -> list[str]:
    """Personalized PageRank로 진입점에서 관련성 확산 → 내용 노드만 반환 (14회차)."""
    entries = [n for n in entry_nodes if G.has_node(n)]
    if not entries:
        return []
    personalization = {n: 1.0 for n in entries}
    # alpha가 높으면(0.95) 수렴이 느리다 — 기본 max_iter(100)로는 실패할 수 있음
    scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                         max_iter=1000)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [n for n in ranked
            if G.nodes[n].get("node_type") in ("T", "S", "A", "H")][:top_k]


class NodeRAGSearch:
    """이중 진입(벡터 + 개체명) → PPR 확산 → 내용 노드 컨텍스트 조립 (14회차)."""

    def __init__(self, G, embed_model="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.G = G
        self.model = SentenceTransformer(embed_model)
        self.content = [n for n, d in G.nodes(data=True) if "text" in d]
        self.embs = self.model.encode([G.nodes[n]["text"] for n in self.content])

    def search(self, question: str, k: int = 8, alpha: float = 0.85) -> str:
        from sentence_transformers import util

        q = self.model.encode(question)
        sims = util.cos_sim(q, self.embs)[0]
        vector_entries = [self.content[i]
                          for i in sims.argsort(descending=True)[:5]]
        name_entries = [n for n, d in self.G.nodes(data=True)
                        if d.get("node_type") == "N" and n in question]

        picked = ppr_search(self.G, vector_entries + name_entries,
                            top_k=k, alpha=alpha)
        return "\n\n".join(f"[{self.G.nodes[n]['node_type']}] "
                           f"{self.G.nodes[n]['text']}" for n in picked)


# ---------------------------------------------------------------
# 공용: 컨텍스트 기반 답변과 정답 판정 (비교표 작성용)
# ---------------------------------------------------------------

def answer_with_context(client, question: str, context: str) -> str:
    """조립된 컨텍스트를 5회차 ask()와 같은 규칙으로 LLM에 투입."""
    return llm_text(
        client,
        "반드시 주어진 컨텍스트에 근거해서만 답변하라. "
        "컨텍스트에 답이 없으면 '제공된 문서에서 답을 찾을 수 없습니다'라고 답하라.",
        f"<컨텍스트>\n{context}\n</컨텍스트>\n\n질문: {question}", max_tokens=1024)


def judge_answer(client, question: str, ground_truth: str, answer: str) -> bool:
    """답변이 정답(ground_truth)의 핵심 사실과 일치하는지 LLM으로 판정 (9회차 원칙)."""
    verdict = llm_json(
        client,
        "답변이 정답의 핵심 사실을 담고 있는지 판정하라. 부분적으로만 맞으면 false. "
        'JSON: {"correct": true|false, "reason": "한 줄 근거"}',
        f"질문: {question}\n정답: {ground_truth}\n답변: {answer}", max_tokens=300)
    return bool(verdict.get("correct"))
