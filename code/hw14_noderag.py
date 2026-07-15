# hw14_noderag.py — 14회차 과제 해답
#
# 과제 1: 미니 NodeRAG 구축 — multi-hop 질문 5개에서 GraphRAG Local Search와 비교
# 과제 2: PPR alpha(0.5 / 0.85 / 0.95) 변화가 검색 결과에 미치는 영향 관찰
# 과제 3: 9회차 평가 하네스로 3대 방법 + 기본 RAG 최종 비교표 → 17회차 지참
#
# 최초 실행 시 분해·추출에 API 키 필요 (hw_cache/에 캐시). 과제 3은 12·13회차
# hw 캐시를 재사용한다 (없으면 해당 회차 --task 1을 먼저 실행).
#
# 실행: python hw14_noderag.py [--task 1 2 3] [--limit N] [--list]

import json
from collections import Counter

from corpus import DATA_DIR, ensure_indexed, load_company_chunks
from graph_common import (GlobalSearch, LocalSearch, NodeRAGSearch,
                          answer_with_context, build_community_reports,
                          build_hetero_graph, build_rich_graph, find_neighbors,
                          guess_entity, judge_answer, ppr_search)
from hw_common import (CACHE_DIR, get_client, load_cache, print_table, run_tasks,
                       save_cache, skip)
from rag_pipeline import SimpleRAG

MULTIHOP_IDS = ["gr-002", "gr-003", "gr-004", "gr-007", "gr-008"]


def _chunks() -> list[str]:
    return [c["text"] for c in load_company_chunks()]


def _graph_eval() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(
        (DATA_DIR / "eval_set_graph.json").read_text(encoding="utf-8"))}


def _hetero():
    return build_hetero_graph(get_client(), _chunks(), "hw14_decompose")


def task1(args):
    """미니 NodeRAG vs GraphRAG Local Search — multi-hop 5문항."""
    client = get_client()
    H = _hetero()
    types = Counter(d.get("node_type") for _, d in H.nodes(data=True))
    print(f"이종 그래프: {H.number_of_nodes()}노드 — 타입 분포 {dict(types)}\n")

    noderag = NodeRAGSearch(H)
    G = build_rich_graph(client, _chunks(), "hw12_extract")   # 12회차 그래프 재사용
    local = LocalSearch(G, _chunks())

    cases = [_graph_eval()[i] for i in MULTIHOP_IDS]
    rows, score = [], {"Local": 0, "NodeRAG": 0}
    for case in cases:
        q = case["question"]
        ctx_node = noderag.search(q)
        ctx_local = local.search(q)
        print(f"─ {case['id']}: {q}")
        print("  NodeRAG 컨텍스트 노드:",
              [line.split("]")[0] + "]" for line in ctx_node.split("\n\n")][:8])
        if client is not None:
            a_local = answer_with_context(client, q, ctx_local)
            a_node = answer_with_context(client, q, ctx_node)
            ok_l = judge_answer(client, q, case["ground_truth"], a_local)
            ok_n = judge_answer(client, q, case["ground_truth"], a_node)
            score["Local"] += ok_l
            score["NodeRAG"] += ok_n
            rows.append([case["id"], "✓" if ok_l else "✗", "✓" if ok_n else "✗"])
    if rows:
        print()
        print_table(["문항", "GraphRAG Local", "NodeRAG"], rows)
        print(f"정답: Local {score['Local']}/{len(rows)} vs NodeRAG {score['NodeRAG']}/{len(rows)}")
    else:
        skip("답변·판정은 ANTHROPIC_API_KEY 필요 — 컨텍스트 조립까지 비교했다.")
    print("""
비교 포인트:
- Local Search는 "개체" 중심(설명+이웃+원문 통째)이라 컨텍스트가 비대해지기 쉽다.
- NodeRAG는 S(사실 단위)·R(관계문) 노드가 잘게 검색·랭킹되므로 multi-hop 질문의
  중간 고리("단종 결정", "부품 수급 불안")가 독립 노드로 걸려든다 — 정밀도의 원천.""")


def task2(args):
    """PPR alpha 스윕 — 확산 강도가 검색 결과를 어떻게 바꾸나."""
    H = _hetero()
    noderag = NodeRAGSearch(H)
    question = "PD-2000 단종의 배경은?"

    from sentence_transformers import util

    q = noderag.model.encode(question)
    sims = util.cos_sim(q, noderag.embs)[0]
    entries = [noderag.content[i] for i in sims.argsort(descending=True)[:5]]
    entries += [n for n, d in H.nodes(data=True)
                if d.get("node_type") == "N" and n in question]
    print(f"질문: {question}\n진입점: {entries}\n")

    picks = {}
    for alpha in (0.5, 0.85, 0.95):
        picked = ppr_search(H, entries, top_k=8, alpha=alpha)
        picks[alpha] = picked
        print(f"alpha={alpha}: " + ", ".join(
            f"{n}({H.nodes[n]['node_type']})" for n in picked))

    base = set(picks[0.85])
    for alpha in (0.5, 0.95):
        diff = set(picks[alpha]) ^ base
        print(f"\nalpha {alpha} vs 0.85 — 달라진 노드 {len(diff)}개: {sorted(diff)[:6]}")
    print("""
해석 (랜덤 서퍼 직관):
- alpha↓(0.5): 매 걸음의 절반을 진입점으로 순간이동 → 진입점 바로 옆만 높게 —
  국소적, 안전하지만 multi-hop 고리를 놓친다.
- alpha↑(0.95): 엣지를 따라 멀리 확산 → 허브 노드(대표 개체·긴 청크)가 끼어들어
  노이즈 위험. 0.85가 관례인 이유를 결과로 확인하라.
- 42청크 코퍼스에선 차이가 미묘할 수 있다 — 그래프가 클수록 alpha가 성능을 가른다.""")


def task3(args):
    """최종 비교표 — 기본 RAG / GraphRAG / OntoRAG / NodeRAG (17회차 지참)."""
    client = get_client()
    if client is None:
        skip("4개 방법의 답변 생성·판정에 ANTHROPIC_API_KEY가 필요하다.")
        return

    rag = SimpleRAG()
    ensure_indexed(rag)
    G = build_rich_graph(client, _chunks(), "hw12_extract")
    reports = build_community_reports(client, G, "hw12_reports")
    local, global_ = LocalSearch(G, _chunks()), GlobalSearch(client, reports)
    H = _hetero()
    noderag = NodeRAGSearch(H)

    from hw13_ontorag import ONTOLOGY, _valid_graph, apply_inference

    onto_graph = apply_inference(ONTOLOGY, _valid_graph())

    def basic_answer(q):
        chunks = rag.retrieve(q, k=4, max_distance=1.0)
        return answer_with_context(client, q, "\n\n".join(c["text"] for c in chunks))

    def graphrag_answer(q, qtype):
        if qtype == "global":
            return global_.search(q)
        return answer_with_context(client, q, local.search(q))

    def onto_answer(q):
        entity = guess_entity(onto_graph, q)
        if entity is None:
            return "제공된 문서에서 답을 찾을 수 없습니다."
        facts = find_neighbors(onto_graph, entity, depth=2)
        ctx = "\n".join(f"({s}, {r}, {o})" for s, r, o in facts)
        return answer_with_context(client, q, ctx)

    cases = list(_graph_eval().values())
    limit = getattr(args, "limit", None)
    if limit:
        cases = cases[:limit]

    cache = load_cache("hw14_final_answers") or {}
    methods = ["기본 RAG", "GraphRAG", "OntoRAG", "NodeRAG"]
    rows, totals = [], Counter()
    for case in cases:
        q, key = case["question"], case["id"]
        if key not in cache:
            cache[key] = {
                "기본 RAG": basic_answer(q),
                "GraphRAG": graphrag_answer(q, case["type"]),
                "OntoRAG": onto_answer(q),
                "NodeRAG": answer_with_context(client, q, noderag.search(q)),
            }
            save_cache("hw14_final_answers", cache)
            print(f"  답변 생성: {key}")
        marks = []
        for m in methods:
            ok = judge_answer(client, q, case["ground_truth"], cache[key][m])
            totals[m] += ok
            marks.append("✓" if ok else "✗")
        rows.append([key, case["type"]] + marks)

    print()
    print_table(["문항", "유형"] + methods, rows)
    n = len(rows)
    summary = [[m, f"{totals[m]}/{n}"] for m in methods]
    print()
    print_table(["방법", "정답률"], summary)

    md = ["| 문항 | 유형 | " + " | ".join(methods) + " |",
          "|---|---|" + "---|" * len(methods)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    md += ["", "정답률: " + ", ".join(f"{m} {totals[m]}/{n}" for m in methods)]
    out = CACHE_DIR / "hw14_final_table.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n최종 비교표 저장: {out}  ← 17회차 종합 프로젝트에 지참")
    print("""
읽는 법 (14회차 종합 비교표와 대조하라):
- global → GraphRAG(Global), multi-hop 세부 → NodeRAG, 계층·규칙 질의 → OntoRAG가
  각자 강해야 정상 — 한 방법이 전부를 이기지 않는 것이 이 표의 교훈이다.
- 17회차 설계 문제: "본인 코퍼스의 질문 분포에서 어떤 조합이 비용 대비 최적인가".""")


if __name__ == "__main__":
    run_tasks("14회차: NodeRAG", {
        1: ("미니 NodeRAG vs GraphRAG Local (multi-hop 5문항)", task1),
        2: ("PPR alpha(0.5/0.85/0.95) 스윕", task2),
        3: ("최종 비교표: 기본/GraphRAG/OntoRAG/NodeRAG", task3),
    }, extra_args=[("--limit", {"type": int, "metavar": "N",
                                "help": "과제 3에서 평가할 문항 수 제한 (비용 절약)"})])
