# hw11_graphrag_concepts.py — 11회차 과제 해답
#
# 과제 1: 그래프를 20문서 이상으로 확장 → 커뮤니티 탐지 → 커뮤니티별 LLM 요약
# 과제 2: 전역 질문 3개 — (a) 기본 RAG vs (b) 커뮤니티 요약 기반 답변 비교
# 과제 3: GraphRAG 도입 판단문 — 9회차 실패 사례 기반 (데이터로 채운 판단문 생성)
#
# 확장 코퍼스: 회사 문서 12건 + 핸드북 소절 (총 42청크 ≥ 20문서).
# 최초 실행 시 추출·요약에 API 키 필요 (hw_cache/에 캐시됨). 과제 3은 키 불필요.
#
# 실행: python hw11_graphrag_concepts.py [--task 1 2 3] [--list]

import json

import networkx.algorithms.community as nx_comm

from corpus import DATA_DIR, ensure_indexed, load_corpus
from graph_common import answer_with_context, build_triple_graph
from hw_common import (get_client, llm_text, load_cache, print_table, run_tasks,
                       save_cache, skip)
from rag_pipeline import SimpleRAG
from retrieval_metrics import hit_at_k, recall_at_k

GLOBAL_QUESTIONS = [
    "이 회사의 사업 구조를 전체적으로 요약해줘",
    "회사 조직은 어떻게 구성되어 있나?",
    "회사가 안고 있는 리스크 요인들을 정리해줘",
]


def _graph():
    return build_triple_graph(get_client(), load_corpus(), "hw11_triples")


def _communities(G):
    return [c for c in nx_comm.louvain_communities(G.to_undirected(), seed=42)
            if len(c) >= 3]


def _summaries(G) -> list[dict]:
    """커뮤니티별 LLM 요약 (캐시). 반환: [{"members": [...], "summary": ...}]"""
    cached = load_cache("hw11_summaries")
    if cached is not None:
        return cached
    client = get_client()
    if client is None:
        raise RuntimeError("요약 캐시가 없어 LLM이 필요하다 — ANTHROPIC_API_KEY 설정 후 재실행.")

    summaries = []
    for i, comm in enumerate(_communities(G)):
        facts = [(u, d["relation"], v) for u, v, d in G.edges(data=True)
                 if u in comm and v in comm]
        facts_text = "\n".join(f"({s}, {r}, {o})" for s, r, o in facts)
        summary = llm_text(
            client,
            "지식 그래프 사실들로 이 주제 군집을 2~3문장으로 요약하라. "
            "첫 줄에 '제목: ...'을 붙여라.",
            facts_text, max_tokens=512)
        summaries.append({"community_id": i, "members": sorted(comm),
                          "summary": summary})
        print(f"  커뮤니티 {i} 요약 완료 ({len(comm)}개 노드)")
    save_cache("hw11_summaries", summaries)
    return summaries


def task1(args):
    """확장 그래프 → 커뮤니티 탐지 → LLM 요약."""
    G = _graph()
    comms = _communities(G)
    print(f"확장 그래프: 노드 {G.number_of_nodes()} / 엣지 {G.number_of_edges()} "
          f"(코퍼스 {len(load_corpus())}청크)")
    print(f"커뮤니티(3개 노드 이상): {len(comms)}개\n")
    for i, comm in enumerate(comms):
        print(f"  커뮤니티 {i} ({len(comm)}): {sorted(comm)[:8]}"
              f"{' ...' if len(comm) > 8 else ''}")

    print("\n--- 커뮤니티 요약 (사전 계산 = GraphRAG 인덱싱의 핵심) ---")
    for s in _summaries(G):
        print(f"\n[커뮤니티 {s['community_id']}] {s['summary']}")
    print("""
관찰 포인트:
- 커뮤니티가 직관적 주제(영업 조직 / 제품·생산 / 인사 규정...)로 나뉘는가?
- 엉뚱하게 묶였다면 원인은 대부분 추출 품질(10회차 과제 2) — 그래프가 곧 입력이다.
- 이 요약들이 12회차 Global Search가 순회할 "사전 계산된 답변 재료"다.""")


def task2(args):
    """전역 질문 3개 — 기본 RAG vs 커뮤니티 요약 기반."""
    client = get_client()
    if client is None:
        skip("답변 생성 비교는 ANTHROPIC_API_KEY가 필요하다.")
        return
    G = _graph()
    summaries = _summaries(G)
    rag = SimpleRAG()
    ensure_indexed(rag)

    summary_context = "\n\n".join(s["summary"] for s in summaries)
    for q in GLOBAL_QUESTIONS:
        print(f"\n{'─' * 60}\n전역 질문: {q}")
        chunks = rag.retrieve(q, k=4, max_distance=1.0)
        chunk_context = "\n\n".join(c["text"] for c in chunks)
        print(f"\n(a) 기본 RAG — top-4 청크({[c['id'] for c in chunks]})로 답변:")
        print(answer_with_context(client, q, chunk_context).strip()[:400])
        print("\n(b) 커뮤니티 요약 기반 답변:")
        print(answer_with_context(client, q, summary_context).strip()[:400])

    print("""
비교 기록 포인트:
- (a)는 우연히 걸린 청크 몇 개의 사실을 나열한다 — 매출 비중(01)만 잡히고
  제품 전환·프로젝트 맥락이 빠지는 식의 "부분 조망"이 전형적.
- (b)는 커뮤니티가 이미 주제 단위이므로 전체 구조(사업 축 → 제품 전환 → 신사업)를
  체계적으로 답한다 — "답을 검색하지 말고 미리 만들어 두라"의 실증.""")


def task3(args):
    """도입 판단문 — 유형별 검색 성적을 재서 숫자가 채워진 판단문을 만든다."""
    rag = SimpleRAG()
    ensure_indexed(rag)
    graph_set = json.loads((DATA_DIR / "eval_set_graph.json").read_text(encoding="utf-8"))

    per_type: dict[str, list[float]] = {}
    rows = []
    for case in graph_set:
        answers = {d.removesuffix(".txt") for d in case["source_docs"]}
        ids = [c["id"] for c in rag.retrieve(case["question"], k=3, max_distance=1.0)]
        rec = recall_at_k(ids, answers, 3)
        per_type.setdefault(case["type"], []).append(rec)
        rows.append([case["id"], case["type"], "✓" if hit_at_k(ids, answers, 3) else "✗",
                     f"{rec:.0%}"])
    print_table(["문항", "유형", "Hit@3", "Recall@3"], rows)

    def avg(t):
        v = per_type.get(t, [])
        return sum(v) / len(v) if v else None

    hard = [t for t in ("multi-hop", "global", "comparison") if t in per_type]
    hard_cnt = sum(len(per_type[t]) for t in hard)
    weak = [t for t in hard if avg(t) is not None and avg(t) < 0.7]
    print(f"""
--- 도입 판단문 (측정값으로 자동 작성 — 본인 프로젝트용 틀로 사용) ---
1. 실패 유형 분석: 그래프 평가셋 {len(graph_set)}문항 중 관계·전역형이 {hard_cnt}건이며,
   기본 RAG의 유형별 평균 Recall@3은 """
          + ", ".join(f"{t}={avg(t):.0%}" for t in per_type) + f"""
2. 판단: Recall이 70% 미만인 유형({', '.join(weak) if weak else '없음'})은 검색기 튜닝이
   아니라 구조의 문제다 — 근거가 여러 문서에 분산되어 top-k 유사도 검색으로는
   전부 모을 수 없다.
3. 결론: {'GraphRAG 도입을 권고한다. 단, 인덱싱 비용(문서 전체 LLM 처리)이 있으므로 코퍼스가 비교적 정적일 때 유리하다.' if weak else '현재 코퍼스에서는 기본 RAG로 충분하다 — 평가셋을 더 어렵게 만들어 재검토하라.'}
4. 유보 조건: 문서가 자주 바뀌면 그래프 재구축 비용을 먼저 산정할 것 (11회차 비용표).""")


if __name__ == "__main__":
    run_tasks("11회차: GraphRAG 개념", {
        1: ("확장 그래프 → 커뮤니티 탐지 → LLM 요약", task1),
        2: ("전역 질문 3개: 기본 RAG vs 커뮤니티 요약", task2),
        3: ("GraphRAG 도입 판단문 (측정 기반)", task3),
    })
