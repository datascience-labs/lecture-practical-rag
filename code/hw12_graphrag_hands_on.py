# hw12_graphrag_hands_on.py — 12회차 과제 해답
#
# 과제 1: 미니 GraphRAG 전체 파이프라인 실행 (추출 캐시 필수!)
# 과제 2: 전역 질문 포함 평가셋으로 기본 RAG vs GraphRAG 평가표 갱신
# 과제 3: (심화) 개체 해소 — 임베딩 유사도 상위 쌍을 LLM 판정으로 병합
# 과제 4: (심화) Microsoft graphrag 라이브러리 비교 — 실행 가이드
#
# 과제 1~3은 최초 실행 시 API 키 필요 (모든 LLM 산출물은 hw_cache/에 캐시).
#
# 실행: python hw12_graphrag_hands_on.py [--task 1 2 3 4] [--list]

import json

from corpus import DATA_DIR, ensure_indexed, load_company_chunks
from graph_common import (GlobalSearch, LocalSearch, answer_with_context,
                          build_community_reports, build_rich_graph, judge_answer)
from hw_common import (get_client, llm_json, load_cache, print_table, run_tasks,
                       save_cache, skip)
from rag_pipeline import SimpleRAG


def _chunks() -> list[str]:
    return [c["text"] for c in load_company_chunks()]


def _pipeline(client):
    """추출(캐시) → 그래프 → 커뮤니티 리포트(캐시). 캐시가 있으면 키 없이 동작."""
    G = build_rich_graph(client, _chunks(), "hw12_extract")
    reports = build_community_reports(client, G, "hw12_reports")
    return G, reports


def _graph_eval() -> list[dict]:
    return json.loads((DATA_DIR / "eval_set_graph.json").read_text(encoding="utf-8"))


def task1(args):
    """미니 GraphRAG 전체 파이프라인 — 추출→그래프→커뮤니티→리포트→질의."""
    client = get_client()
    G, reports = _pipeline(client)
    print(f"그래프: 개체 {G.number_of_nodes()} / 관계 {G.number_of_edges()} / "
          f"커뮤니티 리포트 {len(reports)}개")
    for r in reports:
        print(f"  [{r['community_id']}] {r['title']} — 구성원 {len(r['members'])}")

    # Local Search 시연 (multi-hop)
    q_local = "김철수 과장이 담당하는 제품의 생산 공장은?"
    local = LocalSearch(G, _chunks())
    context = local.search(q_local)
    print(f"\n--- Local Search: {q_local}")
    print("조립된 컨텍스트 (앞부분):")
    print("  " + "\n  ".join(context.splitlines()[:8]))
    if client is not None:
        print("답변:", answer_with_context(client, q_local, context).strip()[:300])

    # Global Search 시연 (전역)
    if client is not None:
        q_global = "이 회사의 사업 구조를 전체적으로 요약해줘"
        print(f"\n--- Global Search: {q_global}")
        print("답변:", GlobalSearch(client, reports).search(q_global).strip()[:400])
    else:
        skip("Local/Global 답변 생성은 ANTHROPIC_API_KEY 필요 — 컨텍스트 조립까지 확인했다.")
    print("\n캐시: hw_cache/hw12_extract.json, hw12_reports.json — 재실행은 LLM 호출 없음.")


def task2(args):
    """전역 질문 포함 평가셋(eval_set_graph.json)으로 기본 RAG vs GraphRAG 평가표."""
    client = get_client()
    if client is None:
        skip("답변 생성·판정에 ANTHROPIC_API_KEY가 필요하다.")
        return
    G, reports = _pipeline(client)
    local = LocalSearch(G, _chunks())
    global_ = GlobalSearch(client, reports)
    rag = SimpleRAG()
    ensure_indexed(rag)

    answers_cache = load_cache("hw12_answers") or {}
    rows, score = [], {"기본": 0, "GraphRAG": 0}
    for case in _graph_eval():
        key = case["id"]
        if key not in answers_cache:
            chunks = rag.retrieve(case["question"], k=4, max_distance=1.0)
            basic = answer_with_context(client, case["question"],
                                        "\n\n".join(c["text"] for c in chunks))
            if case["type"] == "global":
                graph = global_.search(case["question"])
            else:
                graph = answer_with_context(client, case["question"],
                                            local.search(case["question"]))
            answers_cache[key] = {"basic": basic, "graph": graph}
            save_cache("hw12_answers", answers_cache)

        a = answers_cache[key]
        ok_b = judge_answer(client, case["question"], case["ground_truth"], a["basic"])
        ok_g = judge_answer(client, case["question"], case["ground_truth"], a["graph"])
        score["기본"] += ok_b
        score["GraphRAG"] += ok_g
        rows.append([key, case["type"], "✓" if ok_b else "✗", "✓" if ok_g else "✗"])

    n = len(rows)
    print_table(["문항", "유형", "기본 RAG", "GraphRAG"], rows)
    print(f"\n정답률: 기본 RAG {score['기본']}/{n}  vs  GraphRAG {score['GraphRAG']}/{n}")
    print("""
평가표 읽는 법:
- global 유형에서 격차가 가장 커야 정상 — Global Search의 존재 이유.
- single-hop에서 GraphRAG가 오히려 지면 정상적인 신호다: 단순 질문은 기본 RAG가
  더 싸고 정확하다 → 12회차 결론 "질문 유형별 라우팅"의 근거.
- 판정은 LLM-as-Judge(9회차) — 같은 judge로 일관 측정해야 비교 가능하다.""")


def task3(args):
    """(심화) 개체 해소 — 임베딩 유사 쌍을 LLM에게 '같은 개체인가' 판정시켜 병합."""
    client = get_client()
    G, _ = _pipeline(client)

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("BAAI/bge-m3")
    names = list(G.nodes)
    embs = model.encode([f"{n}: {G.nodes[n]['description']}" for n in names])
    sim = util.cos_sim(embs, embs)

    pairs = sorted(
        ((float(sim[i][j]), names[i], names[j])
         for i in range(len(names)) for j in range(i + 1, len(names))
         if sim[i][j] > 0.75),
        reverse=True)[:8]
    print(f"임베딩 유사도 0.75 초과 개체쌍 {len(pairs)}개 (상위 8):")
    for s, a, b in pairs:
        print(f"  {s:.3f}  {a}  ↔  {b}")

    if client is None:
        skip("LLM 판정·병합은 ANTHROPIC_API_KEY가 필요하다.")
        return

    import networkx as nx

    merged = 0
    for s, a, b in pairs:
        verdict = llm_json(
            client,
            "두 개체가 같은 실체를 가리키는지 설명을 근거로 판정하라. "
            'JSON: {"same": true|false, "reason": "한 줄"}',
            f"개체1 {a}: {G.nodes[a]['description']}\n"
            f"개체2 {b}: {G.nodes[b]['description']}", max_tokens=200)
        mark = "병합" if verdict.get("same") else "유지"
        print(f"  [{mark}] {a} ↔ {b} — {verdict.get('reason', '')}")
        if verdict.get("same") and G.has_node(a) and G.has_node(b):
            keep, drop = (a, b) if len(a) >= len(b) else (b, a)
            G = nx.contracted_nodes(G, keep, drop, self_loops=False)
            merged += 1
    print(f"\n병합 {merged}건 → 노드 {G.number_of_nodes()}개")
    print("""
검증 포인트:
- data/company_docs/08-customers.txt는 김철수를 "김 과장"으로 표기한다 — 이 쌍이
  병합됐다면 성공. "대한중공업 담당자" 질문이 이제 김철수의 이웃에서 잡힌다.
- 이름 문자열 유사도가 아니라 "이름+설명" 임베딩을 쓴 이유: 김 과장/김철수는
  글자로는 다르지만 설명(영업1팀, 대한중공업)이 겹친다.""")


def task4(args):
    """(심화) Microsoft graphrag 라이브러리 실행 가이드 — 직접 구현과의 대응표."""
    print("""graphrag 라이브러리는 의존성이 크고 별도 LLM 설정이 필요해 여기서 실행하지
않는다. 아래 절차로 같은 코퍼스를 인덱싱해 직접 구현과 비교하라:

  pip install graphrag
  mkdir -p ./ragproj/input && cp data/company_docs/*.txt ./ragproj/input/
  graphrag init --root ./ragproj          # settings.yaml, .env 생성
  # .env에 API 키, settings.yaml의 models 항목 설정
  graphrag index --root ./ragproj         # 추출→그래프→커뮤니티→리포트
  graphrag query --root ./ragproj --method local  "김철수가 담당하는 제품은?"
  graphrag query --root ./ragproj --method global "전체 사업 구조를 요약해줘"

비교 관찰 포인트 (직접 구현 ↔ 라이브러리):
  1. output/ 의 entities/communities parquet ↔ hw_cache/hw12_extract.json
     — 개체 수·커뮤니티 수가 비슷한 규모인가?
  2. "김 과장" 개체 해소가 라이브러리에선 어떻게 처리되는가 (과제 3과 비교)
  3. 같은 전역 질문의 Global 답변 품질 — 프롬프트 튜닝 없이도 나은가?
  4. 인덱싱 LLM 호출 횟수·비용 — 직접 구현(청크당 1회)과의 차이
결론 틀: "학습·커스터마이즈는 직접 구현, 운영 규모는 라이브러리"가 일반해다.""")


if __name__ == "__main__":
    run_tasks("12회차: GraphRAG 실습", {
        1: ("미니 GraphRAG 전체 파이프라인 (추출 캐시)", task1),
        2: ("기본 RAG vs GraphRAG 평가표 (LLM judge)", task2),
        3: ("(심화) 개체 해소 — 임베딩 후보 + LLM 판정 병합", task3),
        4: ("(심화) Microsoft graphrag 라이브러리 비교 가이드", task4),
    })
