# hw10_knowledge_graphs.py — 10회차 과제 해답
#
# 과제 1: 문서 10개+에서 트리플 추출 → 그래프 구축 → 시각화
# 과제 2: 추출 품질 검수 — 잘못 추출된 트리플 비율과 유형 기록
# 과제 3: multi-hop 질문 3개 — 벡터 RAG vs 그래프 QA 비교
#
# 코퍼스: data/company_docs/*.txt 12건. 최초 실행 시 트리플 추출에 API 키가 필요하고
# 결과는 hw_cache/hw10_triples.json에 캐시된다 (이후 키 없이 재실행 가능).
#
# 실행: python hw10_knowledge_graphs.py [--task 1 2 3] [--list]

import json
from collections import Counter

from corpus import DATA_DIR, ensure_indexed, load_company_chunks
from graph_common import build_triple_graph, find_neighbors, graph_qa, guess_entity
from hw_common import CACHE_DIR, get_client, print_table, run_tasks, skip
from rag_pipeline import SimpleRAG

TITLES = ("과장", "팀장", "본부장", "대리", "소장", "대표이사")   # 개체명 변형 탐지용


def _graph():
    return build_triple_graph(get_client(), load_company_chunks(), "hw10_triples")


def task1(args):
    """트리플 추출 → 그래프 구축 → 시각화 (PNG 저장, 실패 시 GraphML)."""
    G = _graph()
    print(f"그래프: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개 "
          f"(문서 {len(load_company_chunks())}건)")
    for u, v, d in G.edges(data=True):
        print(f"  ({u}) --[{d['relation']}]--> ({v})   ← {d['source']}")

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        # 한글 라벨: nx.draw가 sans-serif를 쓰므로 sans-serif 목록 앞에 한글 폰트 삽입
        plt.rcParams["font.sans-serif"] = (
            ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]
            + plt.rcParams["font.sans-serif"])
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=0.9)
        nx.draw(G, pos, with_labels=True, node_color="#c9b8f0",
                node_size=1600, font_size=9, edge_color="#999999", arrowsize=12)
        nx.draw_networkx_edge_labels(
            G, pos, font_size=7,
            edge_labels={(u, v): d["relation"] for u, v, d in G.edges(data=True)})
        out = CACHE_DIR / "hw10_graph.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n시각화 저장: {out}  (한글이 깨지면 나눔고딕 등 한글 폰트 설치)")
    except ImportError:
        import networkx as nx

        out = CACHE_DIR / "hw10_graph.graphml"
        nx.write_graphml(G, out)
        print(f"\nmatplotlib 없음 → GraphML로 저장: {out} (Gephi/pyvis에서 열기)")
        print("대화형 시각화는 pyvis 권장: pip install pyvis")


def task2(args):
    """추출 품질 검수 — 자동 통계 + 의심 트리플 후보 표시 + 검수 기록 예시."""
    G = _graph()
    triples = [(u, d["relation"], v, d["source"]) for u, v, d in G.edges(data=True)]

    relations = Counter(r for _, r, _, _ in triples)
    print(f"트리플 {len(triples)}개 / 관계 어휘 {len(relations)}종")
    print("관계 어휘 빈도:", dict(relations.most_common()))

    singles = [r for r, n in relations.items() if n == 1]
    print(f"\n[의심 1] 1회만 쓰인 관계 {len(singles)}종: {singles}")
    print("  → 같은 의미의 다른 표기(소속/재직/속해있음)가 섞였는지 확인 — 13회차 OntoRAG의 동기")

    # 개체명 변형 후보: 직급·공백 제거 후 한쪽이 다른 쪽의 접두면 같은 인물일 수 있다
    def norm(name: str) -> str:
        s = name.replace(" ", "")
        for t in TITLES:
            s = s.removesuffix(t)
        return s

    nodes = list(G.nodes)
    variants = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                if norm(a) and norm(b) and norm(a) != norm(b)
                and (norm(a).startswith(norm(b)) or norm(b).startswith(norm(a)))]
    print(f"\n[의심 2] 개체명 변형 후보 {len(variants)}쌍: {variants}")
    print('  → 08-customers.txt의 "김 과장"이 "김철수"와 분리됐다면 개체 해소 필요 (12회차 심화)')

    print("""
[검수 기록 예시] (사람이 트리플을 훑으며 채우는 표 — 위 자동 통계는 후보 탐색용)
  전체 N개 중 오류 n개 (오류율 n/N)
  유형별:
    - 관계 어휘 중복 표기: "소속" vs "재직" 등          → OntoRAG(13회차)로 차단
    - 개체명 변형:        "김철수" vs "김 과장"          → 개체 해소(12회차 심화)
    - 방향 오류:          (PD-2000, 담당, 김철수)        → 스키마의 domain/range로 검증
    - 과잉 추출:          ("2025년", 기준, "매출 비중") 같은 무의미 트리플
  기록 후 추출 프롬프트에 규칙을 추가해 재추출 → 오류율 변화를 측정하는 것까지가 과제다.""")


def task3(args):
    """multi-hop 질문 3개 — 벡터 RAG의 top-3와 그래프 QA를 나란히 비교."""
    G = _graph()
    rag = SimpleRAG()
    ensure_indexed(rag)
    client = get_client()

    graph_eval = {c["id"]: c for c in json.loads(
        (DATA_DIR / "eval_set_graph.json").read_text(encoding="utf-8"))}
    cases = [graph_eval["gr-002"], graph_eval["gr-003"], graph_eval["gr-008"]]

    for case in cases:
        needed = {d.removesuffix(".txt") for d in case["source_docs"]}
        print(f"\n{'─' * 60}\n질문: {case['question']}")
        print(f"필요한 근거 문서: {sorted(needed)}")

        print("  [벡터 RAG top-3]")
        got = set()
        for c in rag.retrieve(case["question"], k=3, max_distance=1.0):
            mark = "○" if c["id"] in needed else " "
            got.add(c["id"])
            print(f"    {mark} ({c['distance']:.3f}) {c['id']}")
        print(f"    → 근거 확보 {len(needed & got)}/{len(needed)} "
              f"{'(전부 확보 — 답변 가능)' if needed <= got else '(누락 — 추론 사슬 끊김)'}")

        entity = guess_entity(G, case["question"])
        facts = find_neighbors(G, entity, depth=2) if entity else []
        print(f"  [그래프 QA] 시작 개체: {entity} → 2-hop 사실 {len(facts)}개 수집")
        if client is not None and entity:
            print("    답변:", graph_qa(client, G, case["question"], entity).strip()[:300])
        else:
            for s, r, o in facts[:8]:
                print(f"    ({s}, {r}, {o})")
            skip("답변 생성은 ANTHROPIC_API_KEY 필요 — 수집된 사실만 표시했다.")
        print(f"    정답 (참고): {case['ground_truth']}")

    print("""
비교 기록 포인트:
- 벡터 RAG는 질문 전체와 비슷한 청크 1~2개만 가져와 사슬의 중간 고리를 놓친다.
- 그래프 QA는 시작 개체에서 2-hop 탐색으로 사슬 전체(김철수→영업1팀→영업본부→
  박영희→5천만 원)를 기계적으로 수집한다 — 추출만 잘 돼 있으면 추론이 안정적.
- 남은 문제(시작 개체 자동 탐지, 전역 질문)는 11~12회차 GraphRAG가 해결한다.""")


if __name__ == "__main__":
    run_tasks("10회차: 지식 그래프", {
        1: ("트리플 추출 → 그래프 구축 → 시각화", task1),
        2: ("추출 품질 검수 (오류 비율·유형 기록)", task2),
        3: ("multi-hop 3문항: 벡터 RAG vs 그래프 QA", task3),
    })
