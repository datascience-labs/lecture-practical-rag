# hw13_ontorag.py — 13회차 과제 해답
#
# 과제 1: 도메인 온톨로지 설계 (클래스 5+, 속성 5+, LLM 초안 → 수동 개선)
# 과제 2: 12회차 코퍼스를 온톨로지 제약으로 재추출 → 검증 통과율·관계 어휘 수를
#         자유 추출(10회차)과 비교
# 과제 3: transitive 속성 — 추론 전에는 답 못하던 질문이 추론 후 답해지는 사례
#
# 최초 실행 시 추출에 API 키 필요 (hw_cache/hw13_triples.json에 캐시).
#
# 실행: python hw13_ontorag.py [--task 1 2 3] [--list]

import json
from collections import Counter

import networkx as nx

from corpus import load_company_chunks
from graph_common import graph_qa
from hw_common import (get_client, llm_json, llm_text, load_cache, print_table,
                       run_tasks, save_cache, skip)

# --- 수동 개선을 마친 퍼플델타 온톨로지 (슬라이드 버전 + 회사 문서 커버 확장) ---
ONTOLOGY = {
    "classes": {
        "사람":    {"parent": None},
        "직원":    {"parent": "사람"},
        "관리자":  {"parent": "직원"},
        "조직":    {"parent": None},
        "회사":    {"parent": "조직"},        # 고객사·공급사 (퍼플델타 외부)
        "제품":    {"parent": None},
        "시설":    {"parent": None},
        "프로젝트": {"parent": None},
    },
    "properties": {
        "소속":     {"domain": "사람", "range": "조직"},
        "상위조직":  {"domain": "조직", "range": "조직", "transitive": True},
        "담당":     {"domain": "직원", "range": "제품", "inverse": "담당자"},
        "생산처":    {"domain": "제품", "range": "시설"},
        "총괄":     {"domain": "관리자", "range": "조직"},
        "고객":     {"domain": "제품", "range": "회사"},
        "공급사":    {"domain": "제품", "range": "회사"},
        "수행조직":  {"domain": "프로젝트", "range": "조직"},
    },
}


def ontology_prompt(onto: dict) -> str:
    """온톨로지를 명시한 제약 추출 프롬프트 (13회차 Step 2)."""
    classes = ", ".join(onto["classes"].keys())
    props = "\n".join(f'- {name}: {p["domain"]} → {p["range"]}'
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


def is_subclass(onto, cls, ancestor) -> bool:
    while cls is not None:
        if cls == ancestor:
            return True
        cls = onto["classes"].get(cls, {}).get("parent")
    return False


def validate_triple(onto: dict, t: dict) -> tuple[bool, str]:
    """추출 결과 검증 — LLM이 규칙을 어겨도 코드가 걸러낸다 (13회차 Step 3)."""
    prop = onto["properties"].get(t["r"])
    if prop is None:
        return False, f"미정의 관계: {t['r']}"
    if not is_subclass(onto, t["s_type"], prop["domain"]):
        return False, f"주어 타입 오류: {t['s_type']} (기대: {prop['domain']})"
    if not is_subclass(onto, t["o_type"], prop["range"]):
        return False, f"목적어 타입 오류: {t['o_type']} (기대: {prop['range']})"
    return True, "OK"


def apply_inference(onto: dict, G: nx.DiGraph) -> nx.DiGraph:
    """transitive / inverse 추론 (13회차 Step 4). inferred=True 표시 유지."""
    for name, p in onto["properties"].items():
        if p.get("transitive"):
            edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d["relation"] == name]
            closure = nx.transitive_closure(nx.DiGraph(edges))
            for u, v in closure.edges():
                if not G.has_edge(u, v):
                    G.add_edge(u, v, relation=name, inferred=True)
        if p.get("inverse"):
            for u, v, d in list(G.edges(data=True)):
                if d["relation"] == name:
                    G.add_edge(v, u, relation=p["inverse"], inferred=True)
    return G


def _constrained_triples() -> list[dict]:
    """온톨로지 제약 추출 (문서별 캐시)."""
    cache = load_cache("hw13_triples") or {}
    docs = load_company_chunks()
    missing = [d for d in docs if d["id"] not in cache]
    if missing:
        client = get_client()
        if client is None:
            raise RuntimeError("추출 캐시가 없어 LLM이 필요하다 — ANTHROPIC_API_KEY 설정 후 재실행.")
        prompt = ontology_prompt(ONTOLOGY)
        for d in missing:
            cache[d["id"]] = llm_json(client, prompt, d["text"],
                                      max_tokens=2048)["triples"]
            save_cache("hw13_triples", cache)
            print(f"  제약 추출: {d['id']} → {len(cache[d['id']])}개")
    return [t for triples in cache.values() for t in triples]


def _valid_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for t in _constrained_triples():
        if validate_triple(ONTOLOGY, t)[0]:
            G.add_edge(t["s"], t["o"], relation=t["r"])
    return G


def task1(args):
    """온톨로지 설계 — LLM 초안을 받아 사람이 다듬는 워크플로 시연."""
    client = get_client()
    draft = load_cache("hw13_draft")
    if draft is None and client is not None:
        sample = "\n\n".join(d["text"] for d in load_company_chunks()[:4])
        draft = llm_json(
            client,
            "문서 샘플을 보고 지식 그래프용 온톨로지 초안을 설계하라. JSON: "
            '{"classes": {이름: {"parent": 부모 또는 null}}, '
            '"properties": {이름: {"domain": ..., "range": ...}}}',
            sample, max_tokens=2048)
        save_cache("hw13_draft", draft)
    if draft:
        print("--- LLM 초안 (검토 전) ---")
        print(json.dumps(draft, ensure_ascii=False, indent=2)[:900])
    else:
        skip("LLM 초안 시연은 ANTHROPIC_API_KEY 필요 — 최종본만 표시한다.")

    print("\n--- 수동 개선을 마친 최종 온톨로지 ---")
    print(json.dumps(ONTOLOGY, ensure_ascii=False, indent=2))
    print(f"""
클래스 {len(ONTOLOGY['classes'])}개 / 속성 {len(ONTOLOGY['properties'])}개 (요구: 각 5개 이상)

초안 → 최종 사이에서 사람이 고치는 전형적 항목 (개선 기록 예시):
1. 과잉 세분화 병합: LLM은 "부서/팀/본부"를 다 다른 클래스로 만들기 쉽다 → "조직" 하나로
2. 계층 추가: 관리자 ⊂ 직원 ⊂ 사람 — 추론(총괄의 domain=관리자)에 필요
3. 추론 속성 지정: LLM 초안엔 transitive/inverse 표시가 없다 — 상위조직(transitive),
   담당(inverse=담당자)은 사람이 질문 유형을 보고 정한다
4. 외부/내부 구분: 고객사·공급사를 "회사 ⊂ 조직"으로 — 소속(사람→조직)의 range를
   더럽히지 않으면서 고객/공급 관계를 표현""")


def task2(args):
    """제약 추출 vs 자유 추출 — 검증 통과율과 관계 어휘 수 비교."""
    triples = _constrained_triples()
    results = [validate_triple(ONTOLOGY, t) for t in triples]
    passed = sum(1 for ok, _ in results if ok)
    print(f"제약 추출 트리플 {len(triples)}개 — 검증 통과 {passed}개 "
          f"({passed / len(triples):.0%})")
    fails = Counter(msg.split(":")[0] for ok, msg in results if not ok)
    if fails:
        print(f"검증 실패 유형: {dict(fails)}")
        for t, (ok, msg) in zip(triples, results):
            if not ok:
                print(f"  ✗ ({t['s']}[{t['s_type']}], {t['r']}, {t['o']}[{t['o_type']}]) — {msg}")

    onto_rel = Counter(t["r"] for t in triples)
    row = [["제약 추출(OntoRAG)", len(onto_rel), f"{passed / len(triples):.0%}"]]

    free = load_cache("hw10_triples")            # 10회차 자유 추출과 비교
    if free:
        free_triples = [t for ts in free.values() for t in ts]
        free_rel = Counter(t["r"] for t in free_triples)
        row.append(["자유 추출(10회차)", len(free_rel), "검증 불가(스키마 없음)"])
        print(f"\n자유 추출 관계 어휘 {len(free_rel)}종: {sorted(free_rel)}")
    else:
        skip("hw10_triples 캐시 없음 — python hw10_knowledge_graphs.py --task 1 을 먼저 실행하면 비교표가 완성된다.")
    print(f"제약 추출 관계 어휘 {len(onto_rel)}종: {sorted(onto_rel)}\n")
    print_table(["방식", "관계 어휘 수", "검증 통과율"], row)
    print("""
기록 포인트:
- 관계 어휘가 수십 종 → 정확히 온톨로지의 속성 수 이하로 줄어드는 것이 핵심 효과.
  어휘가 고정돼야 그래프 탐색("담당 엣지를 따라가라")이 코드로 안정 동작한다.
- 검증 실패분은 버려지는 게 아니라 신호다: 실패가 몰리는 문장은 온톨로지에 없는
  개념(예: 부품, 계약)을 담고 있다 → 온톨로지를 확장할 다음 후보.""")


def task3(args):
    """transitive 추론 — 추론 전엔 못 답하던 질문이 답해지는 사례 1개."""
    question = "영업1팀은 본사 산하 조직인가?"
    G = _valid_graph()

    def check(g, label):
        direct = g.has_edge("영업1팀", "본사") and \
            g["영업1팀"]["본사"]["relation"] == "상위조직"
        print(f"  {label}: (영업1팀, 상위조직, 본사) 엣지 {'존재 ✓' if direct else '없음 ✗'}")
        return direct

    print(f"질문: {question}")
    print("문서에는 (영업1팀→영업본부), (영업본부→본사)만 명시되어 있다 (02-organization.txt).\n")
    before = check(G, "추론 전")
    G = apply_inference(ONTOLOGY, G)
    after = check(G, "추론 후")

    inferred = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("inferred")]
    print(f"\n추론으로 추가된 엣지 {len(inferred)}개 (일부):")
    for u, v, d in inferred[:6]:
        print(f"  ({u}, {d['relation']}, {v})  [inferred]")

    client = get_client()
    if client is not None and after and not before:
        print("\n추론 후 그래프로 답변:")
        print(graph_qa(client, G, question, "영업1팀").strip()[:300])
    print("""
기록 포인트:
- "상위조직"에 transitive=True 한 줄을 정의한 것만으로, 쓴 적 없는 사실
  (영업1팀→본사)이 유도됐다 — SPARQL의 `상위조직*` 경로 질의와 같은 효과.
- inferred=True 꼬리표를 유지하는 이유: 답변에 "명시된 사실"과 "추론된 사실"을
  구분 표기할 수 있어야 신뢰를 준다 (5회차 출처 표기 원칙의 그래프판).""")


if __name__ == "__main__":
    run_tasks("13회차: OntoRAG", {
        1: ("온톨로지 설계 — LLM 초안 → 수동 개선", task1),
        2: ("제약 추출 vs 자유 추출: 통과율·어휘 수 비교", task2),
        3: ("transitive 추론 전/후 답변 가능성 사례", task3),
    })
