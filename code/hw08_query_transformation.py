# hw08_query_transformation.py — 8회차 과제 해답
#
# 과제 1: 쿼리 재작성 + Multi-Query 통합, 멀티턴 대화 시나리오 5개 테스트
# 과제 2: HyDE 적용 전/후 Hit@3 비교
# 과제 3: 복합 질문 3개로 쿼리 분해 동작 확인
#
# 세 과제 모두 ANTHROPIC_API_KEY 필요 (쿼리 변환 자체가 LLM 호출).
#
# 실행: python hw08_query_transformation.py [--task 1 2 3] [--list]

import json

from corpus import DATA_DIR, ensure_indexed
from hw_common import get_client, print_table, run_tasks, skip
from hybrid_search import rrf_fuse
from query_transforms import decompose_query, hyde_search, multi_query, rewrite_query
from rag_pipeline import SimpleRAG
from retrieval_metrics import hit_at_k

# (대화 이력, 후속 질문, 정답 청크 id) — 대명사·생략 때문에 원 질문으로는 검색이 어렵다
SCENARIOS = [
    ([("연차 규정 알려줘", "연차 휴가는 매년 15일 부여됩니다.")],
     "그럼 그거 다음 해로 넘길 수 있어?", "handbook-3.1"),
    ([("재택근무 조건이 뭐야?", "주 2회까지 팀장 승인 하에 가능합니다.")],
     "당일에 신청해도 돼?", "handbook-2.3"),
    ([("PD-2000 스펙 알려줘", "최대 출력 500W, 무게 2.3kg입니다.")],
     "그 제품은 왜 단종돼?", "04-product-pd2000"),
    ([("법인카드 규정 알려줘", "증빙은 매월 5일까지 제출합니다.")],
     "개인 용도로 쓰면 어떻게 돼?", "handbook-5.1"),
    ([("국내 출장 다녀오려고 해", "출장 규정은 5.2절에 있습니다.")],
     "하루 숙박비는 얼마까지 나와?", "handbook-5.2"),
]


def task1(args):
    """멀티턴 시나리오 5개 — 재작성 전/후 + Multi-Query의 검색 성공 비교."""
    client = get_client()
    if client is None:
        skip("ANTHROPIC_API_KEY가 필요한 과제다.")
        return
    rag = SimpleRAG()
    ensure_indexed(rag)

    def top3(q: str) -> list[str]:
        return rag.collection.query(query_texts=[q], n_results=3)["ids"][0]

    rows = []
    for turns, question, answer_id in SCENARIOS:
        history = []
        for u, a in turns:
            history += [{"role": "user", "content": u},
                        {"role": "assistant", "content": a}]
        rewritten = rewrite_query(client, history, question)

        raw_ok = answer_id in top3(question)
        rew_ok = answer_id in top3(rewritten)
        # Multi-Query: 재작성 질문을 여러 관점으로 확장 → RRF 융합
        variants = multi_query(client, rewritten)
        fused = rrf_fuse([top3(v) for v in variants])[:3]
        mq_ok = answer_id in fused

        print(f"\n원 질문: {question}")
        print(f"재작성 → {rewritten}")
        rows.append([question[:22], "✓" if raw_ok else "✗",
                     "✓" if rew_ok else "✗", "✓" if mq_ok else "✗"])

    print()
    print_table(["후속 질문", "원문 Hit@3", "재작성 Hit@3", "+MultiQuery"], rows)
    print("""
기록 포인트:
- "그거", "그 제품" 같은 대명사는 임베딩이 해석하지 못한다 — 재작성이 대명사를
  실제 개체명으로 치환해야 검색이 산다 (멀티턴 챗봇의 필수 전처리).
- Multi-Query는 재현율 보강 — 재작성만으로 이미 잡히는 질문에는 이득이 없고
  지연·비용만 는다 → adaptive_rag.py처럼 조건부 적용이 실무 정답.""")


def task2(args):
    """HyDE 전/후 Hit@3 — 본인 데이터에서 효과가 있는지 숫자로 확인."""
    client = get_client()
    if client is None:
        skip("ANTHROPIC_API_KEY가 필요한 과제다.")
        return
    rag = SimpleRAG()
    ensure_indexed(rag)
    eval_set = [c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
                if c["answer_chunk_ids"]]

    plain_hits, hyde_hits = [], []
    rows = []
    for case in eval_set:
        answers = set(case["answer_chunk_ids"])
        plain = rag.collection.query(query_texts=[case["question"]], n_results=3)["ids"][0]
        hyde = hyde_search(client, rag.collection, case["question"], k=3)["ids"][0]
        p, h = hit_at_k(plain, answers, 3), hit_at_k(hyde, answers, 3)
        plain_hits.append(p)
        hyde_hits.append(h)
        if p != h:
            rows.append([case["id"], case["question"][:26],
                         "✓" if p else "✗", "✓" if h else "✗"])

    n = len(eval_set)
    print(f"평가셋 {n}문항:  일반 검색 Hit@3={sum(plain_hits) / n:.0%}  /  "
          f"HyDE Hit@3={sum(hyde_hits) / n:.0%}")
    if rows:
        print("\n결과가 갈린 문항:")
        print_table(["문항", "질문", "일반", "HyDE"], rows)
    print("""
결론 가이드:
- HyDE는 "질문(짧은 의문문) ↔ 문서(긴 설명문)의 형태 차이"가 클 때 효과적이다.
- 이 코퍼스처럼 청크가 짧고 규정형이면 효과가 작거나 역효과일 수 있다 — 그것이
  정상이며, "측정해보고 채택/폐기"가 이 과제의 진짜 목적 (9회차로 이어진다).
- 비용 주의: 질의마다 LLM 호출 1회 추가 — 채택하려면 지연·비용도 함께 기록하라.""")


def task3(args):
    """복합 질문 3개 분해 — 하위 질문 검색의 합집합이 정답을 커버하는가."""
    client = get_client()
    if client is None:
        skip("ANTHROPIC_API_KEY가 필요한 과제다.")
        return
    rag = SimpleRAG()
    ensure_indexed(rag)

    eval_set = {c["id"]: c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))}
    complex_cases = [eval_set["cd-002"], eval_set["hb-013"], eval_set["hb-014"]]

    def top_ids(q: str, k: int) -> list[str]:
        return rag.collection.query(query_texts=[q], n_results=k)["ids"][0]

    for case in complex_cases:
        answers = set(case["answer_chunk_ids"])
        print(f"\n복합 질문: {case['question']}  (정답 청크 {sorted(answers)})")
        subs = decompose_query(client, case["question"])
        union = set(top_ids(case["question"], 3))          # 직접 검색
        direct_recall = len(answers & union) / len(answers)
        sub_union = set()
        for s in subs:
            hits = top_ids(s, 2)
            sub_union.update(hits)
            print(f"  하위: {s}  → {hits}")
        sub_recall = len(answers & sub_union) / len(answers)
        print(f"  정답 커버율: 직접 검색 {direct_recall:.0%} → 분해 후 합집합 {sub_recall:.0%}")

    print("""
기록 포인트:
- 비교·multi-hop 질문은 "질문 전체"와 비슷한 청크가 없어 직접 검색이 정답 청크의
  일부만 가져온다 — 분해가 각 하위 질문을 해당 청크에 정조준시킨다.
- 하위 질문 수 × 검색 = 지연 증가. adaptive_rag.py처럼 complex로 분류된 질문에만
  적용하라. 분해로도 안 되는 관계 추론은 10회차 지식 그래프의 몫이다.""")


if __name__ == "__main__":
    run_tasks("8회차: 쿼리 변환", {
        1: ("재작성+Multi-Query 통합, 멀티턴 시나리오 5개", task1),
        2: ("HyDE 적용 전/후 Hit@3 비교", task2),
        3: ("복합 질문 3개 쿼리 분해 동작 확인", task3),
    })
