# hw09_evaluation.py — 9회차 과제(중간 점검) 해답
#
# 과제 1: 평가셋 20문항 구축 (자동 생성 + 검수)
# 과제 2: 기본 / +하이브리드 / +리랭킹 / +쿼리변환 4단계 비교표
# 과제 3: 실패 사례 유형별 분류 — multi-hop/전역 질문 비율 확인 (10회차 예고)
#
# 과제 2의 처음 3단계와 과제 3은 API 키 없이 동작한다.
# 쿼리변환 단계와 과제 1의 자동 생성 시연만 키가 필요하다.
#
# 실행: python hw09_evaluation.py [--task 1 2 3] [--list]

import json
import random
from collections import Counter

from corpus import DATA_DIR, ensure_indexed, load_corpus
from eval_harness import HybridRerankedRAG, run_eval
from hw_common import get_client, print_table, run_tasks, save_cache, skip
from hybrid_search import HybridRetriever, rrf_fuse
from rag_pipeline import SimpleRAG
from retrieval_metrics import hit_at_k, recall_at_k


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _setup():
    rag = SimpleRAG()
    ensure_indexed(rag)
    corpus = load_corpus()
    docs = {c["id"]: c["text"] for c in corpus}
    srcs = {c["id"]: c["source"] for c in corpus}
    hybrid = HybridRetriever(rag.collection, docs, srcs)
    return rag, hybrid


def task1(args):
    """평가셋 20문항 구축 — 기존 검수본 확인 + 자동 생성 초안 시연."""
    eval_set = _load("eval_set.json")
    types = Counter(c["type"] for c in eval_set)
    print(f"검수 완료 평가셋: data/eval_set.json — {len(eval_set)}문항 (요구: 20문항 이상)")
    print(f"유형 분포: {dict(types)}")
    print("""
검수 체크리스트 (자동 생성 문항에 반드시 적용):
  1. 문서를 안 본 사용자가 실제로 할 법한 질문인가? ("3.1절의 두 번째 문장은?" ✗)
  2. ground_truth가 청크 내용과 정확히 일치하는가? (수치·조건 왜곡 없음)
  3. answer_chunk_ids가 실제 정답 청크를 가리키는가? (키워드 오탐 확인)
  4. unanswerable 문항이 포함돼 있는가? (환각 검출용 — hb-015가 그 예)
  5. 유형(type) 라벨이 맞는가? — 과제 3의 실패 분류가 이 라벨에 의존한다""")

    client = get_client()
    if client is None:
        skip("자동 생성 시연은 ANTHROPIC_API_KEY가 필요하다 (없어도 과제 성립).")
        return
    from generate_evalset import generate_qa_from_chunk

    print("\n--- 자동 생성 초안 시연 (3문항) — ⚠️ 검수 전에는 평가에 쓰지 말 것 ---")
    corpus = load_corpus()
    random.seed(7)
    drafts = []
    for c in random.sample(corpus, 3):
        try:
            qa = generate_qa_from_chunk(client, c["id"], c["text"])
        except Exception as e:
            print(f"  건너뜀 ({c['id']}): {e}")
            continue
        drafts.append(qa)
        print(f"  [{c['id']}] Q: {qa['question']}\n{'':13}A: {qa['ground_truth'][:60]}...")
    save_cache("hw09_generated_draft", drafts)
    print("  → hw_cache/hw09_generated_draft.json 저장. 체크리스트로 검수 후"
          " data/eval_set.json에 병합하는 것이 원래 워크플로다.")


def task2(args):
    """4단계 비교표 — 기법을 쌓을 때마다 지표가 어떻게 움직이는가."""
    rag, hybrid = _setup()
    eval_set = [c for c in _load("eval_set.json") if c["answer_chunk_ids"]]
    full = HybridRerankedRAG(hybrid)

    systems = [("① 기본", rag), ("② +하이브리드", hybrid), ("③ +리랭킹", full)]

    client = get_client()
    if client is not None:
        from query_transforms import multi_query

        class QueryTransformRAG:
            """④ Multi-Query로 확장한 질의들을 하이브리드+리랭크에 융합."""

            def retrieve(self, question: str, k: int = 3) -> list[dict]:
                variants = multi_query(client, question)
                rankings, pool = [], {}
                for v in variants:
                    chunks = full.retrieve(v, k=k)
                    rankings.append([c["id"] for c in chunks])
                    pool.update({c["id"]: c for c in chunks})
                return [pool[cid] for cid in rrf_fuse(rankings)[:k]]

        systems.append(("④ +쿼리변환", QueryTransformRAG()))
    else:
        skip("④ 쿼리변환 단계는 ANTHROPIC_API_KEY가 필요해 표에서 제외한다.")

    rows = []
    for name, system in systems:
        r = run_eval(system, eval_set)
        rows.append([name, f"{r['hit@3']:.0%}", f"{r['mrr']:.3f}",
                     f"{r['latency'] * 1000:.0f}ms"])
    print(f"평가셋 {len(eval_set)}문항 (unanswerable 제외)\n")
    print_table(["단계", "Hit@3", "MRR", "평균 지연"], rows)
    print("""
읽는 법:
- 단계마다 "지표 상승분 ↔ 지연 증가분"을 함께 보라 — 채택은 트레이드오프 결정이다.
- ④가 ③보다 나빠질 수도 있다: 이미 검색이 잘 되는 코퍼스에서 변형 질의는
  노이즈가 되기도 한다. 그 관찰 자체가 "조건부 적용(adaptive)"의 근거.
- 이 표를 본인 프로젝트 평가셋으로 다시 만들어라 — 숫자는 코퍼스마다 다르다.""")


def task3(args):
    """실패 유형 분류 — multi-hop/전역 질문 비율로 10~12회차 도입 근거 만들기."""
    rag, hybrid = _setup()
    full = HybridRerankedRAG(hybrid)

    # (1) 기본 평가셋에서 최강 파이프라인(③)의 실패를 유형별로 분류
    eval_set = [c for c in _load("eval_set.json") if c["answer_chunk_ids"]]
    fail_types = Counter()
    for case in eval_set:
        ids = [c["id"] for c in full.retrieve(case["question"], k=3)]
        if not hit_at_k(ids, set(case["answer_chunk_ids"]), 3):
            fail_types[case["type"]] += 1
            print(f"  실패: [{case['type']}] {case['question']}")
    total_fail = sum(fail_types.values())
    print(f"\neval_set.json 실패 {total_fail}건 — 유형: {dict(fail_types)}")

    # (2) 그래프 평가셋(multi-hop·전역)은 벡터 파이프라인이 구조적으로 약하다
    graph_set = _load("eval_set_graph.json")
    rows = []
    per_type = {}
    for case in graph_set:
        # 회사 문서는 파일명 = 청크 id (corpus.py) → source_docs가 곧 정답 청크
        answers = {d.removesuffix(".txt") for d in case["source_docs"]}
        ids = [c["id"] for c in full.retrieve(case["question"], k=3)]
        hit = hit_at_k(ids, answers, 3)
        rec = recall_at_k(ids, answers, 3)
        per_type.setdefault(case["type"], []).append(rec)
        rows.append([case["id"], case["type"], "✓" if hit else "✗", f"{rec:.0%}"])
    print("\neval_set_graph.json (multi-hop·전역 질문) — 하이브리드+리랭크로 검색:")
    print_table(["문항", "유형", "Hit@3", "정답문서 Recall@3"], rows)

    hard = [t for t in per_type if t in ("multi-hop", "global", "comparison")]
    hard_n = sum(len(per_type[t]) for t in hard)
    print(f"""
결론 (10회차 예고의 근거):
- multi-hop·전역·비교 질문이 그래프 평가셋의 {hard_n}/{len(graph_set)}건 — 이 유형은
  Hit@3이 아니라 Recall(정답 문서를 '전부' 모았는가)이 본질인데, 벡터 검색은
  질문 전체와 비슷한 청크만 가져와 나머지 근거를 놓친다.
- 실패의 다수가 이 유형이라면 검색기 튜닝(6~8회차)이 아니라 **자료구조 교체**
  (10회차 지식 그래프 → 11~12회차 GraphRAG)가 올바른 다음 수다.""")


if __name__ == "__main__":
    run_tasks("9회차: RAG 평가 (중간 점검)", {
        1: ("평가셋 20문항 구축 — 검수 체크리스트 + 자동 생성 시연", task1),
        2: ("기본/+하이브리드/+리랭킹/+쿼리변환 4단계 비교표", task2),
        3: ("실패 유형 분류 — multi-hop/전역 비율 확인", task3),
    })
