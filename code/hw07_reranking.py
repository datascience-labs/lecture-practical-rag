# hw07_reranking.py — 7회차 과제 해답
#
# 과제 1: 중간 프로젝트에 리랭킹 통합 → Hit@3 전/후 비교표
# 과제 2: candidates 수(10 / 20 / 50)에 따른 정확도·지연시간 측정
# 과제 3: (심화) LLM 리랭킹 vs Cross-encoder 리랭킹 비교  (API 키 필요)
#
# 실행: python hw07_reranking.py [--task 1 2 3] [--list]

import json
import time

from corpus import DATA_DIR, ensure_indexed
from hw_common import get_client, print_table, run_tasks, skip
from rag_pipeline import SimpleRAG
from rag_pipeline_v2 import RerankedRAG, llm_rerank
from retrieval_metrics import hit_at_k, mrr


def _eval_set() -> list[dict]:
    return [c for c in
            json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
            if c["answer_chunk_ids"]]


def _run(retrieve_fn, eval_set, k=3, **kwargs) -> dict:
    hits, mrrs, lat = [], [], []
    for case in eval_set:
        t0 = time.time()
        ids = [c["id"] for c in retrieve_fn(case["question"], k=k, **kwargs)]
        lat.append(time.time() - t0)
        answers = set(case["answer_chunk_ids"])
        hits.append(hit_at_k(ids, answers, k))
        mrrs.append(mrr(ids, answers))
    n = len(eval_set)
    return {"hit": sum(hits) / n, "mrr": sum(mrrs) / n, "lat": sum(lat) / n}


def task1(args):
    """리랭킹 통합 전/후 비교표."""
    baseline = SimpleRAG()
    ensure_indexed(baseline)
    reranked = RerankedRAG()
    eval_set = _eval_set()

    rows = []
    for name, fn in [("리랭킹 전 (벡터)", baseline.retrieve),
                     ("리랭킹 후 (CE)", reranked.retrieve)]:
        r = _run(fn, eval_set)
        rows.append([name, f"{r['hit']:.0%}", f"{r['mrr']:.3f}", f"{r['lat'] * 1000:.0f}ms"])
    print_table(["설정", "Hit@3", "MRR", "평균 지연"], rows)
    print("""
읽는 법:
- Hit@3보다 MRR이 크게 오르는 것이 전형적 — 정답이 후보에는 있었지만
  순위가 낮았던 문항을 리랭커가 1위로 끌어올린다 (리랭킹의 존재 이유).
- 통합 방법: SimpleRAG → RerankedRAG로 클래스만 교체 (retrieve 시그니처 호환).""")


def task2(args):
    """candidates 스윕 — recall과 지연시간의 트레이드오프."""
    reranked = RerankedRAG()
    ensure_indexed(reranked)
    eval_set = _eval_set()

    rows = []
    for cand in (10, 20, 50):
        r = _run(reranked.retrieve, eval_set, candidates=cand)
        rows.append([cand, f"{r['hit']:.0%}", f"{r['mrr']:.3f}", f"{r['lat'] * 1000:.0f}ms"])
    print_table(["candidates", "Hit@3", "MRR", "평균 지연"], rows)
    print(f"""
해석 (코퍼스 42청크 기준):
- candidates=50은 사실상 전체 코퍼스 리랭킹 — recall 손실은 없지만 지연이 최대.
- candidates가 너무 작으면 1단계(벡터)가 정답을 후보에 못 넣어 리랭커가 무력화된다.
- 실무 규칙: "1단계 Hit@candidates ≈ 100%가 되는 최소값"을 골라라 —
  9회차 평가 하네스로 candidates별 1단계 recall을 재면 근거가 생긴다.""")


def task3(args):
    """(심화) LLM 리랭킹 vs Cross-encoder — 정확도와 지연시간."""
    client = get_client()
    if client is None:
        skip("LLM 리랭킹은 ANTHROPIC_API_KEY가 필요하다.")
        return

    reranked = RerankedRAG()
    ensure_indexed(reranked)
    eval_set = _eval_set()[:10]                 # 비용 절약: 10문항 표본

    ce_hits, llm_hits, ce_lat, llm_lat, errors = [], [], [], [], 0
    for case in eval_set:
        r = reranked.collection.query(query_texts=[case["question"]], n_results=10)
        cand_ids, cand_texts = r["ids"][0], r["documents"][0]
        answers = set(case["answer_chunk_ids"])

        t0 = time.time()                        # Cross-encoder
        scores = reranked.reranker.predict([(case["question"], t) for t in cand_texts])
        ce_lat.append(time.time() - t0)
        order = sorted(range(len(cand_ids)), key=lambda i: -scores[i])
        ce_hits.append(hit_at_k([cand_ids[i] for i in order], answers, 3))

        t0 = time.time()                        # LLM 리랭킹
        try:
            order = llm_rerank(client, case["question"], cand_texts)
            llm_hits.append(hit_at_k([cand_ids[i] for i in order[:3]], answers, 3))
        except Exception:                       # JSON 파싱 실패 등 — LLM 방식의 약점
            errors += 1
            llm_hits.append(0.0)
        llm_lat.append(time.time() - t0)

    n = len(eval_set)
    print_table(["방식", "Hit@3", "평균 지연", "실패"], [
        ["Cross-encoder", f"{sum(ce_hits) / n:.0%}", f"{sum(ce_lat) / n * 1000:.0f}ms", "0"],
        ["LLM 리랭킹", f"{sum(llm_hits) / n:.0%}", f"{sum(llm_lat) / n * 1000:.0f}ms",
         f"{errors} (파싱)"],
    ])
    print("""
결론 가이드:
- Cross-encoder: 빠르고(수십 ms) 안정적, 점수가 재현 가능 — 기본 선택.
- LLM 리랭킹: 지시문으로 기준을 바꿀 수 있는 유연함이 강점이지만 느리고 비싸고
  출력 파싱 실패가 발생한다 — "최신 문서 우선" 같은 복잡한 기준이 필요할 때만.""")


if __name__ == "__main__":
    run_tasks("7회차: 리랭킹", {
        1: ("리랭킹 통합 — Hit@3 전/후 비교표", task1),
        2: ("candidates(10/20/50)별 정확도·지연 측정", task2),
        3: ("(심화) LLM vs Cross-encoder 리랭킹 비교", task3),
    })
