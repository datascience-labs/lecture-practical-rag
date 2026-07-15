# eval_harness.py — 9회차 실습: 평가 하네스
#
# 설정 변경(하이브리드, 리랭킹...)의 효과를 평가셋으로 한 번에 비교한다.
#   변경 → 평가셋 실행 → 지표 비교 → 채택/폐기   (이 루프를 자동화하는 것이 핵심)
#
# 실행: python eval_harness.py   (API 키 불필요 — 검색 지표만 측정)
#   기본 / 하이브리드 / 하이브리드+리랭크 3단계를 Hit@3, MRR, 지연시간으로 비교.

import json
import time

from hybrid_search import HybridRetriever
from retrieval_metrics import hit_at_k, mrr


def run_eval(rag_system, eval_set: list[dict], k: int = 3) -> dict:
    """rag_system: retrieve(question, k) -> [{"id": ...}, ...] 를 제공하는 객체."""
    metrics = {"hit@3": [], "mrr": [], "latency": []}
    for case in eval_set:
        t0 = time.time()
        chunks = rag_system.retrieve(case["question"], k=k)
        metrics["latency"].append(time.time() - t0)
        ids = [c["id"] for c in chunks]
        answers = set(case["answer_chunk_ids"])
        metrics["hit@3"].append(hit_at_k(ids, answers, k))
        metrics["mrr"].append(mrr(ids, answers))
    return {m: sum(v) / len(v) for m, v in metrics.items()}


class HybridRerankedRAG:
    """하이브리드 검색으로 후보를 넓게 → Cross-encoder로 재정렬 (6+7회차 결합)."""

    def __init__(self, hybrid: HybridRetriever, candidates: int = 20):
        from sentence_transformers import CrossEncoder   # 사용 시점에만 로딩

        self.hybrid = hybrid
        self.candidates = candidates
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def retrieve(self, question: str, k: int = 4) -> list[dict]:
        chunks = self.hybrid.search(question, k=self.candidates)
        scores = self.reranker.predict([(question, c["text"]) for c in chunks])
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        chunks.sort(key=lambda c: -c["rerank_score"])
        return chunks[:k]


if __name__ == "__main__":
    from corpus import DATA_DIR, ensure_indexed, load_corpus
    from rag_pipeline import SimpleRAG

    # unanswerable 문항(정답 청크 없음)은 생성 평가용이므로 검색 지표에서 제외
    eval_set = [c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
                if c["answer_chunk_ids"]]

    baseline = SimpleRAG()
    ensure_indexed(baseline)

    corpus = load_corpus()
    docs = {c["id"]: c["text"] for c in corpus}
    srcs = {c["id"]: c["source"] for c in corpus}
    hybrid = HybridRetriever(baseline.collection, docs, srcs)
    full = HybridRerankedRAG(hybrid)

    print(f"평가셋 {len(eval_set)}문항 (data/eval_set.json)\n")
    results = {}
    for name, system in [("기본", baseline), ("하이브리드", hybrid),
                         ("하이브리드+리랭크", full)]:
        r = run_eval(system, eval_set)
        results[name] = r
        print(f"{name:>12}: Hit@3={r['hit@3']:.0%}  MRR={r['mrr']:.3f}  "
              f"지연={r['latency'] * 1000:.0f}ms")

    print("""
읽는 법:
- 하이브리드의 Hit 상승 → 키워드형 질문이 개선됐는지 오답 사례를 확인하라
- 리랭크의 MRR 상승 → 정답이 1위로 올라옴. 대신 지연시간 증가 (트레이드오프)
- 집계 점수만 보지 말고 실패 사례를 읽어라 — 개선 방향은 사례에서 나온다
- multi-hop 실패가 많다면 → GraphRAG(10~12회차) 도입의 근거
""")

    # 실패 사례 출력 — "집계 점수만 보지 말라"의 실천
    print("--- [하이브리드+리랭크] 실패 사례 (Hit@3 = 0) ---")
    misses = 0
    for case in eval_set:
        ids = [c["id"] for c in full.retrieve(case["question"], k=3)]
        if not hit_at_k(ids, set(case["answer_chunk_ids"]), 3):
            misses += 1
            print(f"  질문: {case['question']}")
            print(f"    정답: {case['answer_chunk_ids']} / 검색됨: {ids}")
    if not misses:
        print("  없음 — 평가셋을 더 어렵게 만들 차례다 (multi-hop, 전역 질문 추가)")
