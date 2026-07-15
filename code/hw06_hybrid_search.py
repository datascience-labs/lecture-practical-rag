# hw06_hybrid_search.py — 6회차 과제 해답
#
# 과제 1: 중간 프로젝트에 HybridRetriever 통합 → 기본 대비 지표 비교
# 과제 2: 5회차 "검색 실패" 질문을 하이브리드로 재실험 → 개선율 기록
# 과제 3: (심화) RRF k값(10 / 60 / 200)에 따른 결과 변화 관찰
#
# API 키 불필요.
#
# 실행: python hw06_hybrid_search.py [--task 1 2 3] [--list]

import json

from bm25_search import tokenize
from corpus import DATA_DIR, ensure_indexed, load_corpus
from eval_harness import run_eval
from hw_common import print_table, run_tasks
from hybrid_search import HybridRetriever, rrf_fuse
from rag_pipeline import SimpleRAG
from retrieval_metrics import hit_at_k


def _setup():
    rag = SimpleRAG()
    ensure_indexed(rag)
    corpus = load_corpus()
    docs = {c["id"]: c["text"] for c in corpus}
    srcs = {c["id"]: c["source"] for c in corpus}
    eval_set = [c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
                if c["answer_chunk_ids"]]
    return rag, docs, srcs, eval_set


def task1(args):
    """하이브리드 통합 전/후 지표 비교 — 통합 자체가 과제의 핵심."""
    rag, docs, srcs, eval_set = _setup()
    hybrid = HybridRetriever(rag.collection, docs, srcs)

    rows = []
    for name, system in [("기본(벡터만)", rag), ("하이브리드", hybrid)]:
        r = run_eval(system, eval_set)
        rows.append([name, f"{r['hit@3']:.0%}", f"{r['mrr']:.3f}",
                     f"{r['latency'] * 1000:.0f}ms"])
    print_table(["설정", "Hit@3", "MRR", "평균 지연"], rows)
    print("""
통합 방법 (중간 프로젝트에 적용할 때):
  hybrid = HybridRetriever(rag.collection, docs, srcs)
  # SimpleRAG.retrieve 대신 hybrid.search를 쓰면 끝 — 인터페이스가 같다 (retrieve 별칭)
  # 답변 생성까지 연결하려면 rag._build_prompt(question, hybrid.search(q)) 재사용""")


def task2(args):
    """5회차 검색 실패 문항만 하이브리드로 재실험 → 개선율."""
    rag, docs, srcs, eval_set = _setup()
    hybrid = HybridRetriever(rag.collection, docs, srcs)

    failures = []
    for case in eval_set:                       # 5회차 방식(벡터 top-3)으로 실패 수집
        ids = [c["id"] for c in rag.retrieve(case["question"], k=3)]
        if not hit_at_k(ids, set(case["answer_chunk_ids"]), 3):
            failures.append(case)

    if not failures:
        print("기본 파이프라인의 검색 실패가 없다 — 평가셋을 더 어렵게 만들어라.")
        return

    fixed = 0
    rows = []
    for case in failures:
        ids = [c["id"] for c in hybrid.search(case["question"], k=3)]
        ok = hit_at_k(ids, set(case["answer_chunk_ids"]), 3)
        fixed += ok
        rows.append([case["id"], case["question"][:28], case["type"],
                     "✓ 개선" if ok else "✗ 여전히 실패"])
    print_table(["문항", "질문", "유형", "하이브리드 결과"], rows)
    print(f"\n개선율: {fixed}/{len(failures)} "
          f"({fixed / len(failures):.0%}) — 5회차 검색 실패 중 하이브리드로 해결된 비율")
    print("""
기록 포인트:
- 개선된 문항은 대부분 키워드형(제품 코드·고유명사) — BM25가 잡아준 것.
- 여전히 실패하는 문항의 유형을 보라: multi-hop이면 하이브리드로는 해결 불가
  (8회차 쿼리 분해, 10회차 그래프의 근거가 된다).""")


def task3(args):
    """(심화) RRF k값 스윕 — k가 순위 융합의 민감도를 조절한다."""
    rag, docs, srcs, eval_set = _setup()

    class TunableHybrid(HybridRetriever):
        """rrf_fuse의 k만 바꿀 수 있는 변형 (검색 로직은 원본과 동일)."""

        def __init__(self, *a, rrf_k: int = 60, **kw):
            super().__init__(*a, **kw)
            self.rrf_k = rrf_k

        def search(self, query: str, k: int = 4) -> list[dict]:
            n = min(k * 3, len(self.doc_ids))
            vec_ranking = self.collection.query(query_texts=[query], n_results=n)["ids"][0]
            scores = self.bm25.get_scores(tokenize(query))
            bm25_ranking = [self.doc_ids[i]
                            for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
                            if scores[i] > 0]
            fused = rrf_fuse([vec_ranking, bm25_ranking], k=self.rrf_k)[:k]
            return [{"id": i, "text": self.doc_texts[i],
                     "source": self.sources.get(i, "")} for i in fused]

        retrieve = search

    rows = []
    for k_val in (10, 60, 200):
        hybrid = TunableHybrid(rag.collection, docs, srcs, rrf_k=k_val)
        r = run_eval(hybrid, eval_set)
        rows.append([k_val, f"{r['hit@3']:.0%}", f"{r['mrr']:.3f}"])
    print_table(["RRF k", "Hit@3", "MRR"], rows)
    print("""
해석:
- k가 작으면(10) 각 검색기의 1~2위에 점수가 집중 → 상위권 불일치에 민감.
- k가 크면(200) 순위 차이가 완만해져 두 검색기의 '합의'가 중요해진다.
- 42청크 코퍼스에서는 차이가 작게 나오는 것이 정상 — 문서가 수천 개로 늘면
  k의 영향이 커진다. 관례값 60이 무난한 이유를 눈으로 확인하는 것이 목적.""")


if __name__ == "__main__":
    run_tasks("6회차: 하이브리드 검색", {
        1: ("HybridRetriever 통합 — 기본 대비 Hit@3/MRR", task1),
        2: ("5회차 검색 실패 문항 재실험 → 개선율", task2),
        3: ("(심화) RRF k값(10/60/200) 스윕", task3),
    })
