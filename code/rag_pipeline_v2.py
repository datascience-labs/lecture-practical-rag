# rag_pipeline_v2.py — 7회차 실습 2·3: 2단계 검색 (Retrieve → Rerank)
#
# 1단계: 벡터 검색으로 후보를 넓게(candidates=20) 뽑고   ← 목표: Recall
# 2단계: Cross-encoder로 재정렬해 top-k만 남긴다        ← 목표: Precision
#
# 실행: python rag_pipeline_v2.py   (리랭킹 전/후 Hit@3·MRR 비교 — API 키 불필요)

from sentence_transformers import CrossEncoder

from rag_pipeline import SimpleRAG


class RerankedRAG(SimpleRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def retrieve(self, question: str, k: int = 4,
                 candidates: int = 20, min_score: float = 0.1) -> list[dict]:
        # 1단계: 넓게 검색
        n = min(candidates, self.collection.count())
        r = self.collection.query(query_texts=[question], n_results=n)
        chunks = [{"id": cid, "text": d, "source": m["source"]}
                  for cid, d, m in zip(
                      r["ids"][0], r["documents"][0], r["metadatas"][0])]

        # 2단계: 리랭킹
        pairs = [(question, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        chunks.sort(key=lambda c: -c["rerank_score"])
        # 점수 임계값: 관련 없으면 과감히 버린다
        return [c for c in chunks[:k] if c["rerank_score"] > min_score]


def llm_rerank(client, query: str, chunks: list[str]) -> list[int]:
    """LLM 기반 리랭킹 — 유연하지만 느리고 비싸다 (과제 3: Cross-encoder와 비교).

    client: anthropic.Anthropic() 인스턴스. 반환: 관련도 순 인덱스 리스트.
    """
    import json

    listing = "\n".join(f"[{i}] {c[:200]}" for i, c in enumerate(chunks))
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        messages=[{"role": "user", "content":
            f"질문: {query}\n\n다음 문서들을 질문과의 관련도 순으로 정렬해, "
            f"인덱스만 JSON 배열로 출력해줘 (예: [2,0,1]).\n\n{listing}"}],
    )
    return json.loads(response.content[0].text)


if __name__ == "__main__":
    import json

    from corpus import DATA_DIR, ensure_indexed
    from retrieval_metrics import hit_at_k, mrr   # 9회차에서 정식으로 다루는 지표

    baseline = SimpleRAG()
    ensure_indexed(baseline)
    reranked = RerankedRAG()

    # --- 단일 질문으로 리랭킹 효과 확인 ---
    q = "연차를 이월할 수 없는 경우는?"
    print(f"질문: {q}\n")
    print("리랭킹 전 (벡터 검색 순위):")
    for c in baseline.retrieve(q, k=3, max_distance=1.0):
        print(f"  [dist {c['distance']:.3f}] ({c['id']}) {c['text'][:52]}...")
    print("리랭킹 후 (Cross-encoder 순위):")
    for c in reranked.retrieve(q, k=3):
        print(f"  [score {c['rerank_score']:.3f}] ({c['id']}) {c['text'][:52]}...")

    # --- 평가셋으로 전/후 비교 (실습 3) ---
    # unanswerable 문항(정답 청크 없음)은 생성 평가용이므로 검색 지표에서 제외
    eval_set = [c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
                if c["answer_chunk_ids"]]

    def run(retrieve_fn, k: int = 3) -> tuple[float, float]:
        hits, mrrs = [], []
        for case in eval_set:
            ids = [c["id"] for c in retrieve_fn(case["question"], k=k)]
            answers = set(case["answer_chunk_ids"])
            hits.append(hit_at_k(ids, answers, k))
            mrrs.append(mrr(ids, answers))
        return sum(hits) / len(hits), sum(mrrs) / len(mrrs)

    h0, m0 = run(baseline.retrieve)
    h1, m1 = run(reranked.retrieve)
    print(f"\n평가셋 {len(eval_set)}문항 (data/eval_set.json):")
    print(f"  리랭킹 전  Hit@3={h0:.0%}  MRR={m0:.3f}")
    print(f"  리랭킹 후  Hit@3={h1:.0%}  MRR={m1:.3f}")
    print("\nMRR이 크게 오르면 → 정답이 후보에는 있었지만 순위가 낮았다는 뜻 (리랭킹의 존재 이유)")
