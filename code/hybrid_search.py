# hybrid_search.py — 6회차 실습 2·3: RRF 융합과 하이브리드 검색기
#
# 벡터 검색(의미)과 BM25(키워드)의 순위 리스트를 RRF로 융합한다.
# 점수 체계가 다른 두 검색을 "점수가 아닌 순위"로 합치므로 정규화가 필요 없다.
#
# 실행: python hybrid_search.py   (벡터 vs BM25 vs 하이브리드 비교 실험)

from rank_bm25 import BM25Okapi

from bm25_search import tokenize


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings: 각 검색기가 반환한 문서 ID 리스트 (순위순). k=60은 관례."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, collection, documents: dict[str, str],
                 sources: dict[str, str] | None = None):
        self.collection = collection            # ChromaDB (벡터)
        self.doc_ids = list(documents.keys())
        self.doc_texts = documents
        self.sources = sources or {}
        self.bm25 = BM25Okapi([tokenize(t) for t in documents.values()])

    def search(self, query: str, k: int = 4) -> list[dict]:
        n = min(k * 3, len(self.doc_ids))       # 후보는 넉넉히 뽑는다

        # 1) 벡터 검색
        vec = self.collection.query(query_texts=[query], n_results=n)
        vec_ranking = vec["ids"][0]

        # 2) BM25 검색 (점수 0 = 겹치는 단어 없음 → 후보에서 제외)
        scores = self.bm25.get_scores(tokenize(query))
        bm25_ranking = [self.doc_ids[i]
                        for i in sorted(range(len(scores)),
                                        key=lambda i: -scores[i])[:n]
                        if scores[i] > 0]

        # 3) RRF 융합
        fused = rrf_fuse([vec_ranking, bm25_ranking])[:k]
        return [{"id": i, "text": self.doc_texts[i],
                 "source": self.sources.get(i, "")} for i in fused]

    retrieve = search    # 9회차 평가 하네스와 인터페이스 통일


if __name__ == "__main__":
    from corpus import ensure_indexed, load_corpus
    from rag_pipeline import SimpleRAG

    rag = SimpleRAG()
    ensure_indexed(rag)

    corpus = load_corpus()
    docs = {c["id"]: c["text"] for c in corpus}
    srcs = {c["id"]: c["source"] for c in corpus}
    hybrid = HybridRetriever(rag.collection, docs, srcs)
    bm25_only = BM25Okapi([tokenize(t) for t in docs.values()])
    doc_ids = list(docs.keys())

    test_cases = [
        ("PD-2000 스펙 알려줘",        "키워드형 질문"),
        ("집에서 일해도 되나요?",        "의미형 질문"),
        ("PD 시리즈 신형은 뭐가 좋아?",  "혼합형 질문"),
    ]

    def snippet(text: str) -> str:
        return text[:56].replace("\n", " ") + "..."

    for query, qtype in test_cases:
        print(f"\n=== [{qtype}] {query}")
        vec = rag.collection.query(query_texts=[query], n_results=1)
        print(f"      벡터: ({vec['ids'][0][0]}) {snippet(vec['documents'][0][0])}")
        s = bm25_only.get_scores(tokenize(query))
        top = max(range(len(s)), key=lambda i: s[i])
        hit = f"({doc_ids[top]}) {snippet(docs[doc_ids[top]])}" if s[top] > 0 else "(매칭 없음)"
        print(f"      BM25: {hit}")
        h = hybrid.search(query, k=1)[0]
        print(f"  하이브리드: ({h['id']}) {snippet(h['text'])}")

    print("""
기대 결과: 키워드형 → BM25 승 / 의미형 → 벡터 승 /
하이브리드는 양쪽 모두에서 준수 ← 이것이 하이브리드의 핵심 가치
""")
