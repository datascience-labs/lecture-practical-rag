# retrieval_metrics.py — 9회차: 검색 품질 지표
#
# "검색과 생성을 분리해서 측정하라" — 이 파일은 검색 쪽 지표.
# 정답 청크 id 라벨(data/eval_set.json의 answer_chunk_ids)이 필요하다.
#
# 실행: python retrieval_metrics.py   (자체 테스트)


def hit_at_k(retrieved_ids: list[str], answer_ids: set[str], k: int) -> float:
    """정답 청크가 top-k에 하나라도 있으면 1 — 'LLM에 정답이 들어가는가' (가장 실용적)."""
    return float(any(rid in answer_ids for rid in retrieved_ids[:k]))


def recall_at_k(retrieved_ids: list[str], answer_ids: set[str], k: int) -> float:
    """정답 청크 중 top-k에 포함된 비율 (정답이 여러 개일 때)."""
    found = sum(1 for aid in answer_ids if aid in retrieved_ids[:k])
    return found / len(answer_ids)


def mrr(retrieved_ids: list[str], answer_ids: set[str]) -> float:
    """첫 정답의 역순위 — 1위면 1.0, 2위면 0.5, 3위면 0.33... (리랭킹 효과 측정에 적합)."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in answer_ids:
            return 1 / rank
    return 0.0


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d"]

    assert hit_at_k(retrieved, {"b"}, k=3) == 1.0
    assert hit_at_k(retrieved, {"d"}, k=3) == 0.0      # 4위는 top-3 밖
    assert recall_at_k(retrieved, {"a", "c", "x"}, k=3) == 2 / 3
    assert mrr(retrieved, {"b"}) == 0.5                # 첫 정답이 2위
    assert mrr(retrieved, {"x"}) == 0.0                # 정답 없음

    print("retrieval_metrics: 모든 자체 테스트 통과")
