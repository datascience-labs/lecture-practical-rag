# bm25_search.py — 6회차 실습 1: 한국어 BM25 키워드 검색
#
# 한국어는 조사·어미 때문에 공백 분리로는 "재택근무를"과 "재택근무"가 매칭되지
# 않는다 → kiwipiepy 형태소 분석으로 내용어만 추출한 뒤 BM25를 적용한다.
#
# 실행: python bm25_search.py

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    """내용어만 추출: 체언(N*)·용언(V*)·외국어(SL)·숫자(SN)·한자(SH).

    조사(J*)·어미(E*)·기호는 버린다 — "PD-2000"은 ["PD", "2000"]이 되어
    질문과 문서 양쪽에서 같은 토큰으로 매칭된다.
    """
    return [t.form for t in kiwi.tokenize(text)
            if t.tag.startswith(("N", "V")) or t.tag in ("SL", "SN", "SH")]


if __name__ == "__main__":
    from corpus import load_corpus

    print("--- 토큰화 확인 ---")
    print('"재택근무를 신청하려면" →', tokenize("재택근무를 신청하려면"))
    print('"PD-2000 스펙"        →', tokenize("PD-2000 스펙"))

    # 실제 코퍼스(42청크)로 검색 — BM25의 IDF는 문서가 충분해야 의미가 있다.
    # (문서 4~5개짜리 장난감 예제에서는 흔한 단어의 IDF가 0이 되어 전부 0점이 나온다)
    corpus = load_corpus()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in corpus])

    for query in ["PD-2000 스펙", "집에서 일해도 되나요?"]:
        print(f"\n--- 질문: {query}")
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])[:3]
        for c, s in ranked:
            first_line = c["text"][:56].replace("\n", " ")
            print(f"  [{s:5.2f}] ({c['id']}) {first_line}...")

    print("""
관찰 포인트:
- "PD-2000 스펙" → PD-2000이 포함된 문서가 정확히 상위에 랭크된다 (키워드의 힘)
- "집에서 일해도 되나요?" → 재택근무 청크와 겹치는 단어가 없어 BM25는 헛발질
  (의미 검색이 필요한 이유)
→ 두 검색을 결합하는 하이브리드 검색으로: python hybrid_search.py
""")
