# rerank_basic.py — 7회차 실습 1: Cross-encoder 리랭커 기본 사용
#
# Bi-encoder(벡터 검색)는 질문과 문서를 따로 인코딩해 상호작용을 못 보지만,
# Cross-encoder는 [질문 + 문서]를 함께 입력받아 관련도를 직접 채점한다.
#
# 실행: python rerank_basic.py
#   (최초 실행 시 BAAI/bge-reranker-v2-m3 다운로드 — 약 2GB)

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "연차를 이월할 수 없는 경우는?"
candidates = [
    "연차는 다음 해로 이월할 수 있다.",
    "단, 퇴직 예정자는 연차 이월이 불가하다.",
    "재택근무는 주 2회까지 가능하다.",
]

pairs = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)

for doc, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
    print(f"[{s:.4f}] {doc}")

print("""
관찰 포인트:
- "이월할 수 없는"이라는 부정 조건을 이해하고 '불가' 문서를 1위로 올리는가?
- 벡터 검색이었다면 "이월할 수 있다" 문서가 표면적으로 더 유사했을 것이다.
""")
