# hw03_vector_databases.py — 3회차 과제 해답
#
# 과제 1: 문서 30개 이상으로 컬렉션 구축 (메타데이터 2종 이상)
# 과제 2: 필터링 있음/없음 검색 결과 비교 실험 3가지
# 과제 3: 거리 임계값을 바꿔가며 "관련 없음" 판정 정확도 관찰
#
# 실습 코퍼스 42청크(핸드북 소절 + 회사 문서)를 사용한다. API 키 불필요.
#
# 실행: python hw03_vector_databases.py [--task 1 2 3] [--list]

import json

import chromadb
from chromadb.utils import embedding_functions

from corpus import CHROMA_PATH, DATA_DIR, load_corpus
from hw_common import print_table, run_tasks

COLLECTION = "hw03_docs"


def _metadata(c: dict) -> dict:
    """메타데이터 2종 이상: doc_type(문서 종류) + category(장/주제) (+year)."""
    if c["source"] == "sample_handbook.md":
        chapter = c["id"].split("-")[1].split(".")[0]     # handbook-3.1 → 3
        return {"doc_type": "handbook", "category": f"chapter-{chapter}", "year": 2026}
    category = c["id"].split("-")[1]                      # 04-product-pd2000 → product
    return {"doc_type": "company", "category": category, "year": 2025}


def _collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    return db.get_or_create_collection(COLLECTION, embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})


def _ensure_indexed():
    col = _collection()
    if col.count() == 0:
        docs = load_corpus()
        col.add(ids=[d["id"] for d in docs],
                documents=[d["text"] for d in docs],
                metadatas=[_metadata(d) for d in docs])
        print(f"  {len(docs)}개 청크 인덱싱 완료 (컬렉션 {COLLECTION})")
    return col


def task1(args):
    """컬렉션 구축 — 30개 이상 + 메타데이터 2종 이상 확인."""
    col = _ensure_indexed()
    print(f"컬렉션 {COLLECTION}: 총 {col.count()}개 문서 (요구: 30개 이상)")

    sample = col.get(limit=col.count())
    doc_types, categories = set(), set()
    for m in sample["metadatas"]:
        doc_types.add(m["doc_type"])
        categories.add(m["category"])
    print(f"메타데이터 doc_type: {sorted(doc_types)}")
    print(f"메타데이터 category: {sorted(categories)}")
    print("""
설계 포인트 (3회차 설계 팁의 적용):
- id는 내용 기반(handbook-3.1, 04-product-pd2000) → 재인덱싱해도 안정적
- 메타데이터는 지금 미리 설계 — 나중에 필터 조건이 필요해지면 전체 재삽입해야 한다
- year를 넣어둔 이유: 실무에서 "폐기된 구버전 규정 제외" 필터의 기반""")


def task2(args):
    """필터 있음/없음 비교 3가지 — 필터가 검색 결과를 어떻게 바꾸는지 관찰."""
    col = _ensure_indexed()

    experiments = [
        ("휴가 규정 알려줘", {"doc_type": {"$eq": "handbook"}},
         "규정 질문인데 회사 소개 문서가 섞여 들어오는가?"),
        ("전원장치 제품 정보", {"category": {"$eq": "product"}},
         "제품 스펙만 원할 때 조직/공급사 문서를 차단할 수 있는가?"),
        ("결재는 누가 하나요?", {"doc_type": {"$eq": "company"}},
         "핸드북(경비 규정)과 회사 문서(전결 규정)가 경합할 때 필터로 범위를 고정"),
    ]

    for q, where, point in experiments:
        print(f"\n=== 질문: {q}   (관찰: {point})")
        for label, kwargs in [("필터 없음", {}), (f"필터 {where}", {"where": where})]:
            r = col.query(query_texts=[q], n_results=3, **kwargs)
            print(f"  [{label}]")
            for cid, dist in zip(r["ids"][0], r["distances"][0]):
                print(f"    ({dist:.3f}) {cid}")

    print("""
기록 포인트:
- 필터는 정확도 장치이기 이전에 권한·최신성 장치다 (3회차 슬라이드).
- 필터로 검색 공간을 좁히면 상위권의 노이즈가 줄지만, 필터를 잘못 걸면
  정답 문서 자체가 차단된다 — "결재" 질문에서 doc_type을 반대로 걸어 확인해보라.""")


def task3(args):
    """거리 임계값 스윕 — '관련 없음' 판정의 정확도를 숫자로 관찰."""
    col = _ensure_indexed()

    eval_set = json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
    answerable = [c["question"] for c in eval_set if c["answer_chunk_ids"]]
    unanswerable = [c["question"] for c in eval_set if not c["answer_chunk_ids"]]
    unanswerable += ["오늘 점심 메뉴 추천해줘", "주식 시장 전망은?",
                     "파이썬 리스트 정렬하는 법"]

    def top1_distance(q: str) -> float:
        return col.query(query_texts=[q], n_results=1)["distances"][0][0]

    dist_ans = [top1_distance(q) for q in answerable]
    dist_un = [top1_distance(q) for q in unanswerable]

    rows = []
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        accept = sum(d <= t for d in dist_ans) / len(dist_ans)
        reject = sum(d > t for d in dist_un) / len(dist_un)
        rows.append([f"{t:.1f}", f"{accept:.0%}", f"{reject:.0%}"])
    print(f"답 있는 질문 {len(dist_ans)}개 / 답 없는 질문 {len(dist_un)}개\n")
    print_table(["임계값", "답 있는 질문 통과율↑", "답 없는 질문 거부율↑"], rows)

    print(f"\n답 있는 질문 top-1 거리:  min {min(dist_ans):.3f} / max {max(dist_ans):.3f}")
    print(f"답 없는 질문 top-1 거리:  min {min(dist_un):.3f} / max {max(dist_un):.3f}")
    print("""
읽는 법:
- 임계값을 낮추면 환각 방어(거부율)는 좋아지지만 정상 질문까지 거부한다 (트레이드오프).
- 두 분포가 겹치는 구간이 좁을수록 좋은 임베딩 — 겹침이 크면 임계값만으로는
  판정이 불가능하므로 7회차 리랭커 점수를 임계값으로 쓰는 방법을 검토하라.
- SimpleRAG(5회차)의 max_distance=0.6은 이 표에서 고른 값이다 — 본인 데이터에서는
  본인 분포로 다시 골라야 한다.""")


if __name__ == "__main__":
    run_tasks("3회차: 벡터 DB", {
        1: ("30개 이상 컬렉션 구축 (메타데이터 2종 이상)", task1),
        2: ("필터링 있음/없음 비교 실험 3가지", task2),
        3: ("거리 임계값별 '관련 없음' 판정 정확도 관찰", task3),
    })
