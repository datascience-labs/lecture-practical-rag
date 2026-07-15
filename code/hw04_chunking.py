# hw04_chunking.py — 4회차 과제 해답
#
# 과제 1: 10페이지 이상 문서로 4가지 청킹 전략 비교 실험
# 과제 2: 질문 10개로 전략별 top-1 정확도 표 작성
# 과제 3: (심화) 헤더 경로 주입 전/후 검색 품질 비교
#
# 문서는 data/sample_handbook.md, 질문·정답 키워드는 data/eval_set.json을 쓴다.
# API 키 불필요. --semantic을 주면 전략 4(의미 기반)도 비교에 포함한다.
#
# 실행: python hw04_chunking.py [--task 1 2 3] [--semantic] [--list]

import json
import re

import chromadb
from chromadb.utils import embedding_functions

from chunking import fixed_chunk, markdown_chunk, recursive_chunk, semantic_chunk
from corpus import CHROMA_PATH, DATA_DIR, load_handbook_chunks
from hw_common import print_table, run_tasks

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
db = chromadb.PersistentClient(path=CHROMA_PATH)


def _questions() -> list[dict]:
    """핸드북에서 답이 나오는 단일 정답 질문 10개 (키워드로 정답 판정)."""
    eval_set = json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
    picked = [c for c in eval_set
              if c["type"] == "single-hop" and c["answer_keywords"]
              and all(i.startswith("handbook-") for i in c["answer_chunk_ids"])]
    return picked[:10]


def _strategies(with_semantic: bool) -> dict[str, list[str]]:
    document = (DATA_DIR / "sample_handbook.md").read_text(encoding="utf-8")
    strategies = {
        "fixed_300": fixed_chunk(document, size=300, overlap=30),
        "fixed_800": fixed_chunk(document, size=800, overlap=80),
        "recursive_500": recursive_chunk(document, size=500),
        "markdown": [c["text"] for c in markdown_chunk(document)],
    }
    if with_semantic:
        sentences = [s.strip() for s in re.split(r"(?<=다\.)\s+", document) if s.strip()]
        strategies["semantic"] = semantic_chunk(sentences, threshold=0.5)
    return strategies


def _indexed_collection(name: str, chunks: list[str]):
    """전략별 컬렉션 — 이미 있으면 재사용 (임베딩 비용 절약)."""
    col = db.get_or_create_collection(f"hw04_{name}", embedding_function=ef,
                                      metadata={"hnsw:space": "cosine"})
    if col.count() != len(chunks):
        db.delete_collection(f"hw04_{name}")
        col = db.create_collection(f"hw04_{name}", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})
        col.add(ids=[f"{name}-{i}" for i in range(len(chunks))], documents=chunks)
    return col


def task1(args):
    """전략별 청크 통계 + 인덱싱 — 비교 실험의 준비이자 첫 관찰."""
    strategies = _strategies(getattr(args, "semantic", False))
    rows = []
    for name, chunks in strategies.items():
        sizes = [len(c) for c in chunks]
        rows.append([name, len(chunks), sum(sizes) // len(sizes), max(sizes)])
        _indexed_collection(name, chunks)
    print_table(["전략", "청크 수", "평균 크기(자)", "최대(자)"], rows)
    print("""
관찰 포인트:
- fixed_300: 청크가 많고 작다 → 문맥 절단 위험 (두 번째 청크 끝을 눈으로 확인하라)
- fixed_800: 청크가 크다 → 한 청크에 여러 주제가 섞임 (검색 노이즈)
- markdown: 문서 구조(### 소절)를 따르므로 청크 수 = 소절 수, 주제 단위가 깨끗
- --semantic 옵션으로 전략 4까지 포함 가능 (임베딩 비용 ↑)""")


def task2(args):
    """질문 10개 × 전략별 top-1 정확도 표 — 정답 키워드 포함 여부로 판정."""
    strategies = _strategies(getattr(args, "semantic", False))
    questions = _questions()

    accuracy = {}
    detail_rows = []
    for name, chunks in strategies.items():
        col = _indexed_collection(name, chunks)
        hits = 0
        marks = []
        for case in questions:
            top1 = col.query(query_texts=[case["question"]], n_results=1)["documents"][0][0]
            ok = any(kw in top1 for kw in case["answer_keywords"])
            hits += ok
            marks.append("✓" if ok else "✗")
        accuracy[name] = hits
        detail_rows.append([name] + marks + [f"{hits}/{len(questions)}"])

    headers = ["전략"] + [c["id"] for c in questions] + ["top-1 정확도"]
    print_table(headers, detail_rows)
    print("\n질문 목록:")
    for c in questions:
        print(f"  {c['id']}: {c['question']}  (정답 키워드: {c['answer_keywords']})")
    best = max(accuracy, key=accuracy.get)
    print(f"""
결과 해석:
- 최고 전략: {best} ({accuracy[best]}/{len(questions)})
- 구조가 있는 문서(핸드북)는 대체로 markdown(구조 기반)이 유리하다 — 소절 하나가
  정확히 하나의 규정을 담아서, top-1 청크만으로 답이 완결되기 때문.
- fixed 계열이 틀리는 문항은 대부분 정답 문장이 청크 경계에서 잘리거나
  다른 주제와 섞인 경우다 — 틀린 문항의 top-1 청크를 직접 열어 원인을 기록하라.""")


def task3(args):
    """(심화) 헤더 경로 주입 전/후 — 같은 소절 청킹에서 경로 한 줄의 효과."""
    injected = load_handbook_chunks()                     # [문서 > 장 > 절] 주입판
    stripped = [{**c, "text": c["text"].split("]\n", 1)[-1]} for c in injected]

    questions = _questions()
    rows = []
    for label, name, chunks in [("경로 주입 후", "path_on", injected),
                                ("경로 주입 전", "path_off", stripped)]:
        col = _indexed_collection(name, [c["text"] for c in chunks])
        hits, dists = 0, []
        for case in questions:
            r = col.query(query_texts=[case["question"]], n_results=1)
            hits += any(kw in r["documents"][0][0] for kw in case["answer_keywords"])
            dists.append(r["distances"][0][0])
        rows.append([label, f"{hits}/{len(questions)}",
                     f"{sum(dists) / len(dists):.3f}"])
    print_table(["설정", "top-1 정확도", "평균 top-1 거리"], rows)
    print("""
해석:
- "[임직원 핸드북 > 제3장 휴가 > 3.1 연차 휴가]" 한 줄이 청크에 장·절 맥락을 부여한다.
- 효과가 큰 질문 유형: 청크 본문에 주제어가 안 나오는 경우 (본문이 "매년 15일
  부여된다"로 시작하면 '연차'라는 단어는 경로에만 있다).
- corpus.py가 이 방식을 기본 채택한 이유 — 5회차부터의 모든 실습이 이 위에서 돈다.""")


if __name__ == "__main__":
    run_tasks("4회차: 청킹 전략", {
        1: ("4가지 전략 비교 실험 (청크 통계·인덱싱)", task1),
        2: ("질문 10개 전략별 top-1 정확도 표", task2),
        3: ("(심화) 헤더 경로 주입 전/후 검색 품질 비교", task3),
    }, extra_args=[("--semantic", {"action": "store_true",
                                   "help": "전략 4(의미 기반 청킹)도 포함"})])
