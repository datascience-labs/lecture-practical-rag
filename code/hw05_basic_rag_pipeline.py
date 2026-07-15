# hw05_basic_rag_pipeline.py — 5회차 과제(중간 프로젝트) 해답
#
# 과제 1: 문서 20개 이상(또는 30페이지 이상)으로 완전한 RAG 시스템 인덱싱
# 과제 2: 테스트 질문 15개 이상 — 잘 답하는 것 / 검색 실패 / 생성 실패 분류
# 과제 3: 실패 사례를 디버깅 트리로 원인 진단 → 다음 회차 개선 항목 배정
#
# 과제 1·2와 3의 진단은 API 키 없이 동작하고, 과제 3 끝의 생성 검사만 키가 필요하다.
#
# 실행: python hw05_basic_rag_pipeline.py [--task 1 2 3] [--list]

import json

from rank_bm25 import BM25Okapi

from bm25_search import tokenize
from corpus import DATA_DIR, ensure_indexed, load_corpus
from hw_common import get_client, print_table, run_tasks, skip
from rag_pipeline import SimpleRAG


def _eval_set() -> list[dict]:
    return json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))


def _classify(rag: SimpleRAG) -> tuple[list, list]:
    """평가셋 전 문항을 검색 관점에서 분류. (분류 행, 검색 실패 케이스)를 반환."""
    rows, failures = [], []
    for case in _eval_set():
        chunks = rag.retrieve(case["question"], k=4)          # max_distance=0.6 기본
        ids = [c["id"] for c in chunks]
        if not case["answer_chunk_ids"]:                      # 답이 없어야 하는 질문
            verdict = "거부 성공 (모른다고 답할 준비됨)" if not ids else "환각 위험 (무관 청크 통과)"
            rows.append([case["id"], "unanswerable", verdict])
            continue
        ranks = [ids.index(a) + 1 for a in case["answer_chunk_ids"] if a in ids]
        if ranks:
            rows.append([case["id"], case["type"], f"검색 성공 (정답 {min(ranks)}위)"])
        else:
            rows.append([case["id"], case["type"], "검색 실패"])
            failures.append(case)
    return rows, failures


def task1(args):
    """완전한 RAG 시스템 구축 — 인덱싱 규모 확인."""
    rag = SimpleRAG()
    ensure_indexed(rag)
    corpus = load_corpus()
    handbook = sum(1 for c in corpus if c["source"] == "sample_handbook.md")
    print(f"인덱싱 완료: 총 {rag.collection.count()}청크 "
          f"(핸드북 소절 {handbook} + 회사 문서 {len(corpus) - handbook}) — 요구: 문서 20개 이상")
    print("""
시스템 구성 (5회차 파이프라인 그대로):
  corpus.py 로딩 → SimpleRAG.index() → retrieve(k=4, max_distance=0.6) → ask()
대화형으로 써보려면: python main.py  (질문 앞 /r 로 검색만 진단)""")


def task2(args):
    """테스트 질문 21개(eval_set.json)를 성공/실패로 분류."""
    rag = SimpleRAG()
    ensure_indexed(rag)
    rows, failures = _classify(rag)
    print_table(["문항", "유형", "분류"], rows)

    total = len(rows)
    fail = len(failures)
    print(f"\n요약: {total}문항 중 검색 실패 {fail}건")
    print("""
분류 기준 (과제 요구 3분류와의 대응):
- 잘 답하는 것   = 검색 성공 + 생성 정상 (생성 확인은 과제 3에서 표본 검사)
- 검색 실패      = 정답 청크가 top-4에 없음 → 원인 진단은 과제 3
- 생성 실패      = 청크는 좋은데 답이 나쁨 — 검색 성공 문항에 API로 답을 만들어
                  키워드·출처를 검사해야 보인다 (과제 3 후반)""")


def task3(args):
    """검색 실패를 디버깅 트리로 진단하고, 표본으로 생성 품질을 검사."""
    rag = SimpleRAG()
    ensure_indexed(rag)
    _, failures = _classify(rag)

    corpus = load_corpus()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in corpus])
    ids = [c["id"] for c in corpus]

    print(f"--- 검색 실패 {len(failures)}건 진단 (5회차 디버깅 트리) ---")
    if not failures:
        print("  검색 실패 없음 — 평가셋을 더 어렵게 만들 차례다 (multi-hop 추가).")
    for case in failures:
        answers = set(case["answer_chunk_ids"])
        # 벡터 top-20 순위 확인: 후보에는 있는데 순위가 낮은가?
        r = rag.collection.query(query_texts=[case["question"]], n_results=20)
        vec_rank = next((i + 1 for i, cid in enumerate(r["ids"][0]) if cid in answers), None)
        # BM25 top-3: 키워드 검색은 잡는가?
        scores = bm25.get_scores(tokenize(case["question"]))
        bm_top3 = [ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:3]]
        bm25_hit = bool(answers & set(bm_top3))

        print(f"\n  [{case['id']}] {case['question']}  (정답: {sorted(answers)})")
        if vec_rank and vec_rank > 4:
            print(f"    진단: 정답이 벡터 {vec_rank}위 — 후보엔 있으나 순위가 낮다 → 7회차 리랭킹")
        elif bm25_hit:
            print("    진단: BM25는 top-3에서 잡는다 — 의미 매칭 실패 → 6회차 하이브리드")
        elif case["type"] == "multi-hop":
            print("    진단: multi-hop — 한 청크에 답이 없다 → 8회차 쿼리 분해 / 10회차 그래프")
        else:
            print("    진단: 벡터·BM25 모두 실패 → 4회차 청킹 재검토 또는 8회차 쿼리 변환")

    # --- 생성 품질 표본 검사 (검색은 성공하는 문항 + 환각 함정 문항) ---
    client = get_client()
    if client is None:
        skip("생성 품질 표본 검사는 ANTHROPIC_API_KEY가 필요하다.")
        return
    print("\n--- 생성 품질 표본 검사 ---")
    eval_set = {c["id"]: c for c in _eval_set()}
    for cid in ("hb-001", "hb-015"):                     # 정상 문항 + 답 없는 함정 문항
        case = eval_set[cid]
        print(f"\n  [{cid}] {case['question']}")
        answer = rag.ask(case["question"])
        print()
        if case["answer_chunk_ids"]:
            ok = any(kw in answer for kw in case["answer_keywords"])
            print(f"    → 정답 키워드 {case['answer_keywords']} 포함: {'✓' if ok else '✗ (생성 실패 의심)'}")
            print(f"    → 출처 표기 포함: {'✓' if '출처' in answer else '✗ (프롬프트 규칙 위반)'}")
        else:
            ok = "찾을 수 없" in answer
            print(f"    → '문서에 없다'고 거부: {'✓' if ok else '✗ 환각! (생성 실패)'}")


if __name__ == "__main__":
    run_tasks("5회차: 기본 RAG 파이프라인 (중간 프로젝트)", {
        1: ("문서 20개 이상 인덱싱 — 완전한 RAG 시스템", task1),
        2: ("테스트 질문 분류: 성공 / 검색 실패 / 환각 위험", task2),
        3: ("디버깅 트리 진단 + 생성 품질 표본 검사", task3),
    })
