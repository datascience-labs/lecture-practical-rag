# hw02_embeddings.py — 2회차 과제 해답
#
# 과제 1: 도메인 문서 10개로 실습 3 재구성 → 잘 되는 질문 / 안 되는 질문 각 3개
# 과제 2: jhgan/ko-sroberta-multitask 모델로 같은 실험 → bge-m3와 비교
# 과제 3: (심화) MTEB 리더보드 한국어 retrieval 상위 모델 조사 — 조사 예시
#
# "본인 도메인 문서"는 실습 데이터(퍼플델타)로 대신한다 — 본인 문서로 바꾸려면
# DOC_IDS만 교체하면 된다. API 키 불필요 (임베딩 모델 다운로드만 필요).
#
# 실행: python hw02_embeddings.py [--task 1 2 3] [--list]

from corpus import load_corpus
from hw_common import print_table, run_tasks

# 도메인 문서 10개: 핸드북 소절 7 + 회사 문서 3 (id는 corpus.py 기준)
DOC_IDS = ["handbook-2.3", "handbook-3.1", "handbook-3.4", "handbook-4.1",
           "handbook-5.1", "handbook-5.2", "handbook-6.2",
           "04-product-pd2000", "05-product-pd3000", "08-customers"]

# (질문, 정답 문서 id, 유형) — 정답 id가 None이면 "10개 문서로는 답할 수 없는 질문"
QUESTIONS = [
    # 잘 될 것으로 기대: 단어가 달라도 의미가 통하는 패러프레이즈
    ("집에서 일해도 되나요?",             "handbook-2.3",      "의미형(재택 단어 없음)"),
    ("휴가는 1년에 며칠 받을 수 있나요?",   "handbook-3.1",      "의미형(연차→휴가)"),
    ("몸이 아파서 출근을 못 하면?",        "handbook-3.4",      "의미형(병가 단어 없음)"),
    # 안 될 것으로 기대: 임베딩 검색의 전형적 약점
    ("법카 처리 기한",                   "handbook-5.1",      "은어/축약(법카)"),
    ("PD-2000보다 가벼운 모델은?",        "05-product-pd3000", "비교·조건 질문"),
    ("김철수 상사의 결재 한도는?",         None,                "multi-hop(문서 밖)"),
]


def _load_docs() -> dict[str, str]:
    corpus = {c["id"]: c["text"] for c in load_corpus()}
    return {i: corpus[i] for i in DOC_IDS}


def _run_experiment(model_name: str) -> list[list]:
    """모델 하나로 QUESTIONS 전체를 검색해 (top-1, 판정) 행을 만든다."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    docs = _load_docs()
    ids = list(docs)
    doc_embs = model.encode(list(docs.values()))

    rows = []
    for q, answer_id, qtype in QUESTIONS:
        scores = util.cos_sim(model.encode(q), doc_embs)[0]
        top = int(scores.argmax())
        got = ids[top]
        ok = "✓" if got == answer_id else ("△" if answer_id is None else "✗")
        rows.append([q, qtype, got, f"{scores[top]:.3f}", ok])
    return rows


def task1(args):
    """bge-m3로 10문서 검색 실험 — 잘 되는/안 되는 질문의 패턴을 확인한다."""
    rows = _run_experiment("BAAI/bge-m3")
    print_table(["질문", "유형", "top-1 문서", "유사도", "판정"], rows)
    print("""
해설 (기록 예시):
- 잘 되는 질문: 질문에 문서의 단어가 없어도(재택/병가) 의미로 매칭된다.
  이것이 키워드 검색과의 결정적 차이 (2회차 핵심).
- 안 되는 질문의 유형:
  · 은어/축약("법카") — 학습 데이터에 드문 표현은 의미 공간에서 멀 수 있다
  · 비교·조건 질문 — "가벼운 모델"의 답(PD-3000)이 아니라 질문에 등장한
    PD-2000 문서가 끌려오기 쉽다
  · multi-hop — 애초에 한 문서에 답이 없다 (10개 문서 안에 전결 규정 없음)
- 대응은 이후 회차에서: 은어 → 6회차 하이브리드/8회차 재작성,
  multi-hop → 8회차 분해·10회차 그래프.
- △ 표시는 "정답 문서가 없는 질문" — 무엇이 1위로 나왔든 유사도가 낮은지 확인하라
  (3회차 거리 임계값 설계의 근거가 된다).""")


def task2(args):
    """같은 실험을 ko-sroberta로 반복 — 모델에 따라 성패가 갈리는 질문을 찾는다."""
    print("bge-m3 (1024차원, 다국어):")
    rows_a = _run_experiment("BAAI/bge-m3")
    print_table(["질문", "유형", "top-1", "유사도", "판정"], rows_a)

    print("\njhgan/ko-sroberta-multitask (768차원, 한국어 특화):")
    rows_b = _run_experiment("jhgan/ko-sroberta-multitask")
    print_table(["질문", "유형", "top-1", "유사도", "판정"], rows_b)

    diff = [(a[0], a[4], b[4]) for a, b in zip(rows_a, rows_b) if a[2] != b[2]]
    if diff:
        print("\ntop-1이 달라진 질문:")
        for q, a, b in diff:
            print(f"  - {q}  (bge-m3 {a} → ko-sroberta {b})")
    else:
        print("\n이 6문항에서는 두 모델의 top-1이 같았다 — 질문을 더 어렵게 만들어보라.")
    print("""
기록 포인트:
- 유사도 절대값은 모델끼리 비교 불가 (의미 공간이 다름) — 순위와 정답 여부만 비교하라.
- 모델 교체 = 전체 재인덱싱 (2회차 슬라이드의 경고). 성능·속도·차원수를 함께 기록해
  도메인에 맞는 모델을 근거 있게 선택하는 것이 이 과제의 목적이다.""")


def task3(args):
    """(심화) MTEB 한국어 retrieval 상위 모델 조사 — 조사 결과 예시."""
    print("""MTEB 리더보드(huggingface.co/spaces/mteb/leaderboard)에서
Korean + Retrieval 필터로 조사한 예시 (순위는 수시로 바뀌므로 직접 확인할 것):

| 모델 | 특징 | 확인할 것 |
|---|---|---|
| BAAI/bge-m3                        | 다국어, 8192 토큰, dense+sparse 겸용 | 이 강의의 기준선 |
| intfloat/multilingual-e5-large-instruct | 다국어, 지시문(instruct) 프리픽스 필요 | 프리픽스 규칙 준수 여부 |
| nlpai-lab/KURE-v1                  | 한국어 retrieval 특화 파인튜닝       | 라이선스, 입력 길이 |
| jhgan/ko-sroberta-multitask        | 가볍고 빠름(768차원)                | 긴 문서에서 성능 하락 |
| Alibaba-NLP/gte-multilingual-base  | 경량 다국어                         | 한국어 점수 |

조사할 때 점수 외에 반드시 함께 기록할 것:
1. max token — 본인 청크 크기(4회차)보다 큰가?
2. 차원 수 — 저장 공간·검색 속도 비용
3. 라이선스 — 상용 이용 가능 여부
4. 프리픽스/지시문 규칙 — e5 계열은 "query: " 프리픽스를 빼먹으면 성능 급락
결론은 리더보드 점수가 아니라 "본인 평가셋(9회차)에서의 점수"로 내려라.""")


if __name__ == "__main__":
    run_tasks("2회차: 임베딩", {
        1: ("10개 문서 의미 검색 — 잘 되는/안 되는 질문 찾기 (bge-m3)", task1),
        2: ("ko-sroberta-multitask로 동일 실험 → 모델 비교", task2),
        3: ("(심화) MTEB 한국어 retrieval 상위 모델 조사 예시", task3),
    })
