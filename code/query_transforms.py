# query_transforms.py — 8회차: 쿼리 변환 패턴 4종
#
# 사용자의 질문은 검색에 최적화되어 있지 않다 → 검색 전에 LLM으로 변환한다.
#   rewrite_query    패턴 1: 대화 맥락 기반 재작성 (멀티턴 챗봇 필수)
#   multi_query      패턴 2: 여러 관점의 질문으로 확장 → RRF 융합
#   hyde_search      패턴 3: 가상의 정답 문서를 임베딩해 검색
#   decompose_query  패턴 4: 복합 질문을 하위 질문으로 분해
#
# 실행: python query_transforms.py   (ANTHROPIC_API_KEY 필요)

import anthropic


def rewrite_query(client, history: list[dict], question: str) -> str:
    """패턴 1: 대명사·생략을 해소한 독립형(standalone) 질문 생성."""
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=200,
        system="대화 맥락을 반영해 마지막 질문을 검색용 독립 질문으로 재작성하라. "
               "재작성된 질문만 출력하라.",
        messages=[{"role": "user",
                   "content": f"<대화>\n{hist_text}\n</대화>\n\n"
                              f"마지막 질문: {question}"}],
    )
    return response.content[0].text.strip()


def multi_query(client, question: str, n: int = 3) -> list[str]:
    """패턴 2: 하나의 질문을 여러 표현으로 확장 (재현율이 낮을 때)."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        system=f"주어진 질문을 서로 다른 표현/관점의 검색 질문 {n}개로 변형하라. "
               "한 줄에 하나씩만 출력하라.",
        messages=[{"role": "user", "content": question}],
    )
    variants = [q.strip() for q in response.content[0].text.split("\n") if q.strip()]
    return [question] + variants   # 원본도 포함


def hyde_search(client, collection, question: str, k: int = 4):
    """패턴 3: HyDE — 질문 대신 '그럴듯한 가짜 답변'과 비슷한 실제 문서를 찾는다.

    질문(짧은 의문문)과 문서(긴 설명문)의 형태 차이가 클 때 효과적.
    """
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        system="질문에 대한 답변이 담긴 사내 규정 문단을 작성하라. "
               "사실 여부는 중요하지 않다. 형식과 문체가 실제 규정과 같아야 한다.",
        messages=[{"role": "user", "content": question}],
    )
    hypothetical_doc = response.content[0].text
    # 질문 대신 가상 문서로 벡터 검색!
    return collection.query(query_texts=[hypothetical_doc], n_results=k)


def decompose_query(client, question: str) -> list[str]:
    """패턴 4: 복합/비교/multi-hop 질문을 독립적으로 검색 가능한 하위 질문으로 분해."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        system="복합 질문에 답하기 위해 필요한, 각각 독립적으로 검색 가능한 "
               "하위 질문 2~4개로 분해하라. 한 줄에 하나씩만 출력하라.",
        messages=[{"role": "user", "content": question}],
    )
    return [q.strip() for q in response.content[0].text.split("\n") if q.strip()]


if __name__ == "__main__":
    from corpus import ensure_indexed
    from rag_pipeline import SimpleRAG

    client = anthropic.Anthropic()

    # 패턴 1: 재작성 — 대명사가 해소되는지 확인
    history = [
        {"role": "user", "content": "연차 규정 알려줘"},
        {"role": "assistant", "content": "연차 휴가는 입사일 기준으로 매년 15일 부여됩니다."},
    ]
    q = "그럼 이월은 돼?"
    print(f"[재작성] '{q}' → '{rewrite_query(client, history, q)}'")

    # 패턴 2: Multi-Query
    print("\n[Multi-Query] '재택근무 조건이 뭐야?' →")
    for v in multi_query(client, "재택근무 조건이 뭐야?"):
        print(f"  - {v}")

    # 패턴 4: 쿼리 분해
    print("\n[쿼리 분해] 'PD-2000이랑 PD-3000은 각각 어느 공장에서 만들어?' →")
    for sub in decompose_query(client, "PD-2000이랑 PD-3000은 각각 어느 공장에서 만들어?"):
        print(f"  - {sub}")

    # 패턴 3: HyDE — 일반 검색과 top-1 비교
    rag = SimpleRAG()
    ensure_indexed(rag)
    q = "회사 그만두려면 어떻게 해야 돼?"
    plain = rag.collection.query(query_texts=[q], n_results=1)
    hyde = hyde_search(client, rag.collection, q, k=1)
    print(f"\n[HyDE] 질문: {q}")
    print(f"  일반 검색 top-1: ({plain['ids'][0][0]}) dist={plain['distances'][0][0]:.3f}")
    print(f"  HyDE 검색 top-1: ({hyde['ids'][0][0]}) dist={hyde['distances'][0][0]:.3f}")
