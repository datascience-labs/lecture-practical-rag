# adaptive_rag.py — 8회차 실습: 조건부(적응형) 쿼리 변환 파이프라인
#
# 모든 질문에 모든 변환을 걸면 지연시간과 비용만 늘어난다.
# → 질문 유형을 먼저 분류하고, 필요한 변환만 적용한다.
#   simple     → 변환 없이 그대로 검색
#   contextual → 대화 맥락 기반 재작성 (패턴 1)
#   complex    → 하위 질문 분해 (패턴 4)
#
# 실행: python adaptive_rag.py   (ANTHROPIC_API_KEY 필요)

import anthropic

from hybrid_search import rrf_fuse
from query_transforms import decompose_query, rewrite_query
from rag_pipeline import SimpleRAG


def classify_and_transform(client, history, question) -> list[str]:
    """질문 유형에 따라 변환 전략 선택 → 검색에 쓸 질문 리스트 반환."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system="""질문을 분류해 한 단어로 답하라:
- simple: 단일 주제의 독립적 질문
- contextual: 이전 대화를 참조하는 질문 (대명사, 생략)
- complex: 여러 하위 질문이 필요한 복합 질문""",
        messages=[{"role": "user", "content": question}],
    )
    qtype = response.content[0].text.strip().lower()

    if qtype == "contextual":
        return [rewrite_query(client, history, question)]
    elif qtype == "complex":
        return decompose_query(client, question)
    return [question]                       # simple은 변환 없이 그대로


class AdaptiveRAG(SimpleRAG):
    """변환된 질문들로 각각 검색 → RRF 융합 → 답변 생성."""

    def ask(self, question: str, history: list[dict] | None = None) -> str:
        history = history or []
        queries = classify_and_transform(self.llm, history, question)
        if queries != [question]:
            print(f"  (검색 질문으로 변환: {queries})")

        pool: dict[str, dict] = {}
        rankings = []
        for q in queries:
            chunks = super().retrieve(q, k=4)
            rankings.append([c["id"] for c in chunks])
            pool.update({c["id"]: c for c in chunks})
        top = [pool[cid] for cid in rrf_fuse(rankings)[:4]]

        if not top:
            msg = "제공된 문서에서 답을 찾을 수 없습니다."
            print(msg)
            return msg

        from rag_pipeline import SYSTEM_PROMPT
        with self.llm.messages.stream(
            model="claude-opus-4-8",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": self._build_prompt(question, top)}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            return stream.get_final_message().content[0].text


if __name__ == "__main__":
    from corpus import ensure_indexed

    rag = AdaptiveRAG()
    ensure_indexed(rag)

    # 3턴 시나리오: simple → contextual(대명사) → complex(비교·multi-hop)
    scenario = [
        "연차는 며칠이야?",                                  # simple
        "그럼 그거 다음 해로 넘길 수 있어?",                    # contextual
        "PD-2000이랑 PD-3000은 각각 어느 공장에서 만들어?",     # complex
    ]

    history: list[dict] = []
    for q in scenario:
        print(f"\n{'=' * 60}\n질문 > {q}\n")
        answer = rag.ask(q, history)
        print()
        history += [{"role": "user", "content": q},
                    {"role": "assistant", "content": answer}]
