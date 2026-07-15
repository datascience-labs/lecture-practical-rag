# ragas_eval.py — 9회차 선택 실습: RAGAS 프레임워크로 평가
#
# 직접 구현한 지표(retrieval_metrics, llm_judge)의 정제된 버전을 제공하는
# 프레임워크. ⚠️ 선택 실습인 이유:
#   - RAGAS의 기본 judge는 OpenAI API를 사용한다 → OPENAI_API_KEY 필요
#   - 답변 생성에는 우리 파이프라인(Claude)을 사용한다 → ANTHROPIC_API_KEY 필요
#   - ragas는 버전에 따라 API가 자주 바뀐다 (이 코드는 ragas 0.1~0.2 기준)
#
# 실행: pip install ragas datasets && python ragas_eval.py [문항수]

import json
import sys

from corpus import DATA_DIR, ensure_indexed
from rag_pipeline import SYSTEM_PROMPT, SimpleRAG


def answer_without_streaming(rag: SimpleRAG, question: str) -> tuple[str, list[str]]:
    """평가용: 스트리밍 출력 없이 (답변, 검색된 청크 텍스트들)을 반환."""
    chunks = rag.retrieve(question)
    if not chunks:
        return "제공된 문서에서 답을 찾을 수 없습니다.", []
    response = rag.llm.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": rag._build_prompt(question, chunks)}],
    )
    return response.content[0].text, [c["text"] for c in chunks]


if __name__ == "__main__":
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError:
        sys.exit("ragas가 설치되지 않았다: pip install ragas datasets")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # judge 호출이 많아 기본 5문항
    eval_set = [c for c in
                json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
                if c["answer_chunk_ids"]][:n]          # unanswerable 문항 제외

    rag = SimpleRAG()
    ensure_indexed(rag)

    questions, answers, contexts, ground_truths = [], [], [], []
    for case in eval_set:
        ans, ctx = answer_without_streaming(rag, case["question"])
        questions.append(case["question"])
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(case["ground_truth"])
        print(f"  답변 수집: {case['question']}")

    data = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,     # list of list
        "ground_truth": ground_truths,
    })
    result = evaluate(data, metrics=[faithfulness, answer_relevancy, context_recall])
    print(f"\n{result}")   # {'faithfulness': 0.89, 'answer_relevancy': 0.93, ...}
