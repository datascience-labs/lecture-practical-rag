# generate_evalset.py — 9회차: 청크에서 평가셋 초안 자동 생성
#
# 평가셋 구축은 RAG 평가에서 가장 어려운 부분 — LLM으로 초안을 만들 수 있다.
# ⚠️ 자동 생성 후 반드시 사람이 검수하라 (비현실적 질문, 잘못된 정답 제거).
# ⚠️ 실제 사용자 질문 로그가 쌓이면 그것이 최고의 평가셋이다.
#
# 실행: python generate_evalset.py [청크수]     (기본 10개, ANTHROPIC_API_KEY 필요)
#   결과: data/eval_set.generated.json  ← 검수 전 초안.
#   검수를 마친 평가셋은 data/eval_set.json으로 저장해 git으로 버전 관리한다.

import json
import random
import re
import sys

import anthropic

from corpus import DATA_DIR, load_corpus


def generate_qa_from_chunk(client, chunk_id: str, chunk_text: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        system="주어진 문서 조각으로만 답할 수 있는 현실적인 사용자 질문 1개와 "
               "그 정답을 만들어라. JSON으로만 출력: "
               '{"question": ..., "ground_truth": ...}',
        messages=[{"role": "user", "content": chunk_text}],
    )
    text = response.content[0].text
    # 코드펜스(```json ... ```)가 섞여 나와도 JSON 부분만 추출
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    qa = json.loads(match.group() if match else text)
    return {**qa, "type": "generated", "answer_chunk_ids": [chunk_id]}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    client = anthropic.Anthropic()

    corpus = load_corpus()
    random.seed(42)                          # 재실행해도 같은 청크가 뽑히도록
    sampled = random.sample(corpus, min(n, len(corpus)))

    eval_set = []
    for c in sampled:
        try:
            qa = generate_qa_from_chunk(client, c["id"], c["text"])
        except Exception as e:               # JSON 파싱 실패 등은 건너뛰고 계속
            print(f"  건너뜀 ({c['id']}): {e}")
            continue
        eval_set.append(qa)
        print(f"  [{c['id']}] {qa['question']}")

    out = DATA_DIR / "eval_set.generated.json"
    out.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n{len(eval_set)}문항 저장: {out}")
    print("⚠️ 반드시 검수하라: 문서를 보지 않은 사용자가 실제로 할 법한 질문인가? "
          "정답이 청크 내용과 일치하는가?")
