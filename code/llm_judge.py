# llm_judge.py — 9회차: LLM-as-Judge 생성 품질 평가
#
# 정답 라벨 없이도 평가 가능한 두 핵심 지표:
#   judge_faithfulness  충실성 — 답변이 컨텍스트에 근거하는가? (환각 탐지)
#   judge_relevancy     관련성 — 답변이 질문에 실제로 대답하는가? (동문서답 탐지)
#
# 운영 원칙: 같은 judge 모델로 일관되게 측정하라 — 모델을 바꾸면 점수 비교 불가.
#
# 실행: python llm_judge.py   (ANTHROPIC_API_KEY 필요)

import re

import anthropic


def _parse_score(text: str) -> float:
    """마지막 'SCORE: x/y' 표기를 찾아 비율로 변환."""
    match = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", text)
    if not match:
        raise ValueError(f"SCORE 표기를 찾지 못함: {text[-100:]}")
    x, y = int(match.group(1)), int(match.group(2))
    return x / y if y else 0.0


def judge_faithfulness(client, context: str, answer: str) -> float:
    """답변의 주장(claim)별로 컨텍스트 근거 여부를 판정 → 확인된 비율."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        system="""답변의 각 주장(claim)을 추출하고, 각 주장이 컨텍스트에서
확인되는지 판정하라. 마지막 줄에 '확인된 주장 수/전체 주장 수'를
'SCORE: x/y' 형식으로 출력하라.""",
        messages=[{"role": "user",
                   "content": f"<컨텍스트>\n{context}\n</컨텍스트>\n\n"
                              f"<답변>\n{answer}\n</답변>"}],
    )
    return _parse_score(response.content[0].text)


def judge_relevancy(client, question: str, answer: str) -> float:
    """답변이 질문에 실제로 대답하는지 5점 척도로 채점 → 0~1로 정규화."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        system="""답변이 질문에 실제로 대답하는지 평가하라.
5: 질문에 완전하고 직접적으로 대답 / 3: 부분적으로만 대답 /
1: 동문서답. 근거를 짧게 쓰고 마지막 줄에 'SCORE: x/5'를 출력하라.""",
        messages=[{"role": "user",
                   "content": f"<질문>\n{question}\n</질문>\n\n"
                              f"<답변>\n{answer}\n</답변>"}],
    )
    return _parse_score(response.content[0].text)


if __name__ == "__main__":
    client = anthropic.Anthropic()

    context = ("연차 휴가는 입사일 기준으로 매년 15일 부여된다. "
               "사용하지 못한 연차는 최대 5일까지 다음 해로 이월할 수 있다.")
    question = "연차는 며칠인가요?"

    cases = {
        "충실한 답변": "연차 휴가는 매년 15일 부여되며, 최대 5일까지 이월할 수 있습니다.",
        "환각 섞인 답변": "연차 휴가는 매년 15일 부여되며, 미사용 연차는 전액 수당으로 "
                       "자동 지급되고 입사 5년차부터는 30일로 늘어납니다.",
        "동문서답": "휴가를 잘 쓰는 것은 워라밸에 매우 중요합니다. 계획적으로 쉬세요.",
    }

    print(f"컨텍스트: {context}\n질문: {question}\n")
    for label, answer in cases.items():
        f = judge_faithfulness(client, context, answer)
        r = judge_relevancy(client, question, answer)
        print(f"[{label}] faithfulness={f:.2f}  relevancy={r:.2f}")
        print(f"  답변: {answer[:60]}...\n")

    print("기대: 충실한 답변만 두 지표 모두 높고, 환각은 faithfulness↓, 동문서답은 relevancy↓")
