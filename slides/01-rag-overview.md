---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 1회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 1회차: RAG 개요와 LLM의 한계

## 왜 RAG(Retrieval-Augmented Generation)인가?


이미지 파일: [s01-cover.png](image-prompts.md#s01-cover-png)

---

## 오늘의 목표

- LLM의 근본적 한계 3가지를 설명할 수 있다
- RAG의 기본 구조(검색 → 증강 → 생성)를 이해한다
- 파인튜닝 vs RAG를 상황에 맞게 선택할 수 있다
- 실습 환경을 구축하고 첫 Claude API 호출을 실행한다

---

## LLM의 한계 1: 지식 컷오프 (Knowledge Cutoff)

- LLM은 **학습 시점까지의 데이터**만 알고 있음
- 어제 발표된 사내 정책, 오늘 업데이트된 제품 매뉴얼은 모름

```
Q: 우리 회사의 2026년 휴가 정책이 어떻게 되나요?
A: 죄송하지만 귀사의 내부 정책 정보에는 접근할 수 없습니다...
```

→ **외부 지식을 주입할 방법**이 필요

---

## LLM의 한계 2: 환각 (Hallucination)

- 모르는 것을 "모른다"고 하지 않고 그럴듯하게 지어냄
- 근거 없는 답변은 업무에서 치명적 (법률, 의료, 금융...)

| 상황 | LLM 단독 | RAG 적용 |
|---|---|---|
| 근거 문서 | 없음 | 검색된 문서 제공 |
| 답변 검증 | 불가능 | 출처(citation) 확인 가능 |
| 오답 시 | 그럴듯한 거짓말 | "문서에 없음" 응답 유도 가능 |

---

## LLM의 한계 3: 프라이빗 데이터 접근 불가

- 사내 위키, 계약서, 고객 DB는 학습 데이터에 없음
- 전부 프롬프트에 넣자니 → **컨텍스트 윈도우 한계 + 비용**
  - 1M 토큰 컨텍스트라도 수백만 페이지 문서는 불가능
  - 매 요청마다 전체 문서 전송 = 비용 폭발

→ **필요한 부분만 골라서** 넣어야 함 = 검색(Retrieval)

---

## RAG란 무엇인가

> **R**etrieval-**A**ugmented **G**eneration
> 검색으로 증강된 생성

```
사용자 질문
    │
    ▼
┌─────────┐    ┌──────────────┐    ┌─────────┐
│ Retrieval│ →  │ Augmentation │ →  │Generation│
│ (검색)   │    │ (프롬프트 증강)│    │ (생성)   │
└─────────┘    └──────────────┘    └─────────┘
 관련 문서 찾기    질문+문서 결합       LLM이 답변 생성
```

---

## RAG 파이프라인 상세

**인덱싱 단계 (오프라인, 사전 작업)**

1. 문서 수집 (PDF, 웹, DB...)
2. 청킹 (Chunking) — 문서를 적절한 크기로 분할
3. 임베딩 (Embedding) — 텍스트를 벡터로 변환
4. 벡터 DB에 저장

**질의 단계 (온라인, 실시간)**

5. 질문을 임베딩
6. 유사도 검색으로 관련 청크 검색
7. 질문 + 청크를 프롬프트로 구성
8. LLM이 답변 생성

---

## 파인튜닝 vs RAG

| 기준 | 파인튜닝 | RAG |
|---|---|---|
| 지식 업데이트 | 재학습 필요 (느림, 비쌈) | 문서만 추가 (즉시) |
| 출처 제시 | 불가능 | 가능 |
| 초기 비용 | 높음 (GPU, 데이터셋) | 낮음 |
| 스타일/형식 학습 | 강함 | 약함 |
| 환각 억제 | 제한적 | 근거 기반으로 억제 |

**실무 원칙**: 지식 주입 → RAG, 말투·형식·도메인 특화 → 파인튜닝. 둘은 배타적이지 않고 병행 가능.

---

## 왜 지금 Graph 기반 RAG인가 (이 강의의 목적지)

기본 RAG(벡터 검색)의 한계:

- **"우리 회사 전체 매출 흐름을 요약해줘"** → 특정 청크가 답이 아님 (전역 질문)
- **"A 부서장의 상사의 결재 권한은?"** → 여러 문서를 연결(multi-hop)해야 함
- 청크 단위 검색은 **문서 간 관계**를 놓침

→ 10회차부터 배울 **GraphRAG, OntoRAG, NodeRAG**가 해결
→ 그 전에 기초를 탄탄히 (1~9회차)

---

## 17회차 로드맵

| 회차 | 주제 | 회차 | 주제 |
|---|---|---|---|
| 1 | RAG 개요 | 10 | 지식 그래프 기초 |
| 2 | 임베딩 | 11 | GraphRAG 개념 |
| 3 | 벡터 DB | 12 | GraphRAG 실습 |
| 4 | 청킹 전략 | 13 | OntoRAG |
| 5 | 기본 RAG 구축 | 14 | NodeRAG |
| 6 | 하이브리드 검색 | 15 | 오픈소스 VLM·멀티모달 RAG |
| 7 | 리랭킹 | 16 | VLM 실전: 도면 검색·현장 사진 |
| 8 | 쿼리 변환·고급 패턴 | 17 | 종합 프로젝트 |
| 9 | RAG 평가 (RAGAS) | | |

---

<!-- _class: lead -->

# 실습: 환경 구축과 첫 API 호출

---

## 실습 1: 환경 구축

```bash
# Python 3.11+ 권장
python -m venv .venv
source .venv/bin/activate

# 핵심 패키지 설치
pip install anthropic sentence-transformers chromadb

# API 키 설정 (https://platform.claude.com 에서 발급)
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 실습 2: 첫 Claude API 호출

```python
# hello_claude.py
import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "RAG가 무엇인지 한 문단으로 설명해줘."}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

---

## 실습 3: RAG의 필요성 체감하기

LLM에게 **학습 데이터에 없는 내용**을 물어보자:

```python
question = "우리 회사 '판교 오피스'의 회의실 예약 규칙을 알려줘."

# (1) 컨텍스트 없이 질문 → 환각 또는 "모른다" 응답
# (2) 컨텍스트를 넣고 질문 → 정확한 답변
context = """[사내 규정 3.2조] 판교 오피스 회의실은
그룹웨어에서 최대 2주 전부터 예약 가능하며,
6인 이상 회의실은 부서장 승인이 필요하다."""

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"다음 문서를 참고해 답변해줘.\n\n<문서>\n{context}\n</문서>\n\n질문: {question}"
    }],
)
```

→ 이것이 RAG의 "A(증강)"와 "G(생성)". 다음 시간부터 "R(검색)"을 자동화한다.

---

## 과제

1. 실습 3의 코드에서 컨텍스트 유무에 따른 답변 차이를 비교하고 기록하기
2. 본인 업무/관심 도메인에서 RAG로 해결하고 싶은 문제 1가지 정의해오기
   - 어떤 문서? 어떤 질문? 기대하는 답변 형태?
   - → 17회차 종합 프로젝트의 주제가 됩니다

## 다음 시간

**2회차: 임베딩과 벡터 표현** — 텍스트를 숫자로 바꾸면 왜 검색이 되는가?
