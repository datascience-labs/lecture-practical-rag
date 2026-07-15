# 실습 데이터셋

Practical RAG Course의 실습에 사용하는 예제 데이터셋. 모두 **가상 기업 "퍼플델타(주)"** 를 배경으로 하는 창작 문서로, 실제 기업·인물과 무관하다. 슬라이드 코드 예제에 등장하는 사실(연차 15일, 재택 주 2회, PD-2000 → 제조1공장 등)과 정확히 일치하도록 작성되어 있어, 예제 코드를 그대로 실행하면 슬라이드의 예시 출력과 같은 결과를 얻을 수 있다.

## 파일 구성

| 파일 | 내용 | 사용 회차 |
|---|---|---|
| `sample_handbook.md` | 임직원 업무 핸드북 (8개 장, 헤더 구조 + 표 포함) | 4 (청킹), 5 (기본 파이프라인), 6~8 (검색 개선), 9 (평가) |
| `eval_set.json` | 평가셋 21문항 (핸드북 15 + 회사 문서 6) | 7 (리랭킹 전후 비교), 9 (평가 하네스) |
| `company_docs/*.txt` | 회사 소개·조직·제품·고객 문서 12건 | 6~9 (하이브리드·리랭킹·평가 코퍼스), 10 (지식그래프), 11~12 (GraphRAG), 13 (OntoRAG), 14 (NodeRAG) |
| `eval_set_graph.json` | multi-hop·전역 질문 평가셋 8문항 | 10~12 (기본 RAG vs GraphRAG 비교) |

## sample_handbook.md

Markdown 헤더(`##` 장 / `###` 절) 구조가 있는 긴 문서라 4회차 청킹 전략 비교(fixed vs recursive vs markdown)에 적합하다. 표(경조 휴가, 출장비)가 포함되어 있어 "표가 청크 경계에서 잘리면 어떻게 되나"도 관찰할 수 있다.

```python
document = open("data/sample_handbook.md").read()
```

## eval_set.json

각 문항의 필드:

- `question` — 질문
- `type` — `single-hop` / `multi-hop` / `unanswerable` (9회차 과제의 실패 유형 분류에 사용)
- `ground_truth` — 사람이 검수한 정답 (RAGAS 등 LLM 기반 평가에 사용)
- `answer_section` — 정답이 있는 핸드북 절 제목
- `answer_keywords` — 정답 청크에 반드시 포함되는 키워드
- `answer_chunk_ids` — 참조 구현(`code/corpus.py`)의 청킹 기준 정답 청크 ID (`handbook-3.1`, `04-product-pd2000` 등 내용 기반이라 재인덱싱해도 안정적). **본인만의 청킹 전략을 쓴다면** 청크 ID가 달라지므로, 인덱싱 후 아래 스니펫으로 다시 채운다 (`any(kw in ...)` 매칭이라 오탐이 생길 수 있으니 채운 뒤 검수할 것):

```python
import json

eval_set = json.load(open("data/eval_set.json"))
for case in eval_set:
    case["answer_chunk_ids"] = [
        c["id"] for c in chunks          # chunks: 인덱싱에 사용한 [{"id", "text"}, ...]
        if any(kw in c["text"] for kw in case["answer_keywords"])
    ]
json.dump(eval_set, open("data/eval_set.json", "w"), ensure_ascii=False, indent=2)
```

`hb-015`(주차비)는 일부러 정답이 없는 질문이다 — 파이프라인이 "문서에 없다"고 답하는지(환각 여부) 확인하는 용도.

## company_docs/

10~12회차 GraphRAG 실습용 코퍼스. 개체(사람·조직·제품·공장·고객)와 관계가 촘촘히 얽혀 있어 트리플 추출과 multi-hop 탐색이 잘 동작한다.

- **multi-hop 체인 예시**: 김철수 → PD-2000 → 제조1공장 / 김철수 → 영업1팀 → 영업본부 → 박영희 → 5천만 원 전결
- **전역 질문**: 사업 구조(전원장치 60% / ESS 25% / 서비스 15%)가 여러 문서에 흩어져 있어 커뮤니티 요약의 효과가 드러난다
- **개체 해소 함정**: `08-customers.txt`에는 "김철수 과장"이 **"김 과장"** 으로 표기되어 있다. 순진한 트리플 추출은 이를 다른 개체로 만든다 — 12회차 개체 해소(entity resolution) 심화 과제의 소재.
- **사건 노드**: "2026년 3월 PD-2000 단종 결정"과 그 배경(부품 수급)이 두 문서에 나뉘어 있어 14회차 NodeRAG의 "PD-2000 단종의 배경은?" 질문에 대응한다.

```bash
# Microsoft GraphRAG 라이브러리 사용 시 (12회차)
mkdir -p ./ragproj/input && cp data/company_docs/*.txt ./ragproj/input/
```

## eval_set_graph.json

기본 RAG(5회차)가 실패하고 GraphRAG(10~12회차)가 성공해야 하는 질문 모음. `type` 필드(`single-hop` / `multi-hop` / `comparison` / `global`)로 유형별 성공률을 비교하라. `source_docs`는 정답 근거가 흩어져 있는 문서 목록이다.

## 15~16회차 (멀티모달)

CAD 도면·현장 사진은 보안 특성상 저장소에 포함하지 않는다. 본인 소유의 이미지 10~20장을 준비하거나, 슬라이드에 안내된 공개 데이터셋을 사용하라.
