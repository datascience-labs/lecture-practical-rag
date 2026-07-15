# 실습 코드 (4~9회차) & 과제 해답 (1~16회차)

슬라이드에 나오는 실습 코드의 실행 가능한 버전. 파일명은 슬라이드의 주석
(`# rag_pipeline.py` 등)과 일치하며, 세션 간 코드가 누적된다
(5회차 `SimpleRAG` → 7회차 `RerankedRAG`가 상속하는 식).
회차별 과제의 참조 해답은 `hw01_*.py`~`hw16_*.py`에 있다 (아래 [과제 해답](#과제-해답-hw01hw16) 참고).

## 준비

```bash
cd practical-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # 답변 생성이 있는 실습만 필요
cd code
```

- 최초 실행 시 임베딩 모델 `BAAI/bge-m3`(≈2GB), 7회차부터 리랭커
  `BAAI/bge-reranker-v2-m3`(≈2GB)를 다운로드한다.
- 벡터 DB는 저장소 루트의 `chroma_db/`에 생성된다 (git에 커밋하지 않음).
- 검색·평가 실습(4, 6, 7, 9회차 대부분)은 **API 키 없이 동작**한다.

## 회차별 실행 순서

| 회차 | 파일 | 내용 | API 키 |
|---|---|---|---|
| 4 | `chunking.py` | 청킹 전략 4종 구현 (`--semantic`으로 전략 4까지) | — |
| 4 | `chunking_experiment.py` | 전략별 검색 결과 비교 실험 | — |
| 5 | `rag_pipeline.py` | `SimpleRAG` 클래스 (import용) | — |
| 5 | `main.py` | CLI 질의응답 시스템 (`/r 질문`으로 검색만 진단) | 필요 |
| 6 | `bm25_search.py` | 한국어 토큰화 + BM25 검색 | — |
| 6 | `hybrid_search.py` | `rrf_fuse` + `HybridRetriever`, 3-way 비교 | — |
| 7 | `rerank_basic.py` | Cross-encoder 리랭커 기본 사용 | — |
| 7 | `rag_pipeline_v2.py` | `RerankedRAG` + 리랭킹 전/후 Hit@3·MRR 비교 | — |
| 8 | `query_transforms.py` | 재작성 / Multi-Query / HyDE / 쿼리 분해 | 필요 |
| 8 | `adaptive_rag.py` | 질문 유형 분류 → 조건부 변환 파이프라인 | 필요 |
| 9 | `retrieval_metrics.py` | Hit@k, Recall@k, MRR 구현 + 자체 테스트 | — |
| 9 | `generate_evalset.py` | 평가셋 초안 자동 생성 (생성 후 반드시 검수!) | 필요 |
| 9 | `llm_judge.py` | LLM-as-Judge (충실성 / 관련성) | 필요 |
| 9 | `eval_harness.py` | 기본/하이브리드/+리랭크 3단계 비교 하네스 | — |
| 9 | `ragas_eval.py` | (선택) RAGAS 프레임워크 평가 | 필요 + OpenAI |

공용 모듈 `corpus.py`는 `data/sample_handbook.md`(### 소절 단위, 헤더 경로 주입)와
`data/company_docs/*.txt`(파일당 1청크)를 로딩한다. 청크 id(`handbook-3.1`,
`04-product-xk2000` 등)는 내용 기반이라 재인덱싱해도 안정적이며,
`data/eval_set.json`의 `answer_chunk_ids`가 이 id를 가리킨다.

## 과제 해답 (hw01~hw16)

각 회차 과제의 참조 해답. 실습 코드와 파일이 분리되어 있어 **hw 파일 실행 = 과제
부분만 실행**이며, 옵션으로 개별 과제 항목을 고를 수 있다:

```bash
python hw06_hybrid_search.py            # 6회차 과제 전부
python hw06_hybrid_search.py --task 3   # 과제 3만 실행
python hw06_hybrid_search.py --list     # 과제 목록 확인
```

- "본인 도메인 문서" 과제는 실습 데이터(퍼플델타)로 시연한다 — 본인 데이터로
  바꾸는 지점은 각 파일 상단 주석에 적어 두었다.
- LLM 산출물(트리플 추출, 커뮤니티 요약 등)은 저장소 루트 `hw_cache/`에 캐시된다
  → 최초 1회만 API 키가 필요하고, 이후에는 키 없이 재실행된다.
- 공용 모듈: `hw_common.py`(--task 실행기·LLM·캐시), `graph_common.py`(10~14회차
  그래프 RAG 구현 — 슬라이드 코드의 캐시 적용판).

| 회차 | 파일 | API 키 | 비고 |
|---|---|---|---|
| 1 | `hw01_rag_overview.py` | 과제 1만 | 과제 2는 작성 예시·템플릿 출력 |
| 2 | `hw02_embeddings.py` | — | ko-sroberta 추가 다운로드(~300MB) |
| 3 | `hw03_vector_databases.py` | — | 자체 컬렉션 `hw03_docs` 생성 |
| 4 | `hw04_chunking.py` | — | `--semantic`으로 전략 4 포함 |
| 5 | `hw05_basic_rag_pipeline.py` | 생성 검사만 | 검색 분류·진단은 키 불필요 |
| 6 | `hw06_hybrid_search.py` | — | |
| 7 | `hw07_reranking.py` | 과제 3만 | LLM 리랭킹 비교 |
| 8 | `hw08_query_transformation.py` | 필요 | 쿼리 변환 자체가 LLM 호출 |
| 9 | `hw09_evaluation.py` | 일부 | ④쿼리변환·자동 생성 시연만 키 필요 |
| 10 | `hw10_knowledge_graphs.py` | 최초 1회 | 추출 캐시 후 키 불필요 |
| 11 | `hw11_graphrag_concepts.py` | 최초 1회 | 42청크 추출 — LLM 호출 많음 |
| 12 | `hw12_graphrag_hands_on.py` | 최초 1회 | 평가표(과제 2)는 judge 호출 |
| 13 | `hw13_ontorag.py` | 최초 1회 | 비교표 완성엔 hw10 캐시 필요 |
| 14 | `hw14_noderag.py` | 최초 1회 | `--limit N`으로 과제 3 비용 제한 |
| 15 | `hw15_multimodal_vlm.py` | — (로컬 VLM) | 본인 이미지 + Ollama 필요 |
| 16 | `hw16_vlm_cad_field.py` | — (로컬 VLM) | 본인 PDF·사진 + 라벨 JSON 필요 |

## 처음부터 다시 하려면

```bash
rm -rf ../chroma_db   # 인덱스 삭제 → 다음 실행 시 자동 재인덱싱
rm -rf ../hw_cache    # 과제 해답의 LLM 캐시 삭제 (재실행 시 API 재호출)
```
