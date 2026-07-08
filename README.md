# Practical RAG — 실용 RAG 17회차 강의

RAG(Retrieval-Augmented Generation) 기초부터 GraphRAG · OntoRAG · NodeRAG,
그리고 오픈소스 VLM 기반 멀티모달 RAG(CAD 도면 검색, 현장 사진 분석)까지
실습 중심으로 배우는 17회차 강의자료입니다. 모든 슬라이드는 [Marp](https://marp.app/) 형식의 Markdown으로 작성되어 있습니다.

## 커리큘럼

| 회차 | 파일 | 주제 |
|---|---|---|
| 1 | [slides/01-rag-overview.md](slides/01-rag-overview.md) | RAG 개요와 LLM의 한계 |
| 2 | [slides/02-embeddings.md](slides/02-embeddings.md) | 임베딩과 벡터 표현 |
| 3 | [slides/03-vector-databases.md](slides/03-vector-databases.md) | 벡터 데이터베이스 (ChromaDB) |
| 4 | [slides/04-chunking.md](slides/04-chunking.md) | 문서 로딩과 청킹 전략 |
| 5 | [slides/05-basic-rag-pipeline.md](slides/05-basic-rag-pipeline.md) | 기본 RAG 파이프라인 구축 |
| 6 | [slides/06-hybrid-search.md](slides/06-hybrid-search.md) | 하이브리드 검색 (BM25 + RRF) |
| 7 | [slides/07-reranking.md](slides/07-reranking.md) | 리랭킹 (Cross-encoder) |
| 8 | [slides/08-query-transformation.md](slides/08-query-transformation.md) | 쿼리 변환과 고급 RAG 패턴 |
| 9 | [slides/09-evaluation.md](slides/09-evaluation.md) | RAG 평가 (Hit@k, MRR, RAGAS) |
| 10 | [slides/10-knowledge-graphs.md](slides/10-knowledge-graphs.md) | 지식 그래프 기초 |
| 11 | [slides/11-graphrag-concepts.md](slides/11-graphrag-concepts.md) | GraphRAG I — 개념과 아키텍처 |
| 12 | [slides/12-graphrag-hands-on.md](slides/12-graphrag-hands-on.md) | GraphRAG II — 실습 |
| 13 | [slides/13-ontorag.md](slides/13-ontorag.md) | OntoRAG — 온톨로지 기반 RAG |
| 14 | [slides/14-noderag.md](slides/14-noderag.md) | NodeRAG — 이종 노드 그래프 RAG |
| 15 | [slides/15-multimodal-vlm.md](slides/15-multimodal-vlm.md) | 오픈소스 VLM과 멀티모달 RAG |
| 16 | [slides/16-vlm-cad-field.md](slides/16-vlm-cad-field.md) | VLM 실전 — CAD 도면 검색과 현장 사진 분석 |
| 17 | [slides/17-capstone-production.md](slides/17-capstone-production.md) | 종합 프로젝트 — 프로덕션 RAG |

## 슬라이드 보기 / 내보내기

Marp CLI 사용 (Node.js 필요):

```bash
# 실시간 프리뷰 (브라우저)
npx @marp-team/marp-cli -s slides/

# 단일 파일 → HTML
npx @marp-team/marp-cli slides/01-rag-overview.md -o out/01.html

# 전체 → PDF
npx @marp-team/marp-cli slides/*.md --pdf -o out/
```

VS Code 사용자는 **Marp for VS Code** 확장을 설치하면 미리보기가 바로 됩니다.

## 실습 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."   # https://platform.claude.com 에서 발급

# 슬라이드 커버 이미지 Batch 생성 시
export OPENAI_API_KEY="sk-..."
# 선택: 사내 게이트웨이/프록시 사용 시
# export OPENAI_BASE_URL="https://your-proxy.example.com/v1"
# export OPENAI_IMAGE_MODEL="gpt-image-2"
```

15~16회차(멀티모달)는 추가로 [Ollama](https://ollama.com)와 poppler가 필요합니다:

```bash
brew install ollama poppler      # macOS (Linux: 패키지 매니저 사용)
ollama pull qwen2.5vl:7b
```

기술 스택:

- **LLM**: Claude API (`claude-opus-4-8`, `anthropic` Python SDK)
- **임베딩**: `BAAI/bge-m3` (sentence-transformers, 로컬 무료)
- **벡터 DB**: ChromaDB
- **리랭커**: `BAAI/bge-reranker-v2-m3`
- **한국어 형태소 분석**: kiwipiepy
- **그래프**: NetworkX
- **오픈소스 VLM** (15~16회차): Qwen2.5-VL (Ollama / transformers, 로컬 실행)
- **이미지·문서 검색** (15~16회차): CLIP (sentence-transformers), ColQwen2 (colpali-engine)

## 슬라이드 커버 이미지 Batch 생성

[slides/image-prompts.md](slides/image-prompts.md)에 정리된 프롬프트를 OpenAI Batch API용 JSONL로 변환하고, 배치 제출 후 결과 PNG를 저장할 수 있습니다.

```bash
python scripts/openai_batch_images.py prepare
python scripts/openai_batch_images.py submit
python scripts/openai_batch_images.py status
python scripts/openai_batch_images.py download
```

기본 동작:

- 배치 요청 크기: 2048x1152
- 저장 크기: 1920x1080 PNG
- 결과 저장 위치: [slides/generated](slides/generated)
- 배치 메타데이터: [out/openai](out/openai)

사용 가능한 환경 변수:

- `OPENAI_API_KEY`: 필수
- `OPENAI_BASE_URL`: 선택, 프록시/호환 게이트웨이 사용 시
- `OPENAI_ORG_ID`: 선택
- `OPENAI_PROJECT`: 선택
- `OPENAI_IMAGE_MODEL`: 기본 `gpt-image-2`
- `OPENAI_IMAGE_REQUEST_SIZE`: 기본 `2048x1152`
- `OPENAI_IMAGE_EXPORT_SIZE`: 기본 `1920x1080`
- `OPENAI_IMAGE_QUALITY`: 기본 `medium`
- `OPENAI_IMAGE_BACKGROUND`: 기본 `auto`
- `OPENAI_IMAGE_MODERATION`: 기본 `auto`
- `OPENAI_IMAGE_OUTPUT_FORMAT`: 기본 `png`

## 진행 방식

- 각 회차는 이론 절반 + 실습 절반으로 구성되어 있습니다.
- 실습 코드는 슬라이드에 포함되어 있으며, 회차가 진행되며 하나의 파이프라인으로 누적 확장됩니다 (5회차 `SimpleRAG` → 7회차 `RerankedRAG` → 12회차 미니 GraphRAG → 14회차 미니 NodeRAG → 15~16회차 멀티모달 확장).
- 도면·현장 사진 같은 보안 민감 자료를 다루는 15~16회차는 전 과정이 로컬 오픈소스 모델로 구성되어 외부 API 전송이 없습니다.
- 5회차부터 본인 도메인 문서로 중간 프로젝트를 시작해 17회차에 발표합니다.
