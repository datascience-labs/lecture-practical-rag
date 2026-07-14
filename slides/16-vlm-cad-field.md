---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 16회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 16회차: VLM 실전 — CAD 도면 검색과 현장 사진 분석

## 엔지니어링 현장의 두 가지 실전 시나리오


---

![bg](generated/s16-cover.png)

---

## 오늘의 목표

- ColPali/ColQwen2로 CAD 도면 PDF를 **이미지 그대로** 검색한다
- 검색된 도면 페이지를 VLM에 투입해 표제란·치수 QA를 수행한다
- VLM 구조화 출력으로 현장 사진 안전 점검을 자동화한다
- 사진 분석 결과를 RAG에 환류해 "검색 가능한 현장 기록"을 만든다
- VLM 환각(수치 오독) 리스크와 검증 워크플로를 설계한다

---

## 시나리오 A: CAD 도면 검색

**현장의 요구**: "기어박스 조립도 어디 있지?" — 수백 장의 도면 PDF에서

도면이 텍스트 RAG에서 전멸하는 이유:

```
┌────────────────────────────────┐
│    [평면도 그림]      ┌─────┐  │   pypdf 추출 결과:
│   치수선 ←—— 1,250 ——→│ BOM │  │   "1,250 M8 SUS304 GB-A-102 1:20"
│    ⌀40 심볼, 용접기호  │ 표  │  │
│ ┌──────────────────────┴─────┐│   ← 배치·심볼·그림 전부 소실,
│ │표제란: GB-A-102  축척 1:20  ││      숫자만 맥락 없이 나열됨
│ └────────────────────────────┘│
└────────────────────────────────┘
```

- 도면의 정보는 **공간 배치와 심볼**에 있음 — 텍스트 추출·OCR로는 파괴됨
- 참고: DWG/DXF 원본은 구조화 파싱(ezdxf)이 가능하지만, 현장에 유통되는 것은 대부분 PDF/스캔본

→ 해법: **페이지를 이미지 그대로 임베딩** (15회차 패턴 ③)

---

## ColPali: 페이지 이미지의 다중 벡터 검색

> "ColPali: Efficient Document Retrieval with Vision Language Models" (2024)

```
기존:  페이지 → OCR → 청킹 → 텍스트 임베딩 1개        (파이프라인 김, 손실 큼)
ColPali: 페이지 이미지 → VLM 비전 인코더 → 패치별 벡터 ~1000개  (한 방)

질의 처리 (Late Interaction, MaxSim):
  질의 토큰 각각이 페이지 패치들과 최대 유사도를 계산 → 합산
  "기어박스" 토큰 ↔ 도면에서 기어가 그려진 패치가 강하게 반응
```

- 7회차 리랭커(cross-encoder)와 bi-encoder의 중간형 — 정밀하면서도 사전 인덱싱 가능
- **ColQwen2**: Qwen2-VL 기반 구현 — 다국어(한국어) 질의 대응이 상대적으로 좋음
- OCR·레이아웃 분석·청킹이 전부 사라짐 → 도면·표·슬라이드에 특히 강력

---

## 실습 A-1: 도면 PDF 인덱싱

```python
# cad_index.py — pip install colpali-engine pdf2image  (+ poppler 설치)
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16,
    device_map="auto")                    # GPU/Apple Silicon 권장
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

# 도면 PDF → 페이지 이미지 (도면은 dpi 150 이상 권장)
pages = convert_from_path("drawings.pdf", dpi=150)

# 페이지별 다중 벡터 임베딩
page_embeddings = []
for page in pages:
    batch = processor.process_images([page]).to(model.device)
    with torch.no_grad():
        page_embeddings.append(model(**batch).cpu())
print(f"{len(pages)}페이지 인덱싱 완료")
```

인덱싱 산출물(임베딩 텐서)은 파일로 저장해 재사용 — 12회차 캐시 원칙 동일

---

## 실습 A-2: 자연어로 도면 검색

```python
# cad_search.py
queries = [
    "기어박스 조립 도면",
    "배관 계통도에서 밸브 배치",
    "볼트 체결 상세와 토크 값이 있는 페이지",
]

batch_q = processor.process_queries(queries).to(model.device)
with torch.no_grad():
    q_embeddings = model(**batch_q).cpu()

# MaxSim 점수: 질의 × 페이지 행렬
scores = processor.score_multi_vector(q_embeddings, page_embeddings)

for qi, query in enumerate(queries):
    top = scores[qi].argsort(descending=True)[:3]
    print(f"\n질문: {query}")
    for rank, pi in enumerate(top, 1):
        print(f"  {rank}위: {pi+1}페이지 (점수 {scores[qi][pi]:.1f})")
        pages[pi].save(f"hit_q{qi}_r{rank}.png")   # 눈으로 검증!
```

**관찰 포인트**: 도면에 "기어박스"라는 글자가 없어도 **그림으로** 매칭되는가?

---

## 실습 A-3: 검색된 도면을 VLM으로 QA

검색(ColQwen2)과 이해(VLM)의 2단계 — 7회차 retrieve→rerank 구조와 동일한 철학:

```python
# cad_qa.py
import ollama

def drawing_qa(page_image_path: str, question: str) -> str:
    r = ollama.chat(model="qwen2.5vl:7b", messages=[{
        "role": "user",
        "content": f"""이 CAD 도면을 보고 질문에 답하라.
- 표제란(도면 우하단 표)의 정보를 우선 확인하라.
- 확실히 읽을 수 없는 값은 '판독 불가'라고 답하라. 추측 금지.
질문: {question}""",
        "images": [page_image_path],
    }])
    return r["message"]["content"]

print(drawing_qa("hit_q0_r1.png", "도면 번호와 축척, 재질을 알려줘"))
# "도면 번호: GB-A-102, 축척: 1:20, 재질: SUS304 (표제란 기준)"
print(drawing_qa("hit_q0_r1.png", "구동축의 직경 치수는?"))
```

⚠️ **치수 답변은 반드시 원본 대조** — 뒤의 '환각 관리' 슬라이드에서 상세히

---

## 시나리오 B: 현장 사진 분석

**현장의 요구**:

- 매일 수백 장 올라오는 시공 사진 — 안전 위반을 사람이 전수 검사 불가
- "지난달 3공구에서 균열 발견된 사진 다 모아줘" — 검색 불가능한 사진 더미

전략: VLM으로 사진마다 **구조화 점검 데이터(JSON)** 를 생성
→ 메타데이터와 함께 인덱싱 → 검색·집계·알림이 전부 가능해짐

```
사진 → [VLM 구조화 분석] → JSON (안전/품질 체크리스트)
                              ├─▶ 심각도 높음 → 즉시 알림
                              └─▶ ChromaDB 인덱싱 → 자연어 검색/통계
```

---

## 실습 B-1: 구조화 안전 점검

```python
# site_inspect.py
import ollama, json

INSPECT_PROMPT = """건설/제조 현장 사진을 점검해 JSON으로만 출력하라:
{
 "scene": "장면 한 줄 설명",
 "workers": 사진 속 인원 수,
 "ppe": {"helmet": "all|partial|none|n/a", "harness": "...", "vest": "..."},
 "hazards": [{"type": "떨어짐|끼임|감전|화재|낙하물|기타",
              "description": "...", "severity": "high|medium|low"}],
 "quality_issues": ["균열, 누수, 배근 불량 등 관찰된 품질 문제"],
 "confidence_notes": "판단이 불확실한 항목과 이유"
}
보이지 않는 것은 추측하지 말고 n/a 또는 빈 배열로 두라."""

def inspect_photo(path: str) -> dict:
    r = ollama.chat(model="qwen2.5vl:7b",
                    messages=[{"role": "user", "content": INSPECT_PROMPT,
                               "images": [path]}],
                    format="json")               # JSON 모드 강제
    return json.loads(r["message"]["content"])

report = inspect_photo("photos/site_0142.jpg")
print(json.dumps(report, ensure_ascii=False, indent=2))
```

---

## 실습 B-2: 분석 결과를 검색 가능한 기록으로

```python
# site_index.py — 3회차 메타데이터 필터 + 5회차 RAG의 재사용
collection = db.get_or_create_collection("site_photos", embedding_function=ef)

for path in glob.glob("photos/*.jpg"):
    rep = inspect_photo(path)
    max_sev = max((h["severity"] for h in rep["hazards"]),
                  default="none", key=["none","low","medium","high"].index)
    # 검색용 텍스트: 장면 설명 + 위험/품질 서술을 하나로
    doc = rep["scene"] + " " + " ".join(
        h["description"] for h in rep["hazards"]) + " " + " ".join(rep["quality_issues"])
    collection.add(
        ids=[path], documents=[doc],
        metadatas=[{"path": path, "severity": max_sev,
                    "workers": rep["workers"],
                    "site": "3공구", "date": "2026-07-04"}])   # 실무: EXIF/폴더에서 추출

# 자연어 + 필터 결합 검색
results = collection.query(
    query_texts=["콘크리트 균열"],
    where={"severity": {"$in": ["high", "medium"]}},
    n_results=5)
```

"위험도 높은 균열 사진"이 **한 번의 질의**로 — 사진이 데이터가 되는 순간

---

## 실습 B-3: 현장 질의응답 봇 완성

```python
# site_qa.py — 검색된 사진들을 다시 VLM에 넣어 종합 답변
def site_qa(question: str) -> str:
    hits = collection.query(query_texts=[question], n_results=3)
    photo_paths = [m["path"] for m in hits["metadatas"][0]]

    r = ollama.chat(model="qwen2.5vl:7b", messages=[{
        "role": "user",
        "content": f"""다음 현장 사진들을 근거로 질문에 답하라.
각 판단이 몇 번째 사진에 근거하는지 명시하라.
질문: {question}""",
        "images": photo_paths,                  # 여러 장 동시 투입
    }])
    return r["message"]["content"] + f"\n\n[근거 사진: {photo_paths}]"

print(site_qa("3공구 안전 상태에서 가장 시급히 조치할 것은?"))
```

검색(임베딩+필터) → 근거 수집(사진) → 생성(VLM) — **RAG의 R·A·G가 전부 멀티모달로**

---

## VLM 환각 관리 — 이 파이프라인의 최대 리스크

| 위험 | 예시 | 대응 |
|---|---|---|
| **수치 오독** | 치수 1,250을 1,260으로 | 치수·규격 답변엔 근거 영역 crop을 함께 제시, 사람 확인 필수 단계화 |
| 없는 것 생성 | 없는 밸브를 "있다"고 답변 | 프롬프트에 "추측 금지, 판독 불가 명시" + `confidence_notes` 요구 |
| 해상도 부족 | 작은 글씨 오독 | 도면 dpi 200+, 필요시 관심 영역 확대 재질문 (2-pass) |
| 개수 오류 | 작업자 5명을 4명으로 | 집계는 참고치로 취급, 임계 판단엔 미사용 |

**설계 원칙**: VLM 출력은 "후보 판단"이고, 고위험 결정(안전 조치, 발주)은
반드시 **원본 이미지 링크와 함께 사람에게** 전달 — 5회차 출처 표기 원칙의 멀티모달판

평가도 동일(9회차): 정답 라벨이 있는 사진/도면 평가셋을 만들어 정확도를 **측정**하라

---

## 시나리오별 아키텍처 선택 정리

| 요구 | 선택 |
|---|---|
| 도면·표·스캔 문서를 통째로 검색 | ColPali/ColQwen2 (패턴 ③) |
| 도면 내용 QA (표제란, 치수) | 검색 후 VLM QA (2단계) |
| 사진의 시각적 유사 검색 | CLIP/SigLIP (패턴 ②, 15회차) |
| 사진 내용의 서술적 검색·집계 | VLM 구조화 분석 → 텍스트 인덱싱 (패턴 ①) |
| 사진+문서 통합 질문 | 텍스트 RAG와 컬렉션 병렬 검색 → 통합 프롬프트 |

전부 로컬 오픈소스로 구성 가능 — **보안 민감 도면·사진이 외부로 나가지 않음**

---

## 과제

1. 도면(또는 문서 스캔) PDF 20페이지 이상으로 ColQwen2 검색 구축, 질의 10개 Hit@3 측정
2. 현장/작업 사진 30장으로 구조화 점검 파이프라인 실행 → 오탐/미탐 사례 분석
3. 수치 판독 정확도 실험: 도면 치수 10개를 VLM에 질문 → 정답률 기록 (환각 체감)
4. (심화) 관심 영역 crop 후 재질문하는 2-pass QA로 3번 정답률 개선 시도

## 다음 시간

**17회차 (최종): 종합 프로젝트** — 텍스트·그래프·멀티모달을 아우르는 프로덕션 RAG 설계와 발표
