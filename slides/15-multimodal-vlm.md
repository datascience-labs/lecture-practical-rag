---
marp: true
theme: default
paginate: true
header: '실용 RAG 강의 — 15회차'
footer: 'Practical RAG Course'
---

<!-- _class: lead -->

# 15회차: 오픈소스 VLM과 멀티모달 RAG

## 텍스트를 넘어 — 도면, 사진, 문서 이미지를 검색하고 이해하기


---

![bg](generated/s15-cover.png)

---

## 오늘의 목표

- 텍스트 RAG가 비텍스트 자료에서 실패하는 이유를 이해한다
- VLM(Vision-Language Model)의 구조와 오픈소스 생태계를 파악한다
- Ollama와 transformers로 오픈소스 VLM을 로컬 실행한다
- CLIP 계열 이미지 임베딩으로 텍스트→이미지 검색을 구현한다
- 멀티모달 RAG의 3가지 아키텍처 패턴을 비교한다

---

## 문제: 현실의 문서는 텍스트만이 아니다

지금까지의 파이프라인은 **텍스트 추출**을 전제:

| 자료 | pypdf 텍스트 추출 결과 |
|---|---|
| CAD 도면 PDF | 표제란 일부 글자만 조각조각 (치수·심볼·배치 전부 소실) |
| 현장 사진 | **아무것도 없음** |
| 설비 매뉴얼의 분해도 | 그림 소실, 캡션만 |
| 스캔된 계약서 | 없음 (OCR 필요, OCR도 표·서명 취약) |

```
"3층 배관 도면에서 밸브 위치 알려줘"   → 텍스트 인덱스에 도면이 없음
"어제 찍은 현장 사진 중 균열 있는 것?"  → 사진은 검색 자체가 불가
```

→ **이미지를 직접 이해하는 모델(VLM)과 이미지를 검색하는 임베딩**이 필요

---

## VLM(Vision-Language Model)이란

이미지를 "읽을 수 있는" LLM — 구조는 놀랍게 단순:

```
이미지 ─▶ [Vision Encoder (ViT)] ─▶ 패치 토큰들 ─▶ [Projector] ─┐
                                                                ▼
텍스트 ─▶ [토크나이저] ─────────────────────────────────▶ [LLM 본체] ─▶ 답변
```

- 이미지를 패치(격자 조각)로 잘라 **토큰처럼** LLM에 입력
- LLM 입장에서 이미지는 "또 하나의 언어"
- 그래서 할 수 있는 것: 이미지 설명, 이미지 QA, OCR, 표 읽기, 개수 세기, 구조화 추출

Claude도 VLM이지만(1회차부터 사용한 API), 오늘은 **오픈소스·로컬 실행**에 집중
— 도면·현장 사진은 외부 전송이 어려운 **보안 민감 자료**인 경우가 많기 때문

---

## 오픈소스 VLM 지형 (실무 관점)

| 모델 | 크기 | 라이선스 | 특징 |
|---|---|---|---|
| **Qwen2.5-VL** | 3B/7B/32B/72B | Apache-2.0 (72B 제외) | 문서·OCR·좌표 출력 강함, 한국어 우수 — **이 강의 선택** |
| InternVL3 | 1B~78B | MIT 계열 | 벤치마크 상위, 문서 이해 강함 |
| Gemma 3 | 4B/12B/27B | Gemma 라이선스 | 경량 대비 성능 우수 |
| LLaVA-OneVision | 0.5B~72B | Apache-2.0 | 연구 표준, 구조 이해에 좋음 |

선택 기준:

1. **한국어 지원** (도면 주기·현장 보고서는 한국어)
2. **문서/OCR 능력** (도면 표제란, 명판 읽기)
3. **라이선스** (상용 이용 가능 여부 확인 필수)
4. **하드웨어**: 7B 양자화 모델 ≈ VRAM/통합메모리 8GB 내외로 실행 가능

---

## 실행 방법 3가지

| 방법 | 장점 | 용도 |
|---|---|---|
| **Ollama** | 설치·실행 최간단, 자동 양자화 | 학습, 프로토타입 — **오늘 주력** |
| transformers | 세부 제어, 파인튜닝 연계 | 연구, 커스터마이즈 |
| vLLM | 고성능 서빙, OpenAI 호환 API | 프로덕션 |

```bash
# Ollama 설치 (macOS: brew install ollama)
ollama pull qwen2.5vl:7b     # 양자화된 Qwen2.5-VL 7B (~6GB)
pip install ollama            # Python 클라이언트
```

---

## 실습 1: 첫 VLM 호출 — 이미지 설명

```python
# vlm_first.py
import ollama

response = ollama.chat(
    model="qwen2.5vl:7b",
    messages=[{
        "role": "user",
        "content": "이 사진을 자세히 설명해줘. 사람, 장비, 작업 상황을 포함해서.",
        "images": ["site_photo.jpg"],       # 로컬 파일 경로
    }],
)
print(response["message"]["content"])
# "철골 구조물 위에서 두 명의 작업자가 용접 작업을 하고 있습니다.
#  왼쪽 작업자는 안전모와 안전대를 착용했으나..."
```

**해보기**: 현장 사진, 도면 캡처, 스크린샷 등 다양한 이미지로 시험
— 무엇을 잘 읽고 무엇을 틀리는지(특히 숫자!) 감을 잡는 것이 오늘의 절반

---

## 실습 2: transformers로 직접 실행 (세부 제어)

```python
# vlm_transformers.py — pip install transformers accelerate qwen-vl-utils
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "site_photo.jpg"},
    {"type": "text",  "text": "사진 속 작업자 수를 세고 안전모 착용 여부를 알려줘."},
]}]

text = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
images, _ = process_vision_info(messages)
inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                             skip_special_tokens=True)[0])
```

GPU 없으면 Ollama(양자화)로 충분 — 이 코드는 파인튜닝·서빙으로 갈 때의 기반

---

## 이미지 검색: CLIP 계열 임베딩

VLM은 이미지를 "이해"하지만, 수천 장에서 **찾는** 것은 임베딩의 일:

**CLIP / SigLIP**: 이미지와 텍스트를 **같은 벡터 공간**에 매핑하도록 학습

```
"안전모를 쓴 작업자" ──[텍스트 인코더]──▶ ●
                                          ↑ 가까움!
helmet_worker.jpg  ──[이미지 인코더]──▶ ●
crack_wall.jpg     ──[이미지 인코더]──▶       ● (멂)
```

- 2회차 텍스트 임베딩과 같은 원리 — 공간에 이미지가 추가된 것
- 한국어 질의를 쓰려면 **다국어 텍스트 인코더** 필요

---

## 실습 3: 텍스트로 사진 검색하기

```python
# image_search.py
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob

img_model = SentenceTransformer("clip-ViT-B-32")                       # 이미지 인코더
txt_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")       # 다국어 텍스트 인코더

paths = sorted(glob.glob("photos/*.jpg"))
img_embs = img_model.encode([Image.open(p) for p in paths])

query = "콘크리트 벽에 균열이 있는 사진"
q_emb = txt_model.encode(query)

scores = util.cos_sim(q_emb, img_embs)[0]
for i in scores.argsort(descending=True)[:3]:
    print(f"[{scores[i]:.3f}] {paths[i]}")
```

- ChromaDB에 이미지 임베딩을 넣으면 3회차와 동일하게 운영 가능
- 한계 체험: "**균열 폭 0.3mm 이상**인 사진" → CLIP은 세밀한 수치 구분 불가

---

## 멀티모달 RAG의 3가지 아키텍처 패턴

| 패턴 | 방식 | 장점 | 단점 |
|---|---|---|---|
| **① 캡셔닝** | VLM으로 이미지→텍스트 설명 생성 후 **텍스트 RAG에 인덱싱** | 기존 파이프라인 재사용, 하이브리드·리랭킹 전부 적용 가능 | 캡션에 없는 세부는 검색 불가 |
| **② 공동 임베딩** | CLIP/SigLIP으로 이미지 자체를 벡터화 | 캡션 불필요, 시각적 유사성 | 세밀한 텍스트·수치 못 읽음 |
| **③ 페이지 임베딩** | 문서 페이지를 통째로 이미지 취급 (**ColPali** 계열) | 레이아웃·표·도면 보존, OCR 불필요 | 저장 용량↑, 전용 모델 필요 |

**실무 정답은 조합**: ①로 넓게 검색 가능하게 + ②로 시각 유사 검색 + ③은 도면·문서
→ 다음 회차에서 ③(도면)과 ①(현장 사진)을 본격 구현

---

## 패턴 ① 미리보기: 캡셔닝 → 텍스트 RAG

```python
# caption_index.py — 5회차 SimpleRAG를 그대로 재사용!
def caption_photo(path: str) -> str:
    r = ollama.chat(model="qwen2.5vl:7b", messages=[{
        "role": "user",
        "content": "이 현장 사진을 검색 가능하도록 상세히 설명하라. "
                   "장소 특징, 작업 내용, 장비, 인원, 안전 상태, 특이사항 포함.",
        "images": [path],
    }])
    return r["message"]["content"]

rag = SimpleRAG(collection_name="photo_captions")
rag.index([
    {"id": p, "text": caption_photo(p), "source": p}
    for p in glob.glob("photos/*.jpg")
])
rag.ask("용접 작업 중 안전대 미착용이 보이는 사진은?")
# → 캡션이 검색되고, source가 원본 사진 경로를 가리킴
```

이미지가 **기존 텍스트 파이프라인의 일급 시민**이 됨 — 6~9회차 기법 전부 적용 가능

---

## 과제

1. Ollama로 Qwen2.5-VL 실행 환경 구축, 본인 도메인 이미지 10장으로 캡션 품질 평가
   (잘 읽는 것 / 틀리는 것 — 특히 숫자, 한국어 텍스트, 세밀한 부분)
2. 실습 3의 CLIP 검색을 사진 30장 이상으로 확장, 성공/실패 질의 각 5개 기록
3. 패턴 ① 캡셔닝 인덱스를 구축하고 CLIP 검색(②)과 같은 질의로 비교표 작성

## 다음 시간

**16회차: VLM 실전 — CAD 도면 검색과 현장 사진 분석** — ColPali로 도면을 검색하고, VLM으로 현장 사진을 구조화 분석한다
