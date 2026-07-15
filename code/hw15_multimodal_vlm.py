# hw15_multimodal_vlm.py — 15회차 과제 해답
#
# 과제 1: Ollama + Qwen2.5-VL 캡션 생성 — 이미지 10장 품질 평가
# 과제 2: CLIP 텍스트→이미지 검색 (사진 30장 이상) — 성공/실패 질의 기록
# 과제 3: 패턴 ①(캡셔닝 인덱스) vs ②(CLIP)를 같은 질의로 비교표 작성
#
# 도면·현장 사진은 보안 특성상 저장소에 없다 (data/README.md) — 본인 이미지를
# 폴더에 넣고 --images로 지정하라. VLM은 로컬 실행이 원칙(보안 민감 자료):
#   brew install ollama && ollama pull qwen2.5vl:7b && pip install ollama
#
# 실행: python hw15_multimodal_vlm.py --images data/images [--task 1 2 3] [--list]

import glob
import os

from hw_common import REPO_ROOT, load_cache, print_table, run_tasks, save_cache, skip

CAPTION_PROMPT = ("이 현장 사진을 검색 가능하도록 상세히 설명하라. "
                  "장소 특징, 작업 내용, 장비, 인원, 안전 상태, 특이사항 포함.")

# 과제 2·3 공용 질의 — 본인 사진에 맞게 수정해서 사용
QUERIES = [
    "콘크리트 벽에 균열이 있는 사진",
    "안전모를 쓴 작업자",
    "용접 작업 장면",
    "배관과 밸브가 보이는 사진",
    "야간 또는 어두운 환경의 작업",
]


def _images(args, n_required: int) -> list[str] | None:
    paths = sorted(p for ext in ("jpg", "jpeg", "png")
                   for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    if len(paths) < n_required:
        skip(f"이미지가 {len(paths)}장뿐이다 (필요: {n_required}장 이상). "
             f"'{args.images}'에 본인 도메인 이미지를 넣고 --images로 지정하라.")
        return None
    return paths


def _ollama_ready(model: str) -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        skip(f"Ollama에 연결할 수 없다 — `ollama serve` 실행과 "
             f"`ollama pull {model}` 후 재시도하라.")
        return False


def _caption(model: str, path: str) -> str:
    import ollama
    r = ollama.chat(model=model, messages=[{
        "role": "user", "content": CAPTION_PROMPT, "images": [path]}])
    return r["message"]["content"]


def _captions(args, paths: list[str]) -> dict[str, str]:
    """경로→캡션 (캐시 — VLM 호출은 느리다)."""
    cache = load_cache("hw15_captions") or {}
    for p in paths:
        if p not in cache:
            cache[p] = _caption(args.model, p)
            save_cache("hw15_captions", cache)
            print(f"  캡션 생성: {p}")
    return {p: cache[p] for p in paths}


def task1(args):
    """이미지 10장 캡션 생성 + 품질 평가 기록 틀."""
    paths = _images(args, 10)
    if paths is None or not _ollama_ready(args.model):
        return
    captions = _captions(args, paths[:10])
    for p, c in captions.items():
        print(f"\n[{os.path.basename(p)}]\n  {c[:300]}")
    print("""
--- 품질 평가 기록 틀 (이미지마다 원본과 대조하며 채운다) ---
| 이미지 | 잘 읽은 것 | 틀린 것 | 오류 유형 |
오류 유형 분류 (16회차 환각 관리로 이어짐):
  숫자 오독 / 한국어 텍스트 오독 / 없는 것 생성 / 개수 오류 / 세밀 부분 누락
특히 숫자(치수·계기판·명판)는 반드시 확대해 원본과 대조하라 — VLM의 최약점.""")


def task2(args):
    """CLIP 텍스트→이미지 검색 — 질의별 top-3, 성공/실패 기록."""
    paths = _images(args, 30)
    if paths is None:
        return
    from PIL import Image
    from sentence_transformers import SentenceTransformer, util

    img_model = SentenceTransformer("clip-ViT-B-32")
    txt_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
    img_embs = img_model.encode([Image.open(p) for p in paths])

    for query in QUERIES:
        scores = util.cos_sim(txt_model.encode(query), img_embs)[0]
        print(f"\n질의: {query}")
        for i in scores.argsort(descending=True)[:3]:
            print(f"  [{scores[i]:.3f}] {paths[i]}")
    print("""
--- 성공/실패 기록 틀 (top-3를 눈으로 검증해 각 5개 이상 기록) ---
성공 질의: 시각적으로 뚜렷한 개념 (색·형태·장면 유형)이 대체로 잘 된다.
실패 질의의 전형:
  - 세밀한 수치 조건 ("균열 폭 0.3mm 이상") — CLIP은 수치를 못 읽는다
  - 텍스트 판독 ("PD-2000이라고 적힌") — 이미지 임베딩은 글자에 약하다
  - 부정 조건 ("안전모를 안 쓴") — 임베딩 공간에 부정이 없다
→ 실패 유형이 곧 패턴 ①(캡셔닝, 과제 3)과 16회차 구조화 분석의 도입 근거다.""")


def task3(args):
    """패턴 ① 캡셔닝 인덱스 vs ② CLIP — 같은 질의로 비교표."""
    paths = _images(args, 10)
    if paths is None or not _ollama_ready(args.model):
        return
    from PIL import Image
    from sentence_transformers import SentenceTransformer, util

    from rag_pipeline import SimpleRAG

    # 패턴 ①: VLM 캡션 → 텍스트 RAG 인덱싱 (5회차 SimpleRAG 재사용)
    captions = _captions(args, paths)
    rag = SimpleRAG(collection_name="hw15_captions")
    if rag.collection.count() != len(captions):
        docs = [{"id": p, "text": c, "source": p} for p, c in captions.items()]
        rag.collection.add(ids=[d["id"] for d in docs],
                           documents=[d["text"] for d in docs],
                           metadatas=[{"source": d["source"]} for d in docs])

    # 패턴 ②: CLIP 공동 임베딩
    img_model = SentenceTransformer("clip-ViT-B-32")
    txt_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
    img_embs = img_model.encode([Image.open(p) for p in paths])

    rows = []
    for q in QUERIES:
        r = rag.collection.query(query_texts=[q], n_results=1)
        cap_top = os.path.basename(r["ids"][0][0])
        scores = util.cos_sim(txt_model.encode(q), img_embs)[0]
        clip_top = os.path.basename(paths[int(scores.argmax())])
        rows.append([q, cap_top, clip_top, "동일" if cap_top == clip_top else "다름"])
    print_table(["질의", "① 캡셔닝 top-1", "② CLIP top-1", "일치"], rows)
    print("""
비교표 읽는 법 (top-1을 원본 이미지로 검증해 승자를 기록하라):
- ① 캡셔닝 승: 서술적·부정·수치 질의 — 캡션이라는 텍스트를 검색하므로
  6~9회차 기법(하이브리드·리랭킹·평가)이 전부 적용된다.
- ② CLIP 승: 시각적 유사성 질의, 그리고 캡션에 안 적힌 세부.
- 실무 정답은 조합 (15회차 결론): ①로 넓게 + ②로 시각 유사 —
  도면·문서는 ③ ColPali(16회차)로.""")


if __name__ == "__main__":
    run_tasks("15회차: 오픈소스 VLM·멀티모달 RAG", {
        1: ("Qwen2.5-VL 캡션 생성 — 10장 품질 평가", task1),
        2: ("CLIP 텍스트→이미지 검색 (30장+) 성공/실패 기록", task2),
        3: ("패턴 ① 캡셔닝 vs ② CLIP 비교표", task3),
    }, extra_args=[
        ("--images", {"default": str(REPO_ROOT / "data" / "images"), "metavar": "DIR",
                      "help": "본인 이미지 폴더 (기본: <저장소>/data/images)"}),
        ("--model", {"default": "qwen2.5vl:7b",
                     "help": "Ollama VLM 모델명 (기본: qwen2.5vl:7b)"}),
    ])
