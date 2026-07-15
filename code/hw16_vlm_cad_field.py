# hw16_vlm_cad_field.py — 16회차 과제 해답
#
# 과제 1: 도면 PDF 20페이지+로 ColQwen2 검색 구축, 질의 10개 Hit@3 측정
# 과제 2: 현장 사진 30장 구조화 점검 파이프라인 → 오탐/미탐 분석
# 과제 3: 수치 판독 정확도 실험 — 도면 치수 10개 질문 → 정답률 (환각 체감)
# 과제 4: (심화) 관심 영역 crop 후 재질문하는 2-pass QA로 과제 3 개선
#
# 도면·사진은 보안 특성상 저장소에 없다 — 본인 자료를 옵션으로 지정하라.
# 평가 라벨(질의→정답 페이지, 치수 질문→정답)은 JSON으로 준비한다:
# 파일이 없으면 작성용 템플릿을 hw_cache/에 만들어 준다. 전부 로컬 모델만 사용.
#
# 실행: python hw16_vlm_cad_field.py --pdf data/drawings.pdf --photos data/site_photos
#       [--task 1 2 3 4] [--list]

import glob
import json
import os
from pathlib import Path

from hw_common import (CACHE_DIR, REPO_ROOT, load_cache, print_table, run_tasks,
                       save_cache, skip)
from retrieval_metrics import hit_at_k

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


def _template(path: Path, content, guide: str):
    """라벨 파일이 없으면 템플릿을 만들어 주고 None 반환."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    skip(f"{guide} — 템플릿을 만들었다: {path} (채운 뒤 재실행)")
    return None


def _ollama_ready(model: str) -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        skip(f"Ollama에 연결할 수 없다 — `ollama serve`와 `ollama pull {model}` 후 재시도.")
        return False


def _vlm(model: str, prompt: str, image_path: str, force_json: bool = False):
    import ollama
    kwargs = {"format": "json"} if force_json else {}
    r = ollama.chat(model=model, messages=[{
        "role": "user", "content": prompt, "images": [image_path]}], **kwargs)
    return r["message"]["content"]


def _pdf_pages(args) -> list | None:
    if not os.path.exists(args.pdf):
        skip(f"도면 PDF가 없다: {args.pdf} — 본인 도면(또는 문서 스캔) PDF를 --pdf로 지정하라.")
        return None
    from pdf2image import convert_from_path
    pages = convert_from_path(args.pdf, dpi=150)      # 도면은 150dpi 이상 권장
    if len(pages) < 20:
        print(f"  참고: {len(pages)}페이지 (과제 요구: 20페이지 이상) — 그대로 진행한다.")
    return pages


def task1(args):
    """ColQwen2 도면 검색 + 질의 10개 Hit@3."""
    pages = _pdf_pages(args)
    if pages is None:
        return
    queries = _template(
        CACHE_DIR / "hw16_queries.json",
        [{"query": "기어박스 조립 도면", "answer_pages": [1]},
         {"query": "배관 계통도에서 밸브 배치", "answer_pages": []}],
        "질의 10개와 정답 페이지 라벨이 필요하다")
    if queries is None:
        return

    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0",
                                     torch_dtype=torch.bfloat16, device_map="auto")
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    emb_file = CACHE_DIR / "hw16_page_embeddings.pt"   # 인덱싱 산출물 캐시 (12회차 원칙)
    if emb_file.exists():
        page_embeddings = torch.load(emb_file, weights_only=True)
    else:
        page_embeddings = []
        for i, page in enumerate(pages):
            batch = processor.process_images([page]).to(model.device)
            with torch.no_grad():
                page_embeddings.append(model(**batch).cpu())
            print(f"  인덱싱: {i + 1}/{len(pages)}페이지")
        torch.save(page_embeddings, emb_file)

    batch_q = processor.process_queries([q["query"] for q in queries]).to(model.device)
    with torch.no_grad():
        q_embeddings = model(**batch_q).cpu()
    scores = processor.score_multi_vector(q_embeddings, page_embeddings)

    hits, rows = [], []
    for qi, q in enumerate(queries):
        top3 = [int(pi) + 1 for pi in scores[qi].argsort(descending=True)[:3]]
        if q["answer_pages"]:
            h = hit_at_k([str(p) for p in top3],
                         {str(p) for p in q["answer_pages"]}, 3)
            hits.append(h)
        pages[top3[0] - 1].save(CACHE_DIR / f"hw16_hit_q{qi}.png")  # 눈으로 검증!
        rows.append([q["query"][:30], str(top3),
                     str(q["answer_pages"]) if q["answer_pages"] else "(라벨 없음)"])
    print_table(["질의", "top-3 페이지", "정답 페이지"], rows)
    if hits:
        print(f"\nHit@3 = {sum(hits) / len(hits):.0%} ({len(hits)}문항)")
    print(f"top-1 페이지 이미지를 {CACHE_DIR}/hw16_hit_q*.png 로 저장했다 — 반드시 눈으로 검증."
          "\n관찰 포인트: 도면에 글자가 없어도 '그림으로' 매칭되는가? (ColPali의 존재 이유)")


def task2(args):
    """현장 사진 구조화 점검 → 검색 가능한 기록 → 오탐/미탐 분석."""
    photos = sorted(p for ext in ("jpg", "jpeg", "png")
                    for p in glob.glob(os.path.join(args.photos, f"*.{ext}")))
    if len(photos) < 1:
        skip(f"사진이 없다: {args.photos} — 현장/작업 사진 30장을 넣고 --photos로 지정하라.")
        return
    if not _ollama_ready(args.model):
        return

    reports = load_cache("hw16_inspections") or {}
    for p in photos:
        if p not in reports:
            try:
                reports[p] = json.loads(_vlm(args.model, INSPECT_PROMPT, p,
                                             force_json=True))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  파싱 실패({p}): {e} — 건너뜀")
                continue
            save_cache("hw16_inspections", reports)
            print(f"  점검: {p}")

    sev_order = ["none", "low", "medium", "high"]
    rows = []
    for p, rep in reports.items():
        max_sev = max((h.get("severity", "low") for h in rep.get("hazards", [])),
                      default="none", key=sev_order.index)
        rows.append([os.path.basename(p), rep.get("workers", "?"), max_sev,
                     (rep.get("scene", "") or "")[:36]])
    print_table(["사진", "인원", "최대 위험도", "장면"], rows)

    labels = _template(
        CACHE_DIR / "hw16_photo_labels.json",
        {os.path.basename(photos[0]): {"hazard_expected": True,
                                       "note": "실제 위험 요소를 사람이 기록"}},
        "오탐/미탐 분석에는 사람이 만든 정답 라벨이 필요하다")
    if labels:
        fp = fn = 0
        for p, rep in reports.items():
            name = os.path.basename(p)
            if name not in labels:
                continue
            detected = bool(rep.get("hazards"))
            expected = labels[name]["hazard_expected"]
            if detected and not expected:
                fp += 1
                print(f"  오탐(FP): {name} — 없는 위험을 보고")
            if expected and not detected:
                fn += 1
                print(f"  미탐(FN): {name} — 실제 위험을 놓침")
        print(f"\n라벨 {len(labels)}장 기준: 오탐 {fp} / 미탐 {fn}")
    print("""
분석 기록 포인트 (16회차 환각 관리 표와 대조):
- 오탐: 그림자·얼룩을 균열로 — 심각도 임계 판단에 VLM 단독 사용 금지의 근거.
- 미탐: 화면 구석·역광의 위험 요소 — 촬영 가이드(구도·조도)로 줄이는 것이 먼저다.
- confidence_notes가 채워진 사진부터 사람이 검토 — "불확실을 말하게 하는" 프롬프트의 값어치.""")


def _dim_questions(args):
    return _template(
        CACHE_DIR / "hw16_dim_questions.json",
        [{"image": "hw_cache/hw16_hit_q0.png",
          "question": "구동축의 직경 치수는?", "answer": "40"}],
        "치수 질문 10개(image/question/answer)가 필요하다")


def task3(args):
    """수치 판독 정확도 — 치수 10개 1-pass 질문 → 정답률."""
    if not _ollama_ready(args.model):
        return
    questions = _dim_questions(args)
    if questions is None:
        return

    correct, rows = 0, []
    answers = {}
    for i, case in enumerate(questions):
        prompt = (f"이 CAD 도면을 보고 질문에 답하라. 값만 짧게 답하고, "
                  f"확실히 읽을 수 없으면 '판독 불가'라고 답하라. 추측 금지.\n"
                  f"질문: {case['question']}")
        got = _vlm(args.model, prompt, case["image"]).strip()
        answers[str(i)] = got
        ok = case["answer"].replace(",", "") in got.replace(",", "")
        correct += ok
        rows.append([case["question"][:30], got[:24], case["answer"], "✓" if ok else "✗"])
    save_cache("hw16_dim_1pass", answers)
    print_table(["질문", "VLM 답변", "정답", "판정"], rows)
    print(f"\n1-pass 정답률: {correct}/{len(questions)}")
    print("""
환각 체감 포인트:
- 오답의 전형: 1,250 → 1,260 같은 한 자리 오독, 다른 치수선의 값을 가져오는 혼동.
- 이 실험이 16회차 원칙의 근거다: 치수·규격 답변은 반드시 근거 영역과 함께
  사람에게 전달하고, 고위험 결정(발주·시공)에 VLM 단독 판독을 쓰지 않는다.
→ 과제 4(2-pass crop QA)로 정답률이 얼마나 회복되는지 이어서 측정하라.""")


def task4(args):
    """(심화) 2-pass QA — VLM에게 관심 영역 좌표를 받아 crop 후 재질문."""
    if not _ollama_ready(args.model):
        return
    questions = _dim_questions(args)
    if questions is None:
        return
    from PIL import Image

    one_pass = load_cache("hw16_dim_1pass") or {}
    correct, rows = 0, []
    for i, case in enumerate(questions):
        img = Image.open(case["image"])
        w, h = img.size
        # 1-pass: 질문과 관련된 영역의 좌표를 요청 (Qwen2.5-VL은 좌표 출력에 강함)
        loc_prompt = (f"질문과 관련된 영역의 픽셀 좌표를 JSON으로만 출력하라: "
                      f'{{"bbox": [x1, y1, x2, y2]}} (이미지 크기 {w}x{h})\n'
                      f"질문: {case['question']}")
        try:
            bbox = json.loads(_vlm(args.model, loc_prompt, case["image"],
                                   force_json=True))["bbox"]
            m = 40                                     # 여유 마진
            crop = img.crop((max(bbox[0] - m, 0), max(bbox[1] - m, 0),
                             min(bbox[2] + m, w), min(bbox[3] + m, h)))
        except Exception as e:
            print(f"  좌표 실패({case['question'][:20]}): {e} — 원본으로 대체")
            crop = img
        crop_path = str(CACHE_DIR / f"hw16_crop_{i}.png")
        crop.save(crop_path)
        # 2-pass: 확대된 crop에 재질문
        got = _vlm(args.model,
                   f"이 확대 이미지에서 값만 짧게 답하라. 읽을 수 없으면 '판독 불가'.\n"
                   f"질문: {case['question']}", crop_path).strip()
        ok = case["answer"].replace(",", "") in got.replace(",", "")
        correct += ok
        rows.append([case["question"][:26], one_pass.get(str(i), "-")[:18],
                     got[:18], case["answer"], "✓" if ok else "✗"])

    print_table(["질문", "1-pass", "2-pass", "정답", "2-pass 판정"], rows)
    print(f"\n2-pass 정답률: {correct}/{len(questions)} — 과제 3의 1-pass와 비교하라.")
    print("""
해석:
- 개선 원인: 해상도 — VLM 비전 인코더는 전체 도면을 패치로 뭉개므로 작은 치수
  글자가 소실된다. crop은 같은 픽셀 예산을 관심 영역에 몰아준다.
- crop 자체가 틀리면(1-pass 좌표 오류) 2-pass도 실패 — hw_cache/hw16_crop_*.png를
  열어 좌표 품질부터 검증하라. 실무에선 좌표를 사람이 찍는 반자동 UI가 안전하다.""")


if __name__ == "__main__":
    run_tasks("16회차: VLM 실전 — 도면 검색·현장 사진", {
        1: ("ColQwen2 도면 검색 + Hit@3 측정", task1),
        2: ("현장 사진 구조화 점검 → 오탐/미탐 분석", task2),
        3: ("치수 판독 정확도 실험 (1-pass)", task3),
        4: ("(심화) crop 2-pass QA로 정답률 개선", task4),
    }, extra_args=[
        ("--pdf", {"default": str(REPO_ROOT / "data" / "drawings.pdf"), "metavar": "PDF",
                   "help": "도면(또는 스캔 문서) PDF 경로"}),
        ("--photos", {"default": str(REPO_ROOT / "data" / "site_photos"), "metavar": "DIR",
                      "help": "현장 사진 폴더"}),
        ("--model", {"default": "qwen2.5vl:7b",
                     "help": "Ollama VLM 모델명"}),
    ])
