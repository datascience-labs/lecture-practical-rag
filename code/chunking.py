# chunking.py — 4회차: 문서 로딩과 청킹 전략
#
# 슬라이드의 네 가지 청킹 전략:
#   fixed_chunk     전략 1: 고정 크기 청킹
#   recursive_chunk 전략 2: 재귀적 문자 분할 (가장 널리 쓰이는 기본값)
#   markdown_chunk  전략 3: 구조 기반 청킹 (## 헤더 단위)
#   semantic_chunk  전략 4: 의미 기반 청킹 (문장 임베딩 유사도)
#
# 실행: python chunking.py            # 전략 1~3을 sample_handbook.md에 적용
#       python chunking.py --semantic # 전략 4까지 (bge-m3 다운로드 필요)

from __future__ import annotations

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fixed_chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """전략 1: 고정 크기 청킹 — 간단하지만 문장/단어 중간에서 잘린다."""
    assert 0 <= overlap < size, "overlap은 size보다 작아야 한다"
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def recursive_chunk(text: str, size: int = 500,
                    seps: list[str] | None = None) -> list[str]:
    """전략 2: 재귀적 문자 분할 — 큰 구분자(문단)부터 시도하고,
    여전히 큰 청크는 다음 구분자(줄 → 문장 → 단어)로 재귀 분할한다."""
    if seps is None:
        seps = ["\n\n", "\n", ". ", " "]
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep in text:
            parts = text.split(sep)
            chunks, buf = [], ""
            for p in parts:
                if len(buf) + len(p) + len(sep) <= size:
                    buf += p + sep
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = p + sep
            if buf:
                chunks.append(buf.strip())
            # 여전히 큰 청크는 다음 구분자로 재귀 분할
            return [c for ch in chunks
                    for c in (recursive_chunk(ch, size, seps[1:])
                              if len(ch) > size else [ch])]
    # 어떤 구분자도 없으면 문자 단위로 자른다
    return [text[i:i + size] for i in range(0, len(text), size)]


def markdown_chunk(text: str) -> list[dict]:
    """전략 3: ## 헤더 단위로 분할하고, 헤더를 메타데이터로 보존."""
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    chunks = []
    for sec in sections:
        if not sec.strip():
            continue
        header = sec.split("\n")[0].lstrip("# ").strip()
        chunks.append({"text": sec.strip(), "section": header})
    return chunks


def semantic_chunk(sentences: list[str], threshold: float = 0.5) -> list[str]:
    """전략 4: 인접 문장 간 임베딩 유사도가 급락하는 지점(주제 전환점)에서 분할.

    인덱싱 시 임베딩 비용이 늘어나는 대신, 비정형 문서에서 효과가 크다.
    """
    from sentence_transformers import SentenceTransformer, util  # 무거운 의존성이라 지연 로딩

    model = SentenceTransformer("BAAI/bge-m3")
    embs = model.encode(sentences)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = util.cos_sim(embs[i - 1], embs[i]).item()
        if sim < threshold:          # 주제가 바뀌는 지점
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    import sys

    document = (DATA_DIR / "sample_handbook.md").read_text(encoding="utf-8")
    print(f"원문: {len(document):,}자 (sample_handbook.md)\n")

    for name, chunks in [
        ("fixed_300", fixed_chunk(document, size=300, overlap=30)),
        ("fixed_800", fixed_chunk(document, size=800, overlap=80)),
        ("recursive_500", recursive_chunk(document, size=500)),
    ]:
        sizes = [len(c) for c in chunks]
        print(f"{name:>14}: {len(chunks):3d}개 청크 "
              f"(평균 {sum(sizes) // len(sizes)}자, 최대 {max(sizes)}자)")

    md = markdown_chunk(document)
    print(f"{'markdown':>14}: {len(md):3d}개 청크 (섹션 단위)")
    for c in md[:5]:
        print(f"                - [{c['section']}] {len(c['text'])}자")

    # 고정 크기 청킹의 문제: 문장 중간 절단을 직접 확인
    print("\n--- fixed_300의 두 번째 청크 끝부분 (문장이 잘리는가?) ---")
    print("..." + fixed_chunk(document, size=300, overlap=30)[1][-80:])

    if "--semantic" in sys.argv:
        print("\n--- semantic_chunk (bge-m3 로딩, 최초 실행 시 다운로드) ---")
        paragraph = document.split("## 제3장")[1][:1500]
        sentences = [s.strip() for s in re.split(r"(?<=다\.)\s+", paragraph) if s.strip()]
        for i, ch in enumerate(semantic_chunk(sentences, threshold=0.5)):
            print(f"[의미청크 {i}] {ch[:70]}...")
