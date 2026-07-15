# corpus.py — 공용 모듈: 실습 코퍼스 로딩 (5~9회차에서 공유)
#
# 실습 데이터 두 종류를 청크 리스트로 만든다:
#   data/sample_handbook.md   → ### 소절 단위 청크 + 헤더 경로 주입 (4회차 심화 방식)
#   data/company_docs/*.txt   → 파일당 1청크 (짧은 사실 문서)
#
# 청크 id는 내용 기반이라 재인덱싱해도 안정적이다:
#   handbook-3.1 (소절 번호), 04-product-xk2000 (파일명)
# → data/eval_set.json의 answer_chunk_ids가 이 id를 가리킨다 (7·9회차 평가).

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

HANDBOOK = "sample_handbook.md"


def load_handbook_chunks() -> list[dict]:
    """핸드북을 ### 소절 단위로 분할. 각 청크 앞에 '문서 > 장 > 절' 경로를 주입한다.

    4회차에서 배운 구조 기반 청킹 + 헤더 경로 주입의 실전 적용.
    """
    text = (DATA_DIR / HANDBOOK).read_text(encoding="utf-8")
    chapter = ""
    chunks: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            title = line[4:].strip()
            num = title.split()[0]                     # "3.1 연차 휴가" → "3.1"
            current = {
                "id": f"handbook-{num}",
                "text": f"[임직원 핸드북 > {chapter} > {title}]",
                "source": HANDBOOK,
            }
            chunks.append(current)
        elif line.startswith("## "):
            chapter = line[3:].strip()                 # 장 제목은 경로에만 사용
            current = None
        elif line.strip() == "---" and chunks:
            break                                      # 문서 끝 각주는 제외
        elif current is not None:
            current["text"] += "\n" + line
    for c in chunks:
        c["text"] = c["text"].strip()
    return chunks


def load_company_chunks() -> list[dict]:
    """회사 문서(제품·조직·고객 등)는 문서가 짧아 파일당 1청크로 충분하다."""
    return [
        {"id": p.stem, "text": p.read_text(encoding="utf-8").strip(), "source": p.name}
        for p in sorted((DATA_DIR / "company_docs").glob("*.txt"))
    ]


def load_corpus() -> list[dict]:
    """전체 실습 코퍼스: [{"id": ..., "text": ..., "source": ...}, ...]"""
    return load_handbook_chunks() + load_company_chunks()


def ensure_indexed(rag) -> None:
    """컬렉션이 비어 있으면 전체 코퍼스를 인덱싱한다 (최초 1회만 수행됨)."""
    if rag.collection.count() == 0:
        rag.index(load_corpus())


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"총 {len(corpus)}개 청크 "
          f"(핸드북 {len(load_handbook_chunks())} + 회사 문서 {len(load_company_chunks())})\n")
    for c in corpus:
        print(f"  {c['id']:<24} {len(c['text']):4d}자  {c['text'][:44].splitlines()[0]}...")
