# hw_common.py — 과제 해답(hw01~hw16) 공용 유틸
#
# 각 hwNN_*.py는 과제 항목을 함수로 정의하고 run_tasks()에 등록한다.
# 실습 코드(chunking.py 등)와 분리되어 있어 hw 파일 실행 = 과제 부분만 실행이며,
# --task 옵션으로 개별 과제 항목만 골라 실행할 수 있다:
#   python hw05_basic_rag_pipeline.py            # 해당 회차 과제 전부
#   python hw05_basic_rag_pipeline.py --task 2   # 과제 2만
#   python hw05_basic_rag_pipeline.py --list     # 과제 목록 확인
#
# LLM 호출 결과 캐시는 저장소 루트 hw_cache/ 에 저장된다 (12회차 "추출 캐시 필수!" 원칙).

import argparse
import json
import os
import re
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "hw_cache"


# --- 출력 도우미 ---

def hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def skip(msg: str) -> None:
    print(f"  [건너뜀] {msg}")


def _display_width(s: str) -> int:
    """한글은 터미널에서 2칸을 차지한다 — 표 정렬용 근사 폭."""
    return sum(2 if ord(ch) >= 0x1100 else 1 for ch in str(s))


def print_table(headers: list, rows: list[list]) -> None:
    """한글 폭을 고려한 간단한 표 출력."""
    table = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(_display_width(row[i]) for row in table)
              for i in range(len(headers))]
    for ri, row in enumerate(table):
        cells = [c + " " * (widths[i] - _display_width(c))
                 for i, c in enumerate(row)]
        print("  " + " | ".join(cells))
        if ri == 0:
            print("  " + "-+-".join("-" * w for w in widths))


# --- LLM 도우미 ---

def get_client():
    """ANTHROPIC_API_KEY가 있으면 anthropic 클라이언트, 없으면 None.

    LLM이 필요한 과제는 None이면 skip() 후 건너뛴다 — 검색 계열 과제는 키 없이 동작.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


def llm_text(client, system: str, user: str, max_tokens: int = 2048) -> str:
    response = client.messages.create(
        model="claude-opus-4-8", max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}])
    return response.content[0].text


def llm_json(client, system: str, user: str,
             max_tokens: int = 2048, expect: str = "object"):
    """LLM 호출 후 JSON만 추출 — 코드펜스가 섞여 나와도 파싱 (generate_evalset.py 방식)."""
    text = llm_text(client, system, user, max_tokens)
    pattern = r"\[.*\]" if expect == "array" else r"\{.*\}"
    match = re.search(pattern, text, flags=re.DOTALL)
    return json.loads(match.group() if match else text)


# --- 캐시 (LLM 호출 절약 — 재실행 시 hw_cache/*.json 재사용) ---

def load_cache(name: str):
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(name: str, obj) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --- 과제 실행기 ---

def run_tasks(title: str, tasks: dict, extra_args: list | None = None) -> None:
    """tasks: {과제번호: (설명, fn(args))}. --task로 일부만 실행할 수 있다."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--task", type=int, nargs="+", metavar="N",
                        help="실행할 과제 번호 (기본: 전부)")
    parser.add_argument("--list", action="store_true", help="과제 목록만 출력")
    for name, kwargs in (extra_args or []):
        parser.add_argument(name, **kwargs)
    args = parser.parse_args()

    if args.list:
        print(title)
        for n, (desc, _) in tasks.items():
            print(f"  과제 {n}: {desc}")
        return

    for n in args.task or list(tasks):
        if n not in tasks:
            print(f"과제 {n}은(는) 없다. --list로 목록을 확인하라.")
            continue
        desc, fn = tasks[n]
        hr(f"[{title}] 과제 {n}: {desc}")
        try:
            fn(args)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            print(f"  [오류] 과제 {n} 실행 실패 — 위 traceback 참고. 다음 과제로 계속한다.")
