# main.py — 5회차 실습: CLI 질의응답 시스템
#
# 실행: export ANTHROPIC_API_KEY="sk-ant-..." && python main.py
#
# 디버깅 팁 (5회차 디버깅 트리): 답이 이상하면 먼저 검색 문제인지 확인하라.
#   질문 앞에 /r 을 붙이면 답변 생성 없이 retrieve() 결과만 보여준다.
#   예) /r 연차는 며칠인가요?

from corpus import ensure_indexed
from rag_pipeline import SimpleRAG

rag = SimpleRAG()
ensure_indexed(rag)          # --- 인덱싱 (최초 1회) ---

print("\n사내 문서 QA — 한빛테크 핸드북 + 회사 문서")
print("질문 앞에 /r 을 붙이면 검색 결과만 확인 (예: /r 연차는 며칠인가요?)")

# --- 대화 루프 ---
while True:
    try:
        q = input("\n질문 (종료: quit) > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q == "quit":
        break
    if not q:
        continue

    if q.startswith("/r "):                  # 검색 문제 vs 생성 문제 분리 진단
        for c in rag.retrieve(q[3:].strip()):
            first_line = c["text"][:70].replace("\n", " ")
            print(f"  [{c['distance']:.3f}] ({c['id']}) {first_line}...")
        continue

    print()
    rag.ask(q)
    print()
