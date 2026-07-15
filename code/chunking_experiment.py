# chunking_experiment.py — 4회차 실습: 청킹 전략 비교 실험
#
# 같은 문서를 4가지 전략으로 청킹해 각각 ChromaDB 컬렉션에 넣고,
# 같은 질문으로 검색했을 때 어떤 전략이 정답 청크를 1위로 올리는지 비교한다.
#
# 실행: python chunking_experiment.py
#   (최초 실행 시 BAAI/bge-m3 임베딩 모델을 다운로드한다 — 수 GB, 시간 소요)

import chromadb
from chromadb.utils import embedding_functions

from chunking import fixed_chunk, markdown_chunk, recursive_chunk
from corpus import CHROMA_PATH, DATA_DIR

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
client = chromadb.PersistentClient(path=CHROMA_PATH)

document = (DATA_DIR / "sample_handbook.md").read_text(encoding="utf-8")

strategies = {
    "fixed_300":     fixed_chunk(document, size=300, overlap=30),
    "fixed_800":     fixed_chunk(document, size=800, overlap=80),
    "recursive_500": recursive_chunk(document, size=500),
    "markdown":      [c["text"] for c in markdown_chunk(document)],
}

for name, chunks in strategies.items():
    try:                                       # 재실행 시 이전 실험 컬렉션 삭제
        client.delete_collection(f"exp_{name}")
    except Exception:
        pass
    col = client.create_collection(f"exp_{name}", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"{name}-{i}" for i in range(len(chunks))], documents=chunks)
    print(f"{name}: {len(chunks)}개 청크")

test_questions = [
    "연차는 며칠인가요?",
    "재택근무 승인 절차를 알려주세요",
    "법인카드 증빙은 언제까지 내나요?",
]

for q in test_questions:
    print(f"\n=== 질문: {q}")
    for name in strategies:
        col = client.get_collection(f"exp_{name}", embedding_function=ef)
        r = col.query(query_texts=[q], n_results=1)
        doc = r["documents"][0][0][:80].replace("\n", " ")
        dist = r["distances"][0][0]
        print(f"  [{name:>14}] ({dist:.3f}) {doc}...")

print("""
관찰 포인트:
- 어떤 전략이 질문에 정확히 답할 수 있는 청크를 1위로 올리는가?
- fixed_300은 문맥이 잘렸는가? fixed_800은 주제가 섞였는가?
- distance가 작을수록 가깝다 (cosine distance = 1 - 유사도)
""")
