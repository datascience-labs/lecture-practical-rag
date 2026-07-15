# rag_pipeline.py — 5회차: 기본 RAG 파이프라인 (SimpleRAG)
#
# 인덱싱(index) → 검색(retrieve) → 프롬프트 구성(_build_prompt) → 생성(ask, 스트리밍).
# 7회차 RerankedRAG, 8회차 AdaptiveRAG가 이 클래스를 확장한다.
#
# 실행: python main.py  (대화형 CLI — ANTHROPIC_API_KEY 필요)
#   retrieve()만 쓰는 실습(6·7·9회차 검색 평가)은 API 키 없이도 동작한다.

import anthropic
import chromadb
from chromadb.utils import embedding_functions

from corpus import CHROMA_PATH

SYSTEM_PROMPT = """너는 사내 문서 기반 질의응답 도우미다. 규칙:
1. 반드시 <documents> 안의 내용에 근거해서만 답변한다.
2. 답변 끝에 근거 문서의 source를 [출처: ...] 형식으로 표기한다.
3. 문서에 답이 없으면 "제공된 문서에서 답을 찾을 수 없습니다"라고 답한다.
4. 추측하거나 문서 밖 지식으로 답하지 않는다."""


class SimpleRAG:
    def __init__(self, collection_name: str = "rag_docs"):
        self._llm = None                     # 생성(ask) 시점에만 초기화 — 키 없이 검색 가능
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3")
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = db.get_or_create_collection(
            collection_name, embedding_function=ef,
            metadata={"hnsw:space": "cosine"})

    @property
    def llm(self) -> anthropic.Anthropic:
        if self._llm is None:
            self._llm = anthropic.Anthropic()   # ANTHROPIC_API_KEY 환경변수 필요
        return self._llm

    # --- 인덱싱 (최초 1회) ---
    def index(self, docs: list[dict]):
        """docs: [{"id": ..., "text": ..., "source": ...}, ...]"""
        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"source": d["source"]} for d in docs],
        )
        print(f"{len(docs)}개 청크 인덱싱 완료 (총 {self.collection.count()}개)")

    # --- 검색 ---
    def retrieve(self, question: str, k: int = 4,
                 max_distance: float = 0.6) -> list[dict]:
        r = self.collection.query(query_texts=[question], n_results=k)
        return [
            {"id": cid, "text": doc, "source": meta["source"], "distance": dist}
            for cid, doc, meta, dist in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
            if dist <= max_distance          # 관련 없는 청크는 버림
        ]

    # --- 프롬프트 구성과 생성 ---
    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        docs_xml = "\n".join(
            f'<doc id="{i+1}" source="{c["source"]}">\n{c["text"]}\n</doc>'
            for i, c in enumerate(chunks))
        return f"<documents>\n{docs_xml}\n</documents>\n\n질문: {question}"

    def ask(self, question: str) -> str:
        chunks = self.retrieve(question)
        if not chunks:
            msg = "제공된 문서에서 답을 찾을 수 없습니다."
            print(msg)
            return msg

        with self.llm.messages.stream(
            model="claude-opus-4-8",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": self._build_prompt(question, chunks)}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)   # 스트리밍 출력
            return stream.get_final_message().content[0].text
