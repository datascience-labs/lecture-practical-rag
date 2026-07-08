# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A 17-session Korean-language lecture course on practical RAG (Retrieval-Augmented Generation), progressing from RAG basics through GraphRAG, OntoRAG, and NodeRAG (sessions 10–14) to open-source-VLM multimodal RAG — CAD drawing retrieval and field-photo analysis (sessions 15–16). All lecture material lives in `slides/` as Marp-formatted Markdown files (`01-*.md` through `17-*.md`). There is no application source code — practice code is embedded in the slides as fenced Python blocks that learners copy out and run.

## Commands

```bash
# Preview slides live in a browser (Marp CLI, needs Node.js)
npx @marp-team/marp-cli -s slides/

# Export one deck to HTML / all decks to PDF
npx @marp-team/marp-cli slides/01-rag-overview.md -o out/01.html
npx @marp-team/marp-cli slides/*.md --pdf -o out/

# Set up the Python environment the embedded practice code assumes
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Conventions that must hold across sessions

- **Slide format**: every file starts with Marp frontmatter (`marp: true`, `theme: default`, `paginate: true`, a `header` naming the session number, `footer: 'Practical RAG Course'`). Slides are separated by `---`. Title and closing slides use `<!-- _class: lead -->`.
- **Language**: all prose is Korean; code identifiers and library names stay in English.
- **Fixed tech stack in code examples** (changing any of these requires updating every session that uses it): Claude model `claude-opus-4-8` via the `anthropic` Python SDK, embeddings `BAAI/bge-m3` via sentence-transformers, ChromaDB for vectors, `BAAI/bge-reranker-v2-m3` for reranking, kiwipiepy for Korean tokenization, NetworkX for graphs. Multimodal sessions 15–16 deliberately use **local open-source models only** (security-sensitive drawings/photos must not leave the machine): Qwen2.5-VL via Ollama (`qwen2.5vl:7b`) or transformers, CLIP via sentence-transformers, ColQwen2 (`vidore/colqwen2-v1.0`) via colpali-engine — do not swap these for cloud APIs.
- **Code continuity**: practice code builds cumulatively — session 5 defines `SimpleRAG` (with `retrieve`/`ask`/`_build_prompt`), session 6 adds `HybridRetriever` and `rrf_fuse`, session 7 subclasses `RerankedRAG`, session 10's `extract_triples`/`find_neighbors` feed sessions 11–12, session 12's mini-GraphRAG is extended in 14, and session 15's captioning pattern reuses `SimpleRAG` while session 16 reuses session 3's metadata filtering and session 9's eval harness. When editing code in one session, check later sessions that reference or subclass it.
- **Session structure**: each deck has 목표 (goals) near the top, theory slides, a 실습 (hands-on) section, and ends with 과제 (homework) plus a 다음 시간 (next session) teaser that names the following session — renumbering or retitling a session breaks the teaser in the previous deck and the roadmap table in `slides/01-rag-overview.md` and `README.md`.

## Pedagogical through-lines (don't break these when editing content)

- The evaluation-first message: session 5 introduces the "search problem vs generation problem" debugging tree, session 9 formalizes it into metrics, and sessions 11–15 repeatedly justify technique adoption by "measured failure on your own eval set". New content should follow the same evidence-driven framing.
- Sessions 10–14 are motivated by specific failure types established earlier: multi-hop failures (session 9 homework) motivate knowledge graphs, global questions motivate GraphRAG, relation-vocabulary sprawl (session 12) motivates OntoRAG, and node-homogeneity motivates NodeRAG.
- Sessions 15–16 are motivated by text-extraction failure on non-text corpora (CAD drawings, site photos); session 16 treats VLM hallucination (misread dimensions) as a first-class risk with a human-verification workflow — keep that framing in any edits.
