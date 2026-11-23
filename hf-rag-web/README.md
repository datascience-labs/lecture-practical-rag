# 주택금융 RAG 실습 — 강의·퀴즈 웹

한국주택금융공사 8시간 RAG 실습 과정의 강의 자료 겸 실습 웹사이트입니다 (정적 파일, 서버 코드 없음).
문서·수치는 모두 강의용 가상 데이터이며, 화면의 성능 수치는 강사 PC(CPU, qwen3:1.7b · bge-m3)에서 평가셋 40문항으로 실측한 결과입니다.

## 실행

`course.json`을 읽어야 하므로 `index.html`을 더블클릭하지 말고 로컬 서버로 엽니다 (Python 필요).

- Windows: `실행.bat` 더블클릭
- 그 외: 이 폴더에서 `python -m http.server 8765 --bind 127.0.0.1` 후 http://127.0.0.1:8765/quiz/index.html

## 구성

| 파일 | 내용 |
|---|---|
| `quiz/index.html`, `app.js`, `core.mjs`, `labs.mjs`, `style.css` | 화면 · 슬라이드 넘김 · 채점 · 실습 |
| `quiz/course.json` | 8단계 강의 본문 · 퀴즈 · 실측 결과 (빌드 산출물) |
| `quiz/img/*.svg` | 강의 그림 15장 |

진도는 각 브라우저의 localStorage에만 저장됩니다. 01단계의 Colab 버튼은 공유된 실습 노트북을 엽니다.
