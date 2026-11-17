"""의존성 없는 정적 실습 데이터 생성. 원문과 문항을 한 버전으로 묶는다."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
bank = json.loads((HERE.parent / 'questions.json').read_text())
handbook = (ROOT / 'data/sample_handbook.md').read_text()
docs = {}
for m in re.finditer(r'^### (\d+\.\d+) ([^\n]+)\n(.*?)(?=^## |^### |\Z)', handbook, re.M | re.S):
    number, title, body = m.groups()
    docs['handbook-' + number] = {'title': title, 'text': body.strip(), 'source': 'data/sample_handbook.md', 'category': 'HR'}
for f in sorted((ROOT / 'data/company_docs').glob('*.txt')):
    docs[f.stem] = {'title': f.stem, 'text': f.read_text().strip(), 'source': 'data/company_docs/' + f.name, 'category': 'PRODUCT' if 'product' in f.stem else 'COMPANY'}
payload = {'bank': bank, 'docs': docs}
payload['version'] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
(HERE / 'dist/course.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
print(f"생성: 12회차 / {len(docs)}개 원문 / 버전 {payload['version']}")
