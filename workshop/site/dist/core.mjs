// 허용된 식과 비교하며 Python/JavaScript로 실행하지 않는다.
export function gradeCode(value, answers) {
  const input = String(value).trim().replace(/\s+/g, '');
  // 식 전체를 둘러싼 괄호만 제거해 분모의 구조를 보존한다.
  const unwrap = s => {
    while (s.startsWith('(') && s.endsWith(')')) {
      let depth = 0, outer = true;
      for (let i = 0; i < s.length - 1; i++) {
        if (s[i] === '(') depth++;
        if (s[i] === ')') depth--;
        if (depth === 0) { outer = false; break; }
      }
      if (!outer) break;
      s = s.slice(1, -1);
    }
    return s;
  };
  return answers.some(answer => unwrap(input) === unwrap(answer.replace(/\s+/g, '')));
}
export function rrf(rankings, k = 60) {
  const scores = new Map();
  for (const ranking of rankings) ranking.forEach((id, index) => scores.set(id, (scores.get(id) || 0) + 1 / (k + index + 1)));
  return [...scores].sort((a, b) => b[1] - a[1]);
}
export function metrics(ids, answers, k) {
  const relevant = new Set(answers);
  const found = new Set(ids.slice(0, k).filter(id => relevant.has(id)));
  const first = ids.findIndex(id => relevant.has(id));
  return {hit: Number(found.size > 0), recall: relevant.size ? found.size / relevant.size : null, rr: first < 0 ? 0 : 1 / (first + 1)};
}
export function fixedChunks(text, size, overlap) {
  if (!Number.isInteger(size) || size <= 0 || overlap < 0 || overlap >= size) throw Error('청크 크기보다 작은 오버랩이 필요합니다.');
  const result = [];
  for (let start = 0; start < text.length; start += size - overlap) {
    result.push(text.slice(start, start + size));
    if (start + size >= text.length) break;
  }
  return result;
}
export const edges = [
  ['김철수', '담당', 'PD-2000', '03-sales-people'],
  ['김철수', '소속', '영업1팀', '03-sales-people'],
  ['PD-2000', '생산', '제조1공장', '04-product-pd2000'],
  ['제조1공장', '위치', '충남 천안', '07-factories'],
  ['영업1팀', '상위조직', '영업본부', '02-organization'],
  ['박영희', '총괄', '영업본부', '03-sales-people'],
  ['대한중공업', '고객제품', 'PD-2000', '08-customers']
];
export function neighbors(start, depth) {
  let frontier = new Set([start]);
  const visited = new Set([start]), found = new Set();
  for (let i = 0; i < depth; i++) {
    const next = new Set();
    for (const node of frontier) edges.forEach((edge, index) => {
      if (edge[0] !== node && edge[2] !== node) return;
      found.add(index);
      for (const n of [edge[0], edge[2]]) if (!visited.has(n)) { visited.add(n); next.add(n); }
    });
    frontier = next;
  }
  return [...found].map(i => edges[i]);
}
export function cosine(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b));
}
export function chapterPassed(chapter, record, requiredRuns) {
  return Boolean(record && requiredRuns.every(key => record.runs?.[key]) && record.sourceRead && chapter.questions.every(q => record.responses?.[q.id]?.passed));
}
export function canOpen(index, chapters, records, labs) {
  return chapters.slice(0, index).every(c => chapterPassed(c, records[c.id], labs[c.id].required));
}
