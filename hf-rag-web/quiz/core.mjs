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
// edges: [출발, 관계, 도착, 출처] 목록. 방향과 무관하게 depth단계까지 연결된 관계를 모은다.
export function neighbors(start, depth, edges) {
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
export function chapterPassed(chapter, record, requiredRuns) {
  return Boolean(record && requiredRuns.every(key => record.runs?.[key]) && record.sourceRead && chapter.questions.every(q => record.responses?.[q.id]?.passed));
}
