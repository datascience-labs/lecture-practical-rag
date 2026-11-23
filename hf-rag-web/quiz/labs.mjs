// 실습 화면 — 모든 수치·검색 결과·답변은 course.json에 담긴 강사 PC 실측 결과다 (quiz/build.py가 생성).
import {neighbors} from './core.mjs';
export const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const box = (title, text) => `<div class="result-box"><h3>${h(title)}</h3><p>${h(text)}</p></div>`;
const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(t=>`<th>${h(t)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(t=>`<td>${h(t)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const metric = pairs => `<div class="metric-grid">${pairs.map(([name,v])=>`<div class="metric"><small>${h(name)}</small><strong>${h(v)}</strong></div>`).join('')}</div>`;
const MEASURED = '실측 결과 · 강사 PC (CPU · qwen3:1.7b · bge-m3)';

export const labs = {}, codeCards = {};
export function initLabs(course) {
  for (const c of course.bank.chapters) {
    const spec = Object.fromEntries(c.lab.options.map(o => [o.key, o]));
    labs[c.id] = {title: c.lab.title, concept: c.lab.concept, options: c.lab.options.map(o => [o.key, o.label]), spec, required: Object.keys(spec)};
    codeCards[c.id] = c.questions.find(q => q.type === 'code_blank').cards;
  }
}

export function runLab(id, key, course) {
  const o = labs[id]?.spec[key];
  if (!o) throw Error('지원하지 않는 실습 설정입니다.');
  if (o.qid) {
    const q = course.questions[o.qid], r = course.runs[`${o.qid}:${o.strategy}`], rel = new Set(q.relevant_documents);
    let html = box(`질문 ${o.qid} · 유형 ${q.category}`, q.question);
    html += r.retrieved.length
      ? table(['순위', '검색된 문서', '정답 문서?'], r.retrieved.map((d, i) => [i + 1, course.docs[d].title, rel.has(d) ? '✓ 정답' : '—']))
        + metric([['Recall@5', r['recall@5'].toFixed(2)], ['MRR', r.mrr.toFixed(2)], ['정답 문서 수', rel.size]])
      : box('검색', '문서를 찾지 않고 LLM만으로 답했습니다.');
    if (r.route) html += box('Router 판정', `${r.route} → 이 유형에 맞는 전략으로 실행`);
    html += r.answer
      ? box(r.refusal ? '실측 답변 (답변 거절)' : '실측 답변', r.answer.replace(/\*\*/g, ''))
      : box('답변', '이 전략은 검색 단계만 측정했습니다 (답변 생성 없음).');
    html += box('평가셋 정답', `${q.answer} · 정답 문서: ${q.relevant_documents.join(', ')}`);
    return {html, sources: r.retrieved.length ? r.retrieved : q.relevant_documents, label: MEASURED, config: key};
  }
  if (o.table) {
    const t = course.tables[o.table];
    let html = table(t.headers, t.rows) + (t.note ? box('참고', t.note) : '');
    if (o.table === 'overall') html += box('지연시간 (초, 캐시 끔 · CPU · 유형별 2문항 평균)', 'search_s = 질문 변환·검색·리랭크, answer_s = 답변 생성') + table(course.tables.latency.headers, course.tables.latency.rows);
    return {html, sources: t.sources || [], label: MEASURED + ' · 평가셋 40문항', config: key};
  }
  const {start, hops} = o.graph, E = course.graph.entities;
  const found = neighbors(start, hops, course.graph.relations.map(([s, rel, t, , doc]) => [s, rel, t, doc]));
  const value = (s, t) => course.graph.relations.find(r => r[0] === s && r[2] === t)?.[3] || '';
  const html = box('출발 개체', `${E[start].name} (${E[start].type}) · ${hops}-hop`)
    + table(['개체', '관계', '개체', '값', '출처 문서'], found.map(([s, rel, t, doc]) => [E[s]?.name || s, rel, E[t]?.name || t, value(s, t), doc]))
    + box('관찰', '주택보증부가 소관하는 규정(전세자금보증 규정 ─MANAGED_BY→ 주택보증부)을 찾고, 그 규정에 연결된 상품(─RELATED_TO→ 전세자금보증 규정)까지 두 번 건너야 "그 규정이 적용되는 상품"이 확인됩니다. 모든 관계에 출처 문서가 있어 원문으로 검증할 수 있습니다.');
  return {html, sources: [...new Set(found.map(e => e[3]))], label: '브라우저에서 계산 · 강사 검수 그래프 (개체 40 · 관계 89)', config: key};
}
