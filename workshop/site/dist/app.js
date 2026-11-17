import {gradeCode, chapterPassed, canOpen} from './core.mjs';
import {h, labs, runLab, codeCards} from './labs.mjs';

const app = document.querySelector('#app');
const dialog = document.querySelector('#source-dialog');
let course, state, storageKey, storageOK = true, view = 'lesson';
let feedback = {}, hints = {}, toastTimer;
const record = id => state.chapters[id] ||= {runs:{}, responses:{}, drafts:{}, sourceRead:false};
const passed = c => chapterPassed(c, state.chapters[c.id], labs[c.id].required);
const allPassed = () => course.bank.chapters.every(passed);
function fresh() {
  return {version:course.version, learnerId:crypto.randomUUID(), learnerName:'', active:'01', chapters:{}, capstone:{}, savedAt:null};
}
function toast(message) {
  const t=document.querySelector('#toast');t.textContent=message;t.classList.add('visible');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove('visible'),4000);
}
function save() {
  state.savedAt=new Date().toISOString();
  try {localStorage.setItem(storageKey,JSON.stringify(state));storageOK=true;}
  catch {storageOK=false;toast('브라우저 저장을 사용할 수 없습니다. 결과 파일을 내려받아 보관하세요.');}
}
function validState(input) {
  if(input.version!==course.version) throw Error('자료 버전이 다릅니다. 같은 버전의 결과 파일을 선택하세요.');
  if(typeof input.learnerId!=='string'||!input.chapters||typeof input.chapters!=='object') throw Error('실습 결과 파일 형식이 아닙니다.');
  const clean=fresh();clean.learnerId=input.learnerId.slice(0,100);clean.learnerName=String(input.learnerName||'').slice(0,80);
  for(const c of course.bank.chapters){
    const old=input.chapters[c.id];if(!old)continue;
    const runs={};for(const key of labs[c.id].required)if(old.runs?.[key])runs[key]={config:key,at:String(old.runs[key].at||'')};
    const responses={};
    for(const q of c.questions){
      const r=old.responses?.[q.id];if(!r)continue;
      const value=String(r.value??'').slice(0,500);
      responses[q.id]={value,passed:q.type==='choice'?Number(value)===q.answer_index:gradeCode(value,q.answers),attempts:Math.max(1,Math.min(999,Number(r.attempts)||1)),assisted:Boolean(r.assisted),at:String(r.at||'')};
    }
    const drafts={};
    for(const q of c.questions){
      if(typeof old.drafts?.[q.id]==='string')drafts[q.id]=old.drafts[q.id].slice(0,500);
      if(['text','card'].includes(old.drafts?.[q.id+'-mode']))drafts[q.id+'-mode']=old.drafts[q.id+'-mode'];
      if(old.drafts?.[q.id+'-assisted'])drafts[q.id+'-assisted']=true;
    }
    clean.chapters[c.id]={runs,responses,drafts,sourceRead:Boolean(old.sourceRead),current:labs[c.id].required.includes(old.current)?old.current:undefined};
  }
  // 선행 회차가 미완료이면 이후 회차의 기록은 재개용으로 열지 않는다.
  let gap=false;for(const c of course.bank.chapters){if(gap)delete clean.chapters[c.id];else if(!chapterPassed(c,clean.chapters[c.id],labs[c.id].required))gap=true;}
  clean.active=course.bank.chapters.find(c=>!chapterPassed(c,clean.chapters[c.id],labs[c.id].required))?.id||'12';
  const activeIndex=course.bank.chapters.findIndex(c=>c.id===input.active);
  if(activeIndex>=0&&canOpen(activeIndex,course.bank.chapters,clean.chapters,labs))clean.active=input.active;
  if(typeof input.savedAt==='string'&&!Number.isNaN(Date.parse(input.savedAt)))clean.savedAt=input.savedAt;
  if(allImportedPassed(clean))for(const q of course.bank.capstone.questions){const old=input.capstone?.[q.id];if(old){clean.capstone[q.id]={};for(const field of ['chosen_route','source_refs','answer','failure_or_limit','reason_for_selection'])clean.capstone[q.id][field]=String(old[field]||'').slice(0,4000);clean.capstone[q.id].submitted=Boolean(old.submitted)&&capstoneFilled(clean.capstone[q.id]);}}
  return clean;
}
function allImportedPassed(s){return course.bank.chapters.every(c=>chapterPassed(c,s.chapters[c.id],labs[c.id].required));}
function capstoneFilled(r){return ['chosen_route','source_refs','answer','failure_or_limit','reason_for_selection'].every(k=>String(r?.[k]||'').trim().length>0);}

function sourceButtons(ids){return `<div class="source-links">${[...new Set(ids)].filter(id=>course.docs[id]).map(id=>`<button type="button" data-source="${h(id)}">↗ ${h(course.docs[id].title)} · 원문</button>`).join('')}</div>`;}
function render(){
  const completed=course.bank.chapters.filter(passed).length;
  app.innerHTML=`<header class="topbar"><div class="brand"><div class="brand-icon">R.</div><div><strong>Practical RAG</strong><span>직접 확인하며 배우는 RAG 실습</span></div></div><div class="actions"><span class="pill">8시간 · 12단계</span><button data-action="overview">수업 안내</button><button data-action="summary">내 학습 기록</button><button data-action="export">결과 내려받기</button></div></header>
  <div class="shell"><aside class="sidebar" aria-label="강의 진행"><div class="progress-head"><strong>LEARNING PATH</strong><span>${completed} / 12 완료</span></div><progress value="${completed}" max="12" aria-label="회차 완료 진도"></progress><nav aria-label="회차 선택">${course.bank.chapters.map((c,i)=>{const open=canOpen(i,course.bank.chapters,state.chapters,labs);return `<button class="nav-item ${view==='lesson'&&state.active===c.id?'active':''}" data-chapter="${c.id}" ${open?'':'disabled'} ${state.active===c.id&&view==='lesson'?'aria-current="step"':''}><span class="nav-no">${passed(c)?'✓':c.id}</span><span class="nav-title">${h(c.title)}</span><span class="nav-state">${passed(c)?'완료':open?'열림':'잠김'}</span></button>`;}).join('')}<button class="nav-item ${view==='capstone'?'active':''}" data-action="capstone" ${allPassed()?'':'disabled'}><span class="nav-no">↗</span><span class="nav-title">종합 실습 · 40분</span><span class="nav-state">${allPassed()?'열림':'잠김'}</span></button></nav><p class="side-note">차례대로 실습하고 확인 문제를 통과하면 다음 단계가 열립니다.<br><br>진도는 이 브라우저에 저장됩니다. 기기를 바꿀 때 결과 파일을 가져오세요.</p></aside><main id="main" class="main" tabindex="-1">${view==='summary'?summaryHTML():view==='overview'?overviewHTML():view==='capstone'?capstoneHTML():lessonHTML()}</main></div>`;
}
function lessonHTML(){
  const c=course.bank.chapters.find(c=>c.id===state.active),lab=labs[c.id],r=record(c.id);
  const runsDone=lab.required.every(k=>r.runs[k]), activityDone=runsDone&&r.sourceRead,done=passed(c);
  const output=r.current&&r.runs[r.current]?runLab(c.id,r.current,course.docs):null;
  return `<section class="hero"><div><div class="eyebrow">CHAPTER ${c.id} / 12</div><div class="hero-meta"><span class="pill">${c.minutes}분</span><span class="pill">객관식 1 · 코드 완성 1</span><span class="pill">${done?'✓ 완료':'단계별 가이드 실습'}</span></div><h1>${h(c.title)}</h1><p class="muted">${h(c.goal)}</p></div><div class="hero-graphic" aria-hidden="true"><div class="flow-node">질문</div><span class="flow-arrow">→</span><div class="flow-node">근거</div><span class="flow-arrow">→</span><div class="flow-node">답변</div></div></section>
  <div class="step-strip"><span class="${runsDone?'done':''}">01 설정 비교 ${runsDone?'✓':''}</span><span>→</span><span class="${r.sourceRead?'done':''}">02 원문 확인 ${r.sourceRead?'✓':''}</span><span>→</span><span class="${r.responses[c.questions[0].id]?.passed?'done':''}">03 결과 해석</span><span>→</span><span class="${done?'done':''}">04 핵심 코드</span></div>
  <section class="card"><div class="card-header"><h2><span class="number-tag">1</span>${h(lab.title)}</h2><span class="pill">직접 해보기</span></div><div class="concept">${h(lab.concept)}</div><ol class="instructions">${c.activity.steps.map(s=>`<li>${h(s)}</li>`).join('')}<li>아래의 모든 설정을 한 번씩 실행하고, 결과의 <strong>원문</strong> 버튼을 열어 근거를 확인하세요.</li></ol><div class="lab-controls"><div class="field"><label for="lab-config">비교할 설정</label><select id="lab-config">${lab.options.map(([key,label])=>`<option value="${key}" ${r.current===key?'selected':''}>${h(label)}</option>`).join('')}</select></div><button class="primary" data-action="run">실습 실행 →</button></div><div class="run-checks">${lab.options.map(([key,label])=>`<span class="${r.runs[key]?'done':''}">${r.runs[key]?'✓':'○'} ${h(label)}</span>`).join('')}<span class="${r.sourceRead?'done':''}">${r.sourceRead?'✓':'○'} 근거 원문 확인</span></div>${output?`<div class="lab-result" aria-label="실습 결과"><p class="muted">${h(output.label)} · 설정: ${h(lab.options.find(o=>o[0]===r.current)[1])}</p>${output.html}${sourceButtons(output.sources)}</div>`:'<p class="muted" style="margin-top:16px">설정을 선택하고 실행하면 결과가 여기에 표시됩니다.</p>'}</section>
  <section class="card"><div class="card-header"><h2><span class="number-tag">2</span>확인하고 다음 단계로</h2><span class="pill">${c.questions.filter(q=>r.responses[q.id]?.passed).length} / 2 통과</span></div>${!activityDone?'<p class="notice">모든 설정을 실행하고 근거 원문을 확인하면 문제를 풀 수 있습니다.</p>':''}${c.questions.map(q=>quizHTML(q,c,r,!activityDone)).join('')}</section>
  <div class="next-bar"><p>${done?'✓ 실습과 확인 문제를 모두 완료했습니다.':'설정 비교 · 원문 확인 · 두 문항 통과가 필요합니다.'}</p><button class="primary" data-action="next" ${done?'':'disabled'}>${c.id==='12'?'종합 실습으로':'다음 단계로'} →</button></div><p class="footer-note">학습용 발췌 코드입니다. 학생 입력을 실행하지 않고 허용된 답과 비교합니다. 원본: ${c.sources.map(h).join(' · ')}</p>`;
}
function quizHTML(q,c,r,disabled){
  const response=r.responses[q.id], draft=r.drafts[q.id]??response?.value??'',message=feedback[q.id];
  let body;
  if(q.type==='choice')body=`<fieldset ${disabled?'disabled':''}><legend>${h(q.prompt)}</legend>${q.options.map((o,i)=>`<label class="option"><input type="radio" name="${q.id}" value="${i}" ${String(draft)===String(i)?'checked':''}><span>${h(o)}</span></label>`).join('')}</fieldset>`;
  else {const mode=r.drafts[q.id+'-mode']||'text';body=`<p>${h(q.prompt)}</p><pre class="code">${h(q.template)}</pre><div class="code-modes"><span>답변 방식</span><label><input type="radio" name="${q.id}-mode" value="text" ${mode==='text'?'checked':''}> 직접 입력</label><label><input type="radio" name="${q.id}-mode" value="card" ${mode==='card'?'checked':''}> 코드 카드 선택</label></div>${mode==='text'?`<label class="field"><span class="muted">____에 들어갈 코드</span><input class="code-input" type="text" name="${q.id}" value="${h(draft)}" placeholder="한 단어 또는 한 식" autocomplete="off" spellcheck="false" ${disabled?'disabled':''}></label>`:`<fieldset ${disabled?'disabled':''}><legend>빈칸에 들어갈 코드를 선택하세요</legend>${codeCards[c.id].map(o=>`<label class="option"><input type="radio" name="${q.id}" value="${h(o)}" ${draft===o?'checked':''}><code>${h(o)}</code></label>`).join('')}</fieldset>`}`;}
  return `<div class="quiz" id="quiz-${q.id}"><p class="eyebrow">${q.type==='choice'?'관찰 결과 해석':'핵심 코드 완성'} ${response?.passed?'· ✓ 통과':''}</p>${body}<div class="quiz-actions"><button data-check="${q.id}" class="primary" ${disabled?'disabled':''}>답 확인</button><button data-hint="${q.id}" ${disabled?'disabled':''}>힌트 보기</button></div>${hints[q.id]?`<p class="hint">${h(q.hint)}</p>`:''}${message?`<p class="feedback ${message.ok?'success':''}" role="status">${h(message.text)}</p>`:response?.passed?`<p class="feedback success">✓ 통과 · ${h(q.explanation)}</p>`:''}</div>`;
}
function summaryHTML(){
  const count=course.bank.chapters.filter(passed).length,capCount=course.bank.capstone.questions.filter(q=>state.capstone[q.id]?.submitted).length;
  return `<div class="eyebrow">MY LEARNING RECORD</div><h1>내 학습 기록</h1><p class="muted">이 브라우저에 저장된 기록입니다. 강사에게 결과 파일을 제출하세요.</p><div class="summary-grid"><div class="summary-stat"><small>회차 완료</small><strong>${count} / 12</strong></div><div class="summary-stat"><small>종합 실습 제출</small><strong>${capCount} / 3</strong></div><div class="summary-stat"><small>강사 확인</small><strong style="font-size:1.2rem">${capCount===3?'검토 대기':'제출 전'}</strong></div></div><section class="card"><label class="field"><span>제출용 학습자 표시명 (선택)</span><input type="text" id="learner-name" maxlength="80" value="${h(state.learnerName)}" placeholder="예: 3조 / 학습자 07"></label><p class="muted">학습자 ID: ${h(state.learnerId)}<br>자료 버전: ${h(course.version)} · 저장: ${state.savedAt?h(new Date(state.savedAt).toLocaleString('ko-KR')):'아직 없음'}</p><div class="table-wrap"><table><thead><tr><th>회차</th><th>실습 실행</th><th>문항</th><th>힌트</th><th>상태</th></tr></thead><tbody>${course.bank.chapters.map(c=>{const r=state.chapters[c.id];return `<tr><td>${c.id} ${h(c.title)}</td><td>${labs[c.id].required.filter(k=>r?.runs?.[k]).length}/${labs[c.id].required.length}</td><td>${c.questions.filter(q=>r?.responses?.[q.id]?.passed).length}/2</td><td>${c.questions.some(q=>r?.responses?.[q.id]?.assisted)?'사용':'—'}</td><td>${passed(c)?'완료':'미완료'}</td></tr>`;}).join('')}</tbody></table></div><div class="quiz-actions"><button class="primary" data-action="export">결과 JSON 내려받기</button><button data-action="import">다른 기기의 결과 가져오기</button></div><input type="file" id="import-file" accept=".json,application/json" hidden><p class="notice" style="margin-top:18px">${storageOK?'진도는 이 브라우저에 저장됩니다.':'브라우저 저장을 사용할 수 없습니다.'} 중앙 서버에 제출되지는 않습니다. 파일 가져오기는 현재 기록을 대체합니다. 강사는 종합 실습의 근거와 선택 이유를 별도로 검토합니다.</p><button data-action="reset">이 브라우저의 실습 기록 초기화</button></section>`;
}
function overviewHTML(){return `<div class="eyebrow">8H GUIDED WORKSHOP</div><h1>기초 RAG에서<br>GraphRAG까지, 한 단계씩.</h1><p>가상 기업 퍼플델타의 문서를 읽고, 검색을 바꾸고, 근거를 검증합니다. 설치와 API 키 없이 브라우저에서 진행할 수 있습니다.</p><section class="card"><h2>이렇게 진행합니다</h2><ol class="instructions"><li>짧은 개념 설명을 읽고 설정을 선택해 실행합니다.</li><li>모든 필수 설정을 비교하고 근거 원문을 엽니다.</li><li>객관식 1문항과 코드 빈칸 1문항을 통과합니다.</li><li>다음 회차가 열립니다. 이전 회차는 언제든 다시 볼 수 있습니다.</li><li>12회차 완료 후 질문 3개의 종합 실습을 작성하고 결과 파일을 제출합니다.</li></ol><p class="notice">모델 답변·후보 순위·리포트는 강사 작성 예제입니다. 수치 계산·필터·청킹·그래프 탐색은 브라우저에서 실행합니다. 실제 LLM·임베딩·ChromaDB 서버를 호출하지 않습니다.</p></section><section class="card"><h2>8시간 구성</h2><p>회차별 실습 360분 + 종합 실습 40분 + 안내·정리 20분 + 휴식 60분 = 480분. 점심은 별도입니다.</p><div class="table-wrap"><table><thead><tr><th>회차</th><th>주제</th><th>시간</th></tr></thead><tbody>${course.bank.chapters.map(c=>`<tr><td>${c.id}</td><td>${h(c.title)}</td><td>${c.minutes}분</td></tr>`).join('')}</tbody></table></div><p class="muted">권장 휴식: 03 이후 10분 · 05 이후 10분 · 07 이후 15분 · 09 이후 10분 · 11 이후 15분.</p><button class="primary" data-action="resume">${course.bank.chapters.some(passed)?'실습 이어하기':'첫 실습 시작하기'} →</button></section><p class="footer-note">학습용 웹사이트로 정답과 진도는 브라우저에서 처리됩니다. 중앙 진도 집계·시험 보안·다중 사용자 로그인은 포함하지 않습니다.</p>`;}
function capstoneHTML(){
  if(!allPassed())return '<h1>종합 실습은 12회차 완료 후 열립니다.</h1>';
  return `<div class="eyebrow">FINAL PRACTICE · 40 MIN</div><h1>질문에서 근거까지,<br>직접 경로를 선택하세요.</h1><p class="muted">질문 3개에 답하고 결과 파일을 내려받아 강사에게 제출하세요. 작성 완료는 자동 확인하며 답변의 타당성은 강사가 검토합니다.</p><p class="notice">다른 경로를 선택해도 충분한 근거가 있으면 인정합니다. 최소 한 질문에서 검색 단계와 생성 단계 중 우선 점검할 곳을 설명하세요.</p>${course.bank.capstone.questions.map((q,i)=>{const r=state.capstone[q.id]||{};const ids=q.sources.map(s=>s.includes('sample_handbook')?'handbook-3.1':s.split('/').pop().replace('.txt',''));return `<section class="card capstone-form" data-final="${q.id}"><p class="eyebrow">질문 ${i+1} ${r.submitted?'· 작성 완료 / 강사 검토 대기':''}</p><h2>${h(q.question)}</h2>${sourceButtons(ids)}${[['chosen_route','선택한 검색 경로','기본 RAG / 하이브리드 / Local / Global 등'],['source_refs','확인한 원문 출처','문서 ID와 답변을 뒷받침하는 문장'],['answer','근거에 따른 답변','원문에 없는 사실을 추가하지 마세요'],['failure_or_limit','실패 또는 한계','근거 누락·요약 손실·비용 등. 한 질문 이상은 검색/생성 중 우선 점검할 단계를 설명'],['reason_for_selection','비교 후 이 경로를 선택한 이유','다른 경로와 비교한 이유']].map(([field,label,placeholder])=>`<label class="field"><span>${label}</span><textarea data-final-id="${q.id}" data-field="${field}" maxlength="4000" placeholder="${placeholder}">${h(r[field]||'')}</textarea></label>`).join('')}<button class="primary" data-submit-final="${q.id}">이 질문 작성 완료</button></section>`;}).join('')}<div class="next-bar"><p>세 질문 모두 작성한 뒤 결과 JSON을 강사에게 제출하세요.</p><button class="primary" data-action="export">결과 내려받기 ↓</button></div>`;
}
function setView(next){view=next;feedback={};render();document.querySelector('#main').focus();window.scrollTo({top:0,behavior:'instant'});}
function exportReport(){
  save();
  const evidence=course.bank.chapters.flatMap(c=>Object.keys(state.chapters[c.id]?.runs||{}).map(key=>({chapter_id:c.id,config:key,output_ref:`${course.version}/${c.id}/${key}`,source_refs:runLab(c.id,key,course.docs).sources})));
  const report={...state,reportType:'practical-rag-workshop',exportedAt:new Date().toISOString(),evidence,summary:{chaptersPassed:course.bank.chapters.filter(passed).map(c=>c.id),capstoneSubmitted:course.bank.capstone.questions.filter(q=>state.capstone[q.id]?.submitted).map(q=>q.id),reviewStatus:'강사 검토 전',storage:'브라우저 로컬'}};
  const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`rag-workshop-${state.learnerId.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('결과 파일을 내려받았습니다. 강사에게 제출하세요.');
}
app.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button||button.disabled)return;
  if(button.dataset.chapter){state.active=button.dataset.chapter;save();setView('lesson');return;}
  if(button.dataset.source){const doc=course.docs[button.dataset.source];if(!doc)return;document.querySelector('#source-title').textContent=doc.title;document.querySelector('#source-path').textContent=doc.source;document.querySelector('#source-text').textContent=doc.text;
    if(view==='lesson'){record(state.active).sourceRead=true;save();render();}dialog.showModal();return;}
  if(button.dataset.hint){const id=button.dataset.hint;hints[id]=true;const r=record(state.active);r.drafts[id+'-assisted']=true;if(r.responses[id])r.responses[id].assisted=true;save();render();return;}
  if(button.dataset.check){
    const c=course.bank.chapters.find(c=>c.id===state.active),q=c.questions.find(q=>q.id===button.dataset.check),r=record(c.id);
    const element=app.querySelector(`input[name="${q.id}"]:checked`)||app.querySelector(`input[type="text"][name="${q.id}"]`);
    const value=element?.value?.trim()||'';if(!value){toast('답을 선택하거나 입력하세요.');return;}
    const ok=q.type==='choice'?Number(value)===q.answer_index:gradeCode(value,q.answers);
    const previous=r.responses[q.id];r.responses[q.id]={value,passed:ok||Boolean(previous?.passed),attempts:(previous?.attempts||0)+1,assisted:Boolean(previous?.assisted||r.drafts[q.id+'-assisted']),at:new Date().toISOString()};
    // 통과 이후 복습 오답은 통과 기록을 덮어쓰지 않는다.
    if(!ok&&previous?.passed)r.responses[q.id].value=previous.value;
    feedback[q.id]={ok,text:ok?'✓ 통과 · '+q.explanation:'아직 맞지 않습니다. '+q.hint+' 수정한 뒤 다시 확인하세요.'};save();render();document.querySelector('#quiz-'+q.id)?.scrollIntoView({block:'nearest'});return;
  }
  if(button.dataset.submitFinal){const id=button.dataset.submitFinal,r=state.capstone[id];if(!capstoneFilled(r)){toast('다섯 항목을 모두 작성하세요.');return;}r.submitted=true;r.submittedAt=new Date().toISOString();save();render();toast('작성 완료로 저장했습니다. 강사 검토는 결과 제출 후 진행됩니다.');return;}
  const action=button.dataset.action;
  if(action==='run'){const r=record(state.active),key=document.querySelector('#lab-config').value;runLab(state.active,key,course.docs);r.runs[key]={config:key,at:new Date().toISOString()};r.current=key;save();render();document.querySelector('.lab-result').scrollIntoView({block:'nearest'});}
  if(action==='next'){const i=course.bank.chapters.findIndex(c=>c.id===state.active);if(!passed(course.bank.chapters[i]))return;if(i===11)setView('capstone');else{state.active=course.bank.chapters[i+1].id;save();setView('lesson');}}
  if(['summary','overview','capstone'].includes(action))setView(action);
  if(action==='resume')setView('lesson');
  if(action==='export')exportReport();
  if(action==='import')document.querySelector('#import-file').click();
  if(action==='reset'&&confirm('이 브라우저의 실습 기록을 초기화할까요? 필요한 기록은 먼저 내려받으세요.')){state=fresh();save();hints={};setView('overview');}
});
app.addEventListener('input',event=>{
  const el=event.target;
  if(el.id==='learner-name'){state.learnerName=el.value;save();}
  if(el.dataset.finalId){const r=state.capstone[el.dataset.finalId]||={};r[el.dataset.field]=el.value;r.submitted=false;const label=el.closest('.capstone-form')?.querySelector('.eyebrow');if(label)label.textContent='수정 중 · 다시 작성 완료를 확인하세요';save();}
  if(view==='lesson'&&el.name&&!el.name.endsWith('-mode')){record(state.active).drafts[el.name]=el.value;save();}
});
app.addEventListener('change',async event=>{
  const el=event.target;
  if(el.name?.endsWith('-mode')){record(state.active).drafts[el.name]=el.value;save();render();}
  if(el.id==='import-file'&&el.files[0]){
    try{if(el.files[0].size>2000000)throw Error('2MB 이하의 결과 파일을 선택하세요.');const imported=validState(JSON.parse(await el.files[0].text()));if(!confirm('가져온 파일로 현재 브라우저의 기록을 대체할까요?'))return;state=imported;save();setView('summary');toast('기록을 가져왔습니다.');}
    catch(error){toast('가져오기 실패: '+error.message);el.value='';}
  }
});
document.querySelector('#close-source').addEventListener('click',()=>dialog.close());
try {
  const response=await fetch('./course.json');if(!response.ok)throw Error('강의 자료 응답 오류');course=await response.json();storageKey='practical-rag-workshop:'+course.version;
  let saved;try{saved=localStorage.getItem(storageKey);}catch{storageOK=false;}
  try{state=saved?validState(JSON.parse(saved)):fresh();}catch{state=fresh();toast('저장 기록을 읽지 못했습니다. 결과 파일이 있다면 가져오세요.');}
  view=Object.keys(state.chapters).length?'lesson':'overview';render();
} catch(error) {app.innerHTML=`<main class="main"><h1>실습 자료를 열 수 없습니다</h1><p>${h(error.message)}</p><p>인터넷 연결 또는 로컬 서버 실행 상태를 확인하고 새로고침하세요.</p></main>`;}
