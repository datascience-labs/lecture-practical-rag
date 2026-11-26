import {gradeCode, chapterPassed} from './core.mjs';
import {h, labs, runLab, codeCards, initLabs} from './labs.mjs';

const app = document.querySelector('#app');
const dialog = document.querySelector('#source-dialog');
let course, state, storageKey, storageOK = true, view = 'lesson';
let feedback = {}, hints = {}, toastTimer;
const N = () => course.bank.chapters.length, last = () => course.bank.chapters.at(-1).id;
const record = id => state.chapters[id] ||= {runs:{}, responses:{}, drafts:{}, sourceRead:false};
const passed = c => chapterPassed(c, state.chapters[c.id], labs[c.id].required);
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
    const concepts={};
    for(const k of c.concepts||[])for(const q of k.quizzes){const v=old.concepts?.[q.id];if(v!=null&&Number.isInteger(Number(v.value)))concepts[q.id]={value:Number(v.value),passed:Number(v.value)===q.answer_index};}
    clean.chapters[c.id]={runs,responses,drafts,concepts,sourceRead:Boolean(old.sourceRead),current:labs[c.id].required.includes(old.current)?old.current:undefined};
  }
  // 선행 회차가 미완료이면 이후 회차의 기록은 재개용으로 열지 않는다.
  let gap=false;for(const c of course.bank.chapters){if(gap)delete clean.chapters[c.id];else if(!chapterPassed(c,clean.chapters[c.id],labs[c.id].required))gap=true;}
  clean.active=course.bank.chapters.find(c=>!chapterPassed(c,clean.chapters[c.id],labs[c.id].required))?.id||last();
  const activeIndex=course.bank.chapters.findIndex(c=>c.id===input.active);
  if(activeIndex>=0)clean.active=input.active;
  if(typeof input.savedAt==='string'&&!Number.isNaN(Date.parse(input.savedAt)))clean.savedAt=input.savedAt;
  if(allImportedPassed(clean))for(const q of course.bank.capstone.questions){const old=input.capstone?.[q.id];if(old){clean.capstone[q.id]={};for(const field of ['chosen_route','source_refs','answer','failure_or_limit','reason_for_selection'])clean.capstone[q.id][field]=String(old[field]||'').slice(0,4000);clean.capstone[q.id].submitted=Boolean(old.submitted)&&capstoneFilled(clean.capstone[q.id]);}}
  return clean;
}
function allImportedPassed(s){return course.bank.chapters.every(c=>chapterPassed(c,s.chapters[c.id],labs[c.id].required));}
function capstoneFilled(r){return ['chosen_route','source_refs','answer','failure_or_limit','reason_for_selection'].every(k=>String(r?.[k]||'').trim().length>0);}

function sourceButtons(ids){return `<div class="source-links">${[...new Set(ids)].filter(id=>course.docs[id]).map(id=>`<button type="button" data-source="${h(id)}">↗ ${h(course.docs[id].title)} · 원문</button>`).join('')}</div>`;}
let slidePos={},lessonSlides={count:0,firstOf:{}};
function goSlide(i){slidePos[state.active]=Math.max(0,Math.min(i,lessonSlides.count-1));render();document.querySelector('#main')?.scrollTo?.(0,0);window.scrollTo(0,0);}
function render(){
  const completed=course.bank.chapters.filter(passed).length;
  app.innerHTML=`<header class="topbar"><div class="brand"><div class="brand-icon">R.</div><div><strong>주택금융 RAG 실습</strong><span>실측 결과로 배우는 단계별 실습</span></div></div><div class="actions"><span class="pill">09–18시 · ${N()}단계</span><button data-action="overview">수업 안내</button><button data-action="summary">내 학습 기록</button><button data-action="export">결과 내려받기</button></div></header>
  <div class="shell"><aside class="sidebar" aria-label="강의 진행"><div class="progress-head"><strong>LEARNING PATH</strong><span>${completed} / ${N()} 완료</span></div><progress value="${completed}" max="${N()}" aria-label="회차 완료 진도"></progress><nav aria-label="회차 선택">${course.bank.chapters.map(c=>{return `<button class="nav-item ${view==='lesson'&&state.active===c.id?'active':''}" data-chapter="${c.id}" ${state.active===c.id&&view==='lesson'?'aria-current="step"':''}><span class="nav-no">${passed(c)?'✓':c.id}</span><span class="nav-title">${h(c.title)}</span><span class="nav-state">${passed(c)?'완료':''}</span></button>`;}).join('')}<button class="nav-item ${view==='capstone'?'active':''}" data-action="capstone"><span class="nav-no">↗</span><span class="nav-title">종합 실습 · ${course.bank.capstone.minutes}분</span><span class="nav-state"></span></button></nav><p class="side-note">어느 단계든 바로 열 수 있습니다. 확인 문제를 통과하면 ✓ 완료로 표시됩니다.<br><br>진도는 이 브라우저에 저장됩니다. 기기를 바꿀 때 결과 파일을 가져오세요.</p></aside><main id="main" class="main" tabindex="-1">${view==='summary'?summaryHTML():view==='overview'?overviewHTML():view==='capstone'?capstoneHTML():lessonHTML()}</main></div>`;
}
function lessonHTML(){
  const c=course.bank.chapters.find(c=>c.id===state.active),lab=labs[c.id],r=record(c.id);
  const runsDone=lab.required.every(k=>r.runs[k]), activityDone=runsDone&&r.sourceRead,done=passed(c);
  const output=r.current&&r.runs[r.current]?runLab(c.id,r.current,course):null;
  const cs=c.concepts||[],conceptsDone=cs.length&&cs.every(k=>k.quizzes.filter(q=>q.required).every(q=>r.concepts?.[q.id]?.passed)),n=cs.length;
  const slides=[];
  slides.push({label:'표지 · 목차',html:`<section class="hero"><div><div class="eyebrow">STEP ${c.id} / ${N()}</div><div class="hero-meta"><span class="pill">${h(c.time)} · ${c.minutes}분</span><span class="pill">강의 ${n} · 실습 · 확인 2</span><span class="pill">${done?'✓ 완료':'단계별 가이드 실습'}</span></div><h1>${h(c.title)}</h1><p class="muted">${h(c.goal)}</p>${notebookLinks(c)}</div><figure class="hero-figure"><img src="images/step${h(c.id)}_cover.png" alt="${h(course.figures?.[`images/step${c.id}_cover.png`]||c.title)}"></figure></section>
  <div class="step-strip"><span class="${conceptsDone?'done':''}">00 강의 · 퀴즈 ${conceptsDone?'✓':''}</span><span>→</span><span class="${runsDone?'done':''}">01 설정 비교 ${runsDone?'✓':''}</span><span>→</span><span class="${r.sourceRead?'done':''}">02 원문 확인 ${r.sourceRead?'✓':''}</span><span>→</span><span class="${r.responses[c.questions[0].id]?.passed?'done':''}">03 결과 해석</span><span>→</span><span class="${done?'done':''}">04 핵심 코드</span></div><section class="card"><div class="card-header"><h2>이 단계의 강의</h2><span class="pill">← → 키로 넘기기</span></div><ol class="toc">${cs.map((k,i)=>`<li><button type="button" data-goto="${i}">${h(k.title)}</button></li>`).join('')}<li><button type="button" data-goto="lab">${h(lab.title)} (실습)</button></li><li><button type="button" data-goto="check">확인하고 다음 단계로</button></li></ol></section>`});
  const firstOf={};
  cs.forEach((k,i)=>{
    const parts=splitSlides(k.body,k.slides,k.id);firstOf[i]=slides.length;
    parts.forEach((lines,j)=>slides.push({label:`강의 ${i+1} · ${k.title}`,html:`<section class="card concept-card slide-card"><div class="card-header"><h2><span class="number-tag">${i+1}</span>${h(k.title)}</h2><span class="pill">${j+1} / ${parts.length}</span></div><div class="concept-body">${md(lines,true)}</div></section>`}));
    slides.push({label:`강의 ${i+1} · 확인 퀴즈`,html:conceptQuizHTML(k,i+1,r)});
  });
  firstOf.lab=slides.length;slides.push({label:'실습',html:`<section class="card"><div class="card-header"><h2><span class="number-tag">${n+1}</span>${h(lab.title)}</h2><span class="pill">직접 해보기</span></div><div class="concept">${h(lab.concept)}</div><ol class="instructions">${c.activity.steps.map(s=>`<li>${h(s)}</li>`).join('')}<li>아래의 모든 설정을 한 번씩 실행하고, 결과의 <strong>원문</strong> 버튼을 열어 근거를 확인하세요.</li></ol><div class="lab-controls"><div class="field"><label for="lab-config">비교할 설정</label><select id="lab-config">${lab.options.map(([key,label])=>`<option value="${key}" ${r.current===key?'selected':''}>${h(label)}</option>`).join('')}</select></div><button class="primary" data-action="run">실습 실행 →</button></div><div class="run-checks">${lab.options.map(([key,label])=>`<span class="${r.runs[key]?'done':''}">${r.runs[key]?'✓':'○'} ${h(label)}</span>`).join('')}<span class="${r.sourceRead?'done':''}">${r.sourceRead?'✓':'○'} 근거 원문 확인</span></div>${output?`<div class="lab-result" aria-label="실습 결과"><p class="muted">${h(output.label)} · 설정: ${h(lab.options.find(o=>o[0]===r.current)[1])}</p>${output.html}${sourceButtons(output.sources)}</div>`:'<p class="muted" style="margin-top:16px">설정을 선택하고 실행하면 결과가 여기에 표시됩니다.</p>'}</section>`});
  firstOf.check=slides.length;slides.push({label:'확인 문제',html:`<section class="card"><div class="card-header"><h2><span class="number-tag">${n+2}</span>확인하고 다음 단계로</h2><span class="pill">${c.questions.filter(q=>r.responses[q.id]?.passed).length} / 2 통과</span></div>${!activityDone?'<p class="notice">모든 설정을 실행하고 근거 원문을 확인하면 문제를 풀 수 있습니다.</p>':''}${c.questions.map(q=>quizHTML(q,c,r,!activityDone)).join('')}</section>
  <div class="next-bar"><p>${done?'✓ 실습과 확인 문제를 모두 완료했습니다.':'설정 비교 · 원문 확인 · 두 문항 통과하면 완료로 표시됩니다.'}</p><button class="primary" data-action="next">${c.id===last()?'종합 실습으로':'다음 단계로'} →</button></div><p class="footer-note">코드 빈칸은 학생 입력을 실행하지 않고 허용된 답과 비교합니다. 전체 코드는 실습 노트북 ${h(c.notebook_ref)}에서 직접 실행하세요.</p>`});
  lessonSlides={count:slides.length,firstOf};
  const at=Math.min(slidePos[c.id]||0,slides.length-1),cur=slides[at];
  // 진행 막대를 구간(표지·강의별·실습·확인)으로 나눠 이름을 붙인다. 구간 너비 = 슬라이드 수
  firstOf.cover=0;
  const segs=[['표지','cover','표지 · 목차'],...cs.map((k,i)=>[`${i+1}`,i,`강의 ${i+1} · ${k.title}`]),['실습','lab',lab.title],['확인','check','확인 문제']].map(([short,key,full],i,a)=>({short,key,full,s:firstOf[key],e:i+1<a.length?firstOf[a[i+1][1]]:slides.length}));
  const bar=`<div class="seg-bar" role="navigation" aria-label="슬라이드 진행 구간">${segs.map(g=>{const st=at>=g.e?'done':at>=g.s?'current':'';const fill=st==='done'?100:st==='current'?Math.round((at-g.s+1)/(g.e-g.s)*100):0;return `<button type="button" class="seg ${st}" style="flex:${g.e-g.s}" data-goto="${g.key}" title="${h(g.full)} (${g.e-g.s}장)"><span class="seg-fill" style="width:${fill}%"></span><span class="seg-label">${h(st==='current'?g.full:g.short)}</span></button>`;}).join('')}</div>`;
  return `<div class="slide-top"><span class="eyebrow">STEP ${c.id} · ${h(c.title)}</span><span class="muted">${h(cur.label)}</span></div>${bar}<div class="slide">${cur.html}</div>
  <nav class="slide-nav" aria-label="슬라이드 이동"><button data-action="slide-prev" ${at===0?'disabled':''}>◀ 이전</button><span>${at+1} / ${slides.length}</span><button class="primary" data-action="slide-next" ${at===slides.length-1?'disabled':''}>다음 ▶</button></nav>`;
}
// 강의 본문 → 슬라이드. '## 소제목' 단위로 자른 뒤, 이어지는 구간을 한 화면 분량(SLIDE_BUDGET)까지 합친다.
// 합친 장의 소제목은 'A · B' 하나로 (' — ' 뒤 부연은 뺀다). 분량은 글자 수 근사: 그림 550 · 실측표 300 · 표 행 70 · 짧은 줄도 45.
const SLIDE_BUDGET=560;
function splitSlides(lines,groups,id){
  const parts=[];let cur=null;
  // '[[이어서]]' 줄: 같은 소제목으로 이어지는 새 구간(장 나눔용). "slides"에서는 '소제목 [[이어서]]' 이름으로 가리킨다.
  for(const l of lines){if(l.startsWith('## ')){cur={name:l.slice(3),titles:[l.slice(3).split(' — ')[0].trim()],body:[]};parts.push(cur);continue;}if(l==='[[이어서]]'&&cur){cur={name:cur.name+' [[이어서]]',titles:[...cur.titles],body:[]};parts.push(cur);continue;}if(!cur){cur={titles:[],body:[]};parts.push(cur);}cur.body.push(l);}
  const w=b=>b.reduce((n,l)=>n+(l.startsWith('```')?0:l.startsWith('[[그림')?550:l.startsWith('[[표')?300:l.startsWith('|')?70:Math.max(l.length,45)),0);
  // 개념에 "slides"(소제목 원문 묶음)가 있으면 그대로 따른다. 모든 섹션을 원래 순서로, 빠짐·중복·없는 이름 없이 담아야 하며 아니면 자동 분할로 되돌린다.
  let out=null;
  if(groups){const names=parts.map(p=>p.name),valid=Array.isArray(groups)&&groups.every(g=>Array.isArray(g)&&g.length),flat=valid?groups.flat():[];
    if(valid&&flat.length===names.length&&flat.every((t,i)=>t===names[i])){let i=0;out=groups.map(g=>{const ps=parts.slice(i,i+=g.length);return {titles:ps.flatMap(p=>p.titles),body:ps.flatMap(p=>p.body)};});}
    else console.warn(`[slides] ${id}: 구성 정보가 본문 소제목과 맞지 않아 자동 분할을 사용합니다.`,{groups,names});}
  if(!out){out=[];
  for(const p of parts){const pw=w(p.body),last=out[out.length-1];if(last&&last.w+pw<=SLIDE_BUDGET){last.titles.push(...p.titles);last.body.push(...p.body);last.w+=pw;}else out.push({...p,body:[...p.body],w:pw});}}
  return out.map(o=>(o.titles.length?['## '+o.titles.join(' · ')]:[]).concat(o.body));
}
// 강의 본문(줄 배열) → HTML. 지원: ## 소제목, - 목록, | 표 |, ``` 코드, > 인용, [[표 이름 rows=.. cols=..]](실측표), **굵게**, `코드`
const inline=s=>h(s).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>');
// 원문 팝업: Markdown 표(헤더 + |---| 구분 행 + 같은 열 수의 행)만 <table>로, 나머지는 원문 그대로 텍스트로 (innerHTML 미사용)
function renderSourceText(el,text){
  const cells=l=>l.trim().slice(1,-1).split('|').map(x=>x.trim()),isRow=l=>/^\s*\|.*\|\s*$/.test(l),isSep=l=>/^\s*\|(\s*:?-+:?\s*\|)+\s*$/.test(l);
  const lines=text.split('\n'),out=[];let buf=[];
  const flush=()=>{if(buf.length)out.push(document.createTextNode(buf.join('\n')));buf=[];};
  for(let i=0;i<lines.length;){
    const n=isRow(lines[i])&&isSep(lines[i+1]||'')?cells(lines[i]).length:0;
    if(!n||cells(lines[i+1]).length!==n){buf.push(lines[i++]);continue;}
    flush();const wrap=document.createElement('div'),table=document.createElement('table'),thead=table.createTHead(),tbody=table.createTBody();
    const row=(sec,cs,tag)=>{const tr=sec.insertRow();cs.forEach(c=>{const td=document.createElement(tag);td.textContent=c;tr.append(td);});};
    row(thead,cells(lines[i]),'th');i+=2;
    while(i<lines.length&&isRow(lines[i])&&cells(lines[i]).length===n)row(tbody,cells(lines[i++]),'td');
    wrap.className='source-table';wrap.append(table);out.push(wrap);
  }
  flush();el.replaceChildren(...out);
}
function tableHTML(head,rows){return `<div class="table-wrap"><table><thead><tr>${head.map(x=>`<th>${inline(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${inline(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function measured(name,opts){
  const t=course.tables[name],pick=k=>opts.match(new RegExp(k+'=([\\d,]+)'))?.[1].split(',').map(Number);
  const rows=pick('rows'),cols=pick('cols'),sel=r=>cols?cols.map(i=>r[i]):r;
  const body=(rows?rows.map(i=>t.rows[i]):t.rows).map(r=>sel(r).map((x,i)=>i===0?String(x).replace(/^\d+\.\s*/,''):x===''?'—':x));
  return `<div class="measured">${tableHTML(sel(t.headers),body)}<p class="muted">실측 · instructor/expected_results${t.note?' · '+h(t.note):''}</p></div>`;
}
// figsLast: 강의 슬라이드에서는 그림을 원래 순서대로 모아 본문(설명·목록·표·코드) 아래에 둔다
function md(lines,figsLast){
  let out='',figs='',i=0;
  while(i<lines.length){
    const l=lines[i];
    if(l.startsWith('```')){const code=[];i++;while(i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i++]);i++;out+=`<pre class="code">${h(code.join('\n'))}</pre>`;continue;}
    if(l.startsWith('|')){const rows=[];while(i<lines.length&&lines[i].startsWith('|'))rows.push(lines[i++].slice(1,-1).split('|').map(x=>x.trim()));const body=rows.slice(1).filter(r=>!r.every(x=>/^-+$/.test(x)));out+=tableHTML(rows[0],body);continue;}
    if(l.startsWith('- ')){out+='<ul>';while(i<lines.length&&lines[i].startsWith('- '))out+=`<li>${inline(lines[i++].slice(2))}</li>`;out+='</ul>';continue;}
    if(/^\d+\. /.test(l)){out+='<ol>';while(i<lines.length&&/^\d+\. /.test(lines[i]))out+=`<li>${inline(lines[i++].replace(/^\d+\. /,''))}</li>`;out+='</ol>';continue;}
    if(l.startsWith('> ')){out+='<blockquote>';while(i<lines.length&&lines[i].startsWith('> '))out+=`<p>${inline(lines[i++].slice(2))}</p>`;out+='</blockquote>';continue;}
    const f=l.match(/^\[\[그림 ([\w./-]+)(?: (\d+))?(?: min(\d+))?(?: fit(\d+))?(?: (vh))?(?: \+(\d+))?(?: w(\d+))?\]\]$/);if(f){const t=course.figures?.[f[1]]||'',src=f[1].includes('/')?f[1]:`img/${f[1]}.svg`;const fig=`<figure class="lecture-fig"${f[6]?` style="margin-top:${12+Number(f[6])}px"`:''}><img src="${h(src)}" alt="${h(t)}" loading="lazy"${f[5]?' class="fig-vh"':''}${f[2]?` style="--fig-max:${f[2]}px${f[3]?`;--fig-min:${f[3]}px`:''}${f[4]?`;--fig-fit:${f[4]}px`:''}${f[7]?`;--fig-w:${f[7]}px`:''}"`:''}><figcaption>${h(t)}</figcaption></figure>`;if(figsLast)figs+=fig;else out+=fig;i++;continue;}
    const m=l.match(/^\[\[표 (\w+)([^\]]*)\]\]$/);
    out+=m?measured(m[1],m[2]):l.startsWith('## ')?`<h3>${inline(l.slice(3))}</h3>`:`<p>${inline(l)}</p>`;i++;
  }
  return out+figs;
}
function conceptQuizHTML(k,no,r){
  const quiz=(q)=>{const saved=r.concepts?.[q.id],message=feedback[q.id];
    return `<div class="quiz ${q.required?'required':''}" id="quiz-${q.id}"><p class="quiz-tag">${q.required?'<span class="badge-req">★ 필수</span>':''}${saved?.passed?'<span class="ok-tag">✓ 정답</span>':''}</p><fieldset><legend>${h(q.prompt)}</legend>${q.options.map((o,i)=>`<label class="option"><input type="radio" name="${q.id}" value="${i}" ${saved?.value===i?'checked':''}><span>${h(o)}</span></label>`).join('')}</fieldset><div class="quiz-actions"><button class="primary" data-concept="${q.id}">답 확인</button></div>${message?`<p class="feedback ${message.ok?'success':''}" role="status">${h(message.text)}</p>`:saved?.passed?`<p class="feedback success">✓ 정답 · ${h(q.explanation)}</p>`:''}</div>`;};
  return `<section class="card concept-card slide-card"><div class="card-header"><h2><span class="number-tag">${no}</span>확인 퀴즈 · ${h(k.title)}</h2><span class="pill">${k.quizzes.length}문항 · ★ 표시는 수업 중 필수</span></div>${k.quizzes.map(quiz).join('')}</section>`;
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
  return `<div class="eyebrow">MY LEARNING RECORD</div><h1>내 학습 기록</h1><p class="muted">이 브라우저에 저장된 기록입니다. 강사에게 결과 파일을 제출하세요.</p><div class="summary-grid"><div class="summary-stat"><small>회차 완료</small><strong>${count} / ${N()}</strong></div><div class="summary-stat"><small>종합 실습 제출</small><strong>${capCount} / 3</strong></div><div class="summary-stat"><small>강사 확인</small><strong style="font-size:1.2rem">${capCount===3?'검토 대기':'제출 전'}</strong></div></div><section class="card"><label class="field"><span>제출용 학습자 표시명 (선택)</span><input type="text" id="learner-name" maxlength="80" value="${h(state.learnerName)}" placeholder="예: 3조 / 학습자 07"></label><p class="muted">학습자 ID: ${h(state.learnerId)}<br>자료 버전: ${h(course.version)} · 저장: ${state.savedAt?h(new Date(state.savedAt).toLocaleString('ko-KR')):'아직 없음'}</p><div class="table-wrap"><table><thead><tr><th>회차</th><th>실습 실행</th><th>문항</th><th>힌트</th><th>상태</th></tr></thead><tbody>${course.bank.chapters.map(c=>{const r=state.chapters[c.id];return `<tr><td>${c.id} ${h(c.title)}</td><td>${labs[c.id].required.filter(k=>r?.runs?.[k]).length}/${labs[c.id].required.length}</td><td>${c.questions.filter(q=>r?.responses?.[q.id]?.passed).length}/2</td><td>${c.questions.some(q=>r?.responses?.[q.id]?.assisted)?'사용':'—'}</td><td>${passed(c)?'완료':'미완료'}</td></tr>`;}).join('')}</tbody></table></div><div class="quiz-actions"><button class="primary" data-action="export">결과 JSON 내려받기</button><button data-action="import">다른 기기의 결과 가져오기</button></div><input type="file" id="import-file" accept=".json,application/json" hidden><p class="notice" style="margin-top:18px">${storageOK?'진도는 이 브라우저에 저장됩니다.':'브라우저 저장을 사용할 수 없습니다.'} 중앙 서버에 제출되지는 않습니다. 파일 가져오기는 현재 기록을 대체합니다. 강사는 종합 실습의 근거와 선택 이유를 별도로 검토합니다.</p><button data-action="reset">이 브라우저의 실습 기록 초기화</button></section>`;
}
function colabURL(){const id=course.bank.colab.notebook_id;return id?`https://colab.research.google.com/drive/${encodeURIComponent(id)}`:'';}
function notebookLinks(c){const url=colabURL();if(c.id===course.bank.chapters[0].id)return `<p class="notebook-links">${url?`<a class="colab-link" href="${h(url)}" target="_blank" rel="noopener">▶ Colab에서 실습 노트북 열기</a>`:''}<span class="muted">하루 동안 이 노트북 하나로 실습합니다. 첫 코드 셀(Colab 준비)을 한 번 실행하세요.<br>이 단계: ${h(c.notebook_ref)}</span></p>`;return `<p class="notebook-links"><span class="muted">실습 노트북 <strong>${h(c.notebook_ref)}</strong> — Colab 왼쪽 목차(☰)에서 Part 제목을 눌러 이동하세요.</span></p>`;}
function overviewHTML(){return `<div class="eyebrow">HF RAG · ${h(course.bank.schedule)}</div><h1>기법을 하나씩 더하며,<br>숫자로 확인하는 RAG.</h1><p>한국주택금융공사 업무를 본뜬 가상 문서 44개로 RAG의 근거부터 임베딩 · 하이브리드 검색 · 리랭킹 · 쿼리 변환 · 평가 · 지식 그래프 · 경로 선택까지 단계별로 확인합니다.</p><section class="card concept-body"><div class="eyebrow">오리엔테이션 · 09:00–09:15</div>${md(course.bank.intro)}</section><div class="next-bar"><p class="muted">화면의 검색 결과·답변·수치는 강사 PC(CPU, qwen3:1.7b · bge-m3)에서 평가셋 40문항으로 측정한 실제 결과입니다.</p><button class="primary" data-action="resume">${course.bank.chapters.some(passed)?'실습 이어하기':'첫 실습 시작하기'} →</button></div>`;}
function capstoneHTML(){
  
  return `<div class="eyebrow">FINAL PRACTICE · ${course.bank.capstone.minutes} MIN</div><h1>질문에서 근거까지,<br>직접 경로를 선택하세요.</h1><p class="muted">질문 3개에 답하고 결과 파일을 내려받아 강사에게 제출하세요. 작성 완료는 자동 확인하며 답변의 타당성은 강사가 검토합니다.</p><p class="notice">다른 경로를 선택해도 충분한 근거가 있으면 인정합니다. 최소 한 질문에서 검색 단계와 생성 단계 중 우선 점검할 곳을 설명하세요.</p>${course.bank.capstone.questions.map((q,i)=>{const r=state.capstone[q.id]||{};const eq=course.questions[q.qid],ids=eq.relevant_documents;return `<section class="card capstone-form" data-final="${q.id}"><p class="eyebrow">질문 ${i+1} ${r.submitted?'· 작성 완료 / 강사 검토 대기':''}</p><h2>${h(eq.question)}</h2>${sourceButtons(ids)}${[['chosen_route','선택한 검색 경로','Basic RAG / Hybrid+Rerank / 쿼리 변환 / 인지평가 / Graph+Hybrid 등'],['source_refs','확인한 원문 출처','문서 ID와 답변을 뒷받침하는 문장'],['answer','근거에 따른 답변','원문에 없는 사실을 추가하지 마세요'],['failure_or_limit','실패 또는 한계','근거 누락·요약 손실·비용 등. 한 질문 이상은 검색/생성 중 우선 점검할 단계를 설명'],['reason_for_selection','비교 후 이 경로를 선택한 이유','다른 경로와 비교한 이유']].map(([field,label,placeholder])=>`<label class="field"><span>${label}</span><textarea data-final-id="${q.id}" data-field="${field}" maxlength="4000" placeholder="${placeholder}">${h(r[field]||'')}</textarea></label>`).join('')}<button class="primary" data-submit-final="${q.id}">이 질문 작성 완료</button></section>`;}).join('')}<div class="next-bar"><p>세 질문 모두 작성한 뒤 결과 JSON을 강사에게 제출하세요.</p><button class="primary" data-action="export">결과 내려받기 ↓</button></div>`;
}
function setView(next){view=next;feedback={};render();document.querySelector('#main').focus();window.scrollTo({top:0,behavior:'instant'});}
function exportReport(){
  save();
  const evidence=course.bank.chapters.flatMap(c=>Object.keys(state.chapters[c.id]?.runs||{}).map(key=>({chapter_id:c.id,config:key,output_ref:`${course.version}/${c.id}/${key}`,source_refs:runLab(c.id,key,course).sources})));
  const report={...state,reportType:'hf-rag-quiz',exportedAt:new Date().toISOString(),evidence,summary:{chaptersPassed:course.bank.chapters.filter(passed).map(c=>c.id),capstoneSubmitted:course.bank.capstone.questions.filter(q=>state.capstone[q.id]?.submitted).map(q=>q.id),reviewStatus:'강사 검토 전',storage:'브라우저 로컬'}};
  const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`hf-rag-quiz-${state.learnerId.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('결과 파일을 내려받았습니다. 강사에게 제출하세요.');
}
app.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button||button.disabled)return;
  if(button.dataset.chapter){state.active=button.dataset.chapter;save();setView('lesson');return;}
  if(button.dataset.source){const doc=course.docs[button.dataset.source];if(!doc)return;document.querySelector('#source-title').textContent=doc.title;document.querySelector('#source-path').textContent=doc.source;renderSourceText(document.querySelector('#source-text'),doc.text);
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
  if(button.dataset.goto!=null){goSlide(lessonSlides.firstOf[button.dataset.goto]);return;}
  if(button.dataset.action==='slide-prev'){goSlide((slidePos[state.active]||0)-1);return;}
  if(button.dataset.action==='slide-next'){goSlide((slidePos[state.active]||0)+1);return;}
  if(button.dataset.concept){
    const c=course.bank.chapters.find(c=>c.id===state.active),k=c.concepts.flatMap(k=>k.quizzes).find(q=>q.id===button.dataset.concept),r=record(c.id);
    const el=app.querySelector(`input[name="${k.id}"]:checked`);if(!el){toast('답을 선택하세요.');return;}
    const value=Number(el.value),ok=value===k.answer_index;r.concepts||={};
    if(ok||!r.concepts[k.id]?.passed)r.concepts[k.id]={value,passed:ok};
    feedback[k.id]={ok,text:ok?'✓ 정답 · '+k.explanation:'아직 아닙니다. 위 설명을 다시 읽고 골라 보세요.'};save();render();document.querySelector('#quiz-'+k.id)?.scrollIntoView({block:'nearest'});return;
  }
  if(button.dataset.submitFinal){const id=button.dataset.submitFinal,r=state.capstone[id];if(!capstoneFilled(r)){toast('다섯 항목을 모두 작성하세요.');return;}r.submitted=true;r.submittedAt=new Date().toISOString();save();render();toast('작성 완료로 저장했습니다. 강사 검토는 결과 제출 후 진행됩니다.');return;}
  const action=button.dataset.action;
  if(action==='run'){const r=record(state.active),key=document.querySelector('#lab-config').value;runLab(state.active,key,course);r.runs[key]={config:key,at:new Date().toISOString()};r.current=key;save();render();document.querySelector('.lab-result').scrollIntoView({block:'nearest'});}
  if(action==='next'){const i=course.bank.chapters.findIndex(c=>c.id===state.active);if(i===N()-1)setView('capstone');else{state.active=course.bank.chapters[i+1].id;save();setView('lesson');}}
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
document.addEventListener('keydown',e=>{if(view!=='lesson'||e.altKey||e.ctrlKey||e.metaKey||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();goSlide((slidePos[state.active]||0)+1);}if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();goSlide((slidePos[state.active]||0)-1);}});
document.querySelector('#close-source').addEventListener('click',()=>dialog.close());
try {
  const response=await fetch('./course.json');if(!response.ok)throw Error('강의 자료 응답 오류');course=await response.json();storageKey='hf-rag-quiz:'+course.version;initLabs(course);
  let saved;try{saved=localStorage.getItem(storageKey);}catch{storageOK=false;}
  try{state=saved?validState(JSON.parse(saved)):fresh();}catch{state=fresh();toast('저장 기록을 읽지 못했습니다. 결과 파일이 있다면 가져오세요.');}
  view=Object.keys(state.chapters).length?'lesson':'overview';render();
} catch(error) {app.innerHTML=`<main class="main"><h1>실습 자료를 열 수 없습니다</h1><p>${h(error.message)}</p><p>인터넷 연결 또는 로컬 서버 실행 상태를 확인하고 새로고침하세요.</p></main>`;}
