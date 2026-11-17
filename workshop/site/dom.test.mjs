import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parseHTML} from 'linkedom';
import {labs} from './dist/labs.mjs';
const course=JSON.parse(readFileSync(new URL('./dist/course.json',import.meta.url)));
const storage=new Map();
let bootCount=0,doc,download;
async function boot(){
  const parsed=parseHTML(readFileSync(new URL('./dist/index.html',import.meta.url),'utf8'));
  doc=parsed.document;globalThis.document=doc;globalThis.window=parsed.window;
  window.scrollTo=()=>{};window.HTMLElement.prototype.scrollIntoView=()=>{};
  doc.querySelector('dialog').showModal=function(){this.setAttribute('open','');};
  doc.querySelector('dialog').close=function(){this.removeAttribute('open');};
  globalThis.localStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)};
  globalThis.fetch=async()=>({ok:true,json:async()=>structuredClone(course)});
  globalThis.confirm=()=>true;
  URL.createObjectURL=blob=>{download=blob;return 'blob:unit-test';};URL.revokeObjectURL=()=>{};
  await import(`./dist/app.js?dom-test=${++bootCount}`);
}
function click(selector){const el=doc.querySelector(selector);assert(el,selector);assert(!el.hasAttribute('disabled'),selector+' is disabled');el.click();}
function choose(name,value,event='input'){
  const elements=[...doc.querySelectorAll(`input[name="${name}"]`)];
  const el=elements.find(el=>el.value===String(value));assert(el,`${name}: ${value}`);
  for(const e of elements){e.removeAttribute('checked');e.checked=false;}el.setAttribute('checked','');el.checked=true;
  el.dispatchEvent(new window.Event(event,{bubbles:true}));
}
function fill(selector,value){const el=doc.querySelector(selector);assert(el,selector);el.value=value;el.dispatchEvent(new window.Event('input',{bubbles:true}));}
function run(key){const select=doc.querySelector('#lab-config');Object.defineProperty(select,'value',{value:key,configurable:true});click('[data-action="run"]');}
function saved(){return JSON.parse(storage.values().next().value);}
test('12회차 전체 흐름·오답·카드·진도 복원·종합 실습·결과 파일',async()=>{
  await boot();assert(doc.querySelector('h1').textContent.includes('기초 RAG'));
  click('[data-action="resume"]');assert(doc.querySelector('[data-chapter="02"]').hasAttribute('disabled'));
  for(const [index,c] of course.bank.chapters.entries()){
    assert(doc.querySelector('h1').textContent.includes(c.title));
    assert(doc.querySelector(`[data-check="${c.questions[0].id}"]`).hasAttribute('disabled'));
    for(const key of labs[c.id].required)run(key);
    assert(doc.querySelector('[data-action="next"]').hasAttribute('disabled'));
    click('[data-source]');assert(doc.querySelector('dialog').hasAttribute('open'));click('#close-source');
    const mc=c.questions[0],code=c.questions[1];
    if(index===0){
      fill(`input[name="${code.id}"]`,'cont');
      await boot();assert(doc.querySelector('h1').textContent.includes(c.title));assert.equal(doc.querySelector(`input[name="${code.id}"]`).value,'cont');
      choose(mc.id,(mc.answer_index+1)%4);click(`[data-check="${mc.id}"]`);assert(doc.querySelector('.feedback').textContent.includes('아직'));
      click(`[data-hint="${mc.id}"]`);assert(doc.querySelector('.hint'));
    }
    choose(mc.id,mc.answer_index);click(`[data-check="${mc.id}"]`);
    if(index===5){fill(`input[name="${code.id}"]`,'1 / k + rank');click(`[data-check="${code.id}"]`);assert(doc.querySelector('[data-action="next"]').hasAttribute('disabled'));}
    if(index%2){choose(code.id+'-mode','card','change');choose(code.id,code.answers[0]);}
    else fill(`input[name="${code.id}"]`,code.answers[0]);
    click(`[data-check="${code.id}"]`);assert(!doc.querySelector('[data-action="next"]').hasAttribute('disabled'));
    if(index===0){
      assert(saved().chapters['01'].responses[mc.id].assisted);
      click('[data-action="next"]');
    }else click('[data-action="next"]');
  }
  assert(doc.querySelector('h1').textContent.includes('직접 경로'));
  click('[data-submit-final="final-1"]');assert(!saved().capstone['final-1']?.submitted);
  for(const q of course.bank.capstone.questions){for(const field of ['chosen_route','source_refs','answer','failure_or_limit','reason_for_selection'])fill(`[data-final-id="${q.id}"][data-field="${field}"]`,'테스트 작성: '+field);click(`[data-submit-final="${q.id}"]`);}
  fill('[data-final-id="final-1"][data-field="failure_or_limit"]','수정한 검색 실패 진단');assert(!saved().capstone['final-1'].submitted);assert(doc.querySelector('[data-final="final-1"] .eyebrow').textContent.includes('수정 중'));click('[data-submit-final="final-1"]');
  click('[data-action="export"]');const report=JSON.parse(await download.text());assert.equal(report.summary.chaptersPassed.length,12);assert.equal(report.summary.capstoneSubmitted.length,3);assert(report.evidence.length>=24);
  await boot();click('[data-action="capstone"]');assert.equal(doc.querySelector('[data-final-id="final-1"][data-field="answer"]').value,'테스트 작성: answer');
  click('[data-action="summary"]');assert(doc.querySelector('main').textContent.includes('검토 대기'));
  // 가져오기에서 잘못된 버전은 현재 진도를 덮어쓰지 않는다.
  const fileInput=doc.querySelector('#import-file');Object.defineProperty(fileInput,'files',{configurable:true,value:[{size:20,text:async()=>JSON.stringify({...report,version:'wrong'})}]});fileInput.dispatchEvent(new window.Event('change',{bubbles:true}));await new Promise(setImmediate);assert.equal(Object.keys(saved().chapters).length,12);
  // 잘못된 선행 답변은 재채점되며 뒤 회차는 잠긴다.
  const damaged=structuredClone(report);damaged.chapters['01'].responses['01-code'].value='wrong';Object.defineProperty(fileInput,'files',{configurable:true,value:[{size:200,text:async()=>JSON.stringify(damaged)}]});fileInput.dispatchEvent(new window.Event('change',{bubbles:true}));await new Promise(setImmediate);assert(doc.querySelector('[data-chapter="02"]').hasAttribute('disabled'));
});
