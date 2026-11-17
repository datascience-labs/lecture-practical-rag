import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gradeCode, rrf, metrics, fixedChunks, neighbors, chapterPassed, canOpen} from './dist/core.mjs';
import {labs, runLab, codeCards} from './dist/labs.mjs';
const {bank,docs}=JSON.parse(readFileSync(new URL('./dist/course.json',import.meta.url)));
test('RRF는 서로 다른 순위 목록을 실제로 합산한다',()=>{
  const ranking=rrf([['b','a','c'],['a','c','b']]);assert.equal(ranking[0][0],'a');assert.equal(ranking[0][1],1/62+1/61);
});
test('정답 누락·복수 정답·순위 변화의 평가값',()=>{
  assert.deepEqual(metrics(['X','A','Y'],['A','B'],3),{hit:1,recall:.5,rr:.5});
  assert.equal(metrics(['X','Y','A'],['A','B'],3).rr,1/3);
  assert.deepEqual(metrics(['X','Y','Z'],['A','B'],3),{hit:0,recall:0,rr:0});
});
test('코드 채점은 분모 구조와 대소문자를 지키고 입력을 실행하지 않는다',()=>{
  assert(gradeCode(' ( 1 / (k + rank) ) ',['1 / (k + rank)']));
  assert(!gradeCode('1 / k + rank',['1 / (k + rank)']));
  assert(!gradeCode('reverse=true',['reverse=True']));
  assert(!gradeCode('context; alert(1)',['context']));
  assert(!gradeCode('',['context']));
});
test('고정 청킹은 오버랩을 보존하고 잘못된 간격을 거부한다',()=>{
  assert.deepEqual(fixedChunks('abcdefghij',5,2),['abcde','defgh','ghij']);
  assert.throws(()=>fixedChunks('abc',5,5));
});
test('김철수의 생산 공장은 2-hop에서 확인된다',()=>{
  assert(!neighbors('김철수',1).some(e=>e[2]==='제조1공장'));
  assert(neighbors('김철수',2).some(e=>e[2]==='제조1공장'));
});
test('모든 실습 설정과 코드 카드가 원문·문항에 연결된다',()=>{
  for(const chapter of bank.chapters){
    const lab=labs[chapter.id];assert(lab);
    for(const key of lab.required){const output=runLab(chapter.id,key,docs);assert(output.html.length>50);assert(output.sources.length);for(const source of output.sources)assert(docs[source],source);}
    const q=chapter.questions.find(q=>q.type==='code_blank');assert.equal(codeCards[chapter.id].filter(v=>gradeCode(v,q.answers)).length,1);
  }
});
test('실행만으로 통과하지 않으며 모든 선행 회차를 검증한다',()=>{
  const chapter=bank.chapters[0],required=labs['01'].required;
  const record={runs:Object.fromEntries(required.map(k=>[k,{config:k}])),sourceRead:false,responses:{}};
  assert(!chapterPassed(chapter,record,required));record.sourceRead=true;
  assert(!chapterPassed(chapter,record,required));
  for(const q of chapter.questions)record.responses[q.id]={passed:true};
  assert(chapterPassed(chapter,record,required));
  assert(canOpen(1,bank.chapters,{'01':record},labs));
  assert(!canOpen(2,bank.chapters,{'01':record},labs));
});
