import {rrf, metrics, fixedChunks, neighbors, cosine} from './core.mjs';
export const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const box = (title, text) => `<div class="result-box"><h3>${h(title)}</h3><p>${h(text)}</p></div>`;
const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(t=>`<th>${h(t)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(t=>`<td>${h(t)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const definitions = [
  ['01', '근거를 넣으면 무엇이 달라질까요?', '질문에 관련된 문서를 찾아 프롬프트에 함께 제공하는 것이 RAG입니다. 검색된 내용과 답변의 주장이 일치하는지까지 확인해야 합니다.', [['without','문서 컨텍스트 없음'],['with','핸드북 2.3 제공']], '강사 작성 예제 · 실제 모델 호출 없음'],
  ['02', '단어가 달라도 의미를 찾을 수 있을까요?', '임베딩은 문장을 벡터로 표현합니다. 코사인 유사도는 방향의 유사성이지 문서가 참일 확률이 아닙니다. 여기서는 계산을 이해하기 위한 2차원 교육용 벡터를 사용합니다.', [['semantic','의미 질문: 집에서 일하기'],['code','코드 질문: PD-2000']], '브라우저 계산 · 교육용 벡터, BGE 실측 아님'],
  ['03', '검색 범위를 좁혀 보세요', '벡터 DB는 임베딩과 문서·메타데이터를 연결합니다. 여기서는 준비된 후보 목록에 category 조건을 적용합니다. category는 이 실습에서 부여한 분류입니다.', [['all','필터 없음'],['hr','category = HR']], '브라우저 필터 · 강사 작성 후보 순위'],
  ['04', '일반 규정과 예외를 함께 읽으세요', '문서를 나누는 경계가 답변에 필요한 조건을 잘라낼 수 있습니다. 오버랩과 구조 기반 분할은 문맥을 보존하는 방법입니다. 직접 만든 청크를 보여주며 검색 성능은 측정하지 않습니다.', [['fixed','고정 크기: 70자, 오버랩 0'],['overlap','고정 크기: 70자, 오버랩 25'],['structure','구조 기반: 3.1 절 전체']], '브라우저에서 실제 청킹'],
  ['05', '틀린 답변의 원인을 찾아보세요', '검색 → 문맥 조립 → 생성은 서로 다른 단계입니다. 정답 근거가 전달되지 않았는지, 전달됐지만 답변에서 무시됐는지를 나눠 진단합니다. 답할 근거가 없는 질문은 보류합니다.', [['grounded','정상: 재택근무 근거 연결'],['missing','실패 A: 예외 근거 누락'],['generation','실패 B: 근거는 있지만 답변 오류'],['unanswerable','문서에 정답 없음: 주차비']], '강사 작성 진단 사례 · 실제 모델 호출 없음'],
  ['06', '두 검색의 순위를 합쳐 보세요', '벡터 검색은 의미, BM25는 단어 일치에 강점을 가집니다. RRF는 점수 척도 대신 순위를 사용합니다. 아래는 알고리즘을 확인하기 위한 교육용 순위표입니다.', [['vector','벡터 검색 후보'],['bm25','BM25 검색 후보'],['hybrid','두 목록을 RRF로 융합']], '브라우저 RRF 계산 · 순위는 교육용'],
  ['07', '후보와 최종 순위를 구분하세요', '리랭커는 질문과 후보 문서를 함께 평가해 순서를 바꿉니다. 여기서는 강사가 지정한 관련도 점수를 정렬합니다. 입력 후보에 없는 문서는 복구할 수 없습니다.', [['before','리랭킹 전'],['after','리랭킹 후'],['missing','정답이 후보에 없는 경우']], '브라우저 정렬 · 점수는 교육용, 모델 실측 아님'],
  ['08', '검색할 수 있는 독립 질문으로 바꾸세요', '대화에서 생략된 대상을 복원하거나 복합 질문을 작은 질문으로 나눕니다. 재작성에서 정답을 미리 단정하지 않습니다. 아래 변환 문장은 강사가 작성한 예제입니다.', [['raw','원래 질문: 그럼 이월은 돼?'],['rewrite','대화 맥락으로 재작성'],['decompose','PD 제품 비교 질문 분해']], '강사 작성 변환 예제 · 실제 모델 호출 없음'],
  ['09', '정답이 몇 위에 있는지 계산하세요', 'Hit@k는 정답 포함 여부, Recall@k는 전체 정답 중 찾은 비율입니다. 첫 정답 역순위의 여러 질문 평균이 MRR입니다. 검색 평가와 답변 충실성 평가는 별개입니다.', [['rank2','검색 [X, A, Y] · 정답 {A, B}'],['rank3','검색 [X, Y, A] · 정답 {A, B}'],['missing','검색 [X, Y, Z] · 정답 {A, B}']], '브라우저에서 실제 지표 계산'],
  ['10', '관계와 출처를 따라가세요', '지식 그래프는 개체와 관계를 연결합니다. 담당자 → 제품 → 생산 공장의 두 관계를 따라가면 문서 간 질문을 풀 수 있습니다. 아래 그래프는 원문을 검수해 강사가 구성했습니다.', [['one','김철수에서 1-hop'],['two','김철수에서 2-hop']], '브라우저 그래프 탐색 · 강사 검수 트리플'],
  ['11', 'Local과 Global의 입력을 비교하세요', 'Local Search는 특정 개체 주변의 관계·원문을 사용합니다. Global Search는 여러 커뮤니티 리포트의 부분 답변을 종합합니다. 여기서는 강사가 묶은 주제와 리포트를 보여주며 커뮤니티 탐지를 실행하지 않습니다.', [['local','Local: 김철수의 담당 제품'],['global','Global: 전체 사업 구조']], '강사 작성 커뮤니티·리포트 예제'],
  ['12', '질문별로 적절한 경로를 선택하세요', '기본 RAG·Local·Global은 서로 다른 근거 조립 전략입니다. 여기서는 각 경로가 제공할 수 있는 근거 범위를 비교합니다. 실제 품질·지연시간 비교에는 같은 평가셋과 실행 환경의 측정이 필요합니다.', [['single','단일 사실: PD-3000 최대 출력'],['multi','관계 질문: 김철수 담당 제품의 공장'],['global','전역 질문: 전체 사업 구조']], '강사 작성 경로 비교 · 실측 성능표 아님']
];
export const labs = Object.fromEntries(definitions.map(([id,title,concept,options,label])=>[id,{title,concept,options,label,required:options.map(o=>o[0])}]));

export function runLab(id, key, docs) {
  if (!labs[id]?.options.some(o=>o[0]===key)) throw Error('지원하지 않는 실습 설정입니다.');
  const section = n => docs['handbook-'+n].text;
  const product = ['04-product-pd2000','05-product-pd3000'];
  let html = '', sources = [];
  if (id === '01') {
    html = box('질문', '재택근무는 주 몇 회까지 가능한가요?') + (key === 'without'
      ? box('근거 없는 예시 답변', '주 3회 가능합니다. → 이 횟수를 뒷받침하는 문서가 없습니다.')
      : box('핸드북 2.3을 제공한 예시 답변', '주 2회까지 팀장 승인 하에 가능합니다. 수습 기간에는 사용할 수 없습니다. [핸드북 2.3]'));
    sources = ['handbook-2.3'];
  } else if (id === '02') {
    const query = key === 'semantic' ? [1,0.2] : [0.1,1];
    const vectors = [['재택근무',[0.96,0.22],'handbook-2.3'],['연차 휴가',[0.6,0.6],'handbook-3.1'],['PD-2000',[0.1,0.98],product[0]],['PD-3000',[0.13,0.96],product[1]]];
    const ranked = vectors.map(([name,v,source])=>({name,v,source,score:cosine(query,v)})).sort((a,b)=>b.score-a.score);
    html = box('계산 입력', `질문 벡터 [${query.join(', ')}]. 이 숫자는 원리를 보여주기 위해 직접 지정했습니다.`) + table(['문서','교육용 벡터','코사인 유사도'], ranked.map(r=>[r.name,JSON.stringify(r.v),r.score.toFixed(4)])) + box('확인할 것','PD-2000과 PD-3000은 벡터가 가깝지만 서로 다른 제품입니다. 모델 실측 결과나 정답 확률로 해석하지 마세요.');
    sources = ranked.map(r=>r.source);
  } else if (id === '03') {
    const candidates = ['handbook-2.3','05-product-pd3000','handbook-3.1','04-product-pd2000'];
    const filtered = candidates.filter(s=>key==='all'||docs[s].category==='HR');
    html = table(['문서 ID','category','내용'], filtered.map(s=>[s,docs[s].category,docs[s].text.slice(0,100)])) + box('결과', `${filtered.length}개 문서. 필터는 원문을 삭제하거나 수정하지 않습니다.`);
    sources = filtered;
  } else if (id === '04') {
    const text = section('3.1').split('\n\n').find(s=>s.includes('사용하지 못한 연차'));
    const chunks = key==='structure' ? [section('3.1')] : fixedChunks(text,70,key==='overlap'?25:0);
    html = box('질문','퇴직 예정자도 연차를 이월할 수 있나요?') + chunks.map((s,i)=>box(`청크 ${i+1}${s.includes('퇴직 예정자')?' · 예외 조항 확인':''}`,s)).join('') + box('관찰','일반 이월 규정과 퇴직 예정자 예외를 함께 확인하세요. 조각을 만들었다고 검색 품질까지 검증한 것은 아닙니다.');
    sources = ['handbook-3.1'];
  } else if (id === '05') {
    const examples = {
      grounded:['재택근무는 주 몇 회 가능한가요?',section('2.3'),'주 2회까지 팀장 승인 하에 가능합니다. [핸드북 2.3]','근거와 답변 조건이 일치합니다.'],
      missing:['퇴직 예정자도 연차 이월이 가능한가요?','사용하지 못한 연차는 최대 5일까지 다음 해로 이월할 수 있다.','최대 5일 이월 가능합니다.','검색·문맥 조립 실패: 예외 조항이 빠졌습니다.'],
      generation:['퇴직 예정자도 연차 이월이 가능한가요?',section('3.1'),'퇴직 예정자도 5일 이월 가능합니다.','생성 실패: 예외 근거는 전달됐으나 답변이 무시했습니다.'],
      unanswerable:['주차비 지원 금액은 얼마인가요?','관련 주차비 규정 없음','제공된 문서에서 답을 찾을 수 없습니다.','근거가 없으면 수치를 추측하지 않고 답변을 보류합니다.']
    };
    const [question,context,answer,diagnosis] = examples[key];
    html = box('질문',question)+box('검색 → 최종 컨텍스트',context)+box('예시 답변',answer)+box('진단',diagnosis);
    sources = key==='grounded'?['handbook-2.3']:['handbook-3.1'];
  } else if (id === '06') {
    const vec = ['05-product-pd3000','04-product-pd2000','06-product-ess500'];
    const bm = ['04-product-pd2000','06-product-ess500','05-product-pd3000'];
    html = box('질문','PD-2000 스펙 알려줘. 아래 후보 순위는 강사 작성 예제입니다.');
    if (key==='hybrid') html += table(['융합 순위','문서','RRF 점수 (k=60)'],rrf([vec,bm]).map(([s,score],i)=>[i+1,s,score.toFixed(6)]));
    else html += table(['순위','후보 문서'],(key==='vector'?vec:bm).map((s,i)=>[i+1,s]));
    html += box('RRF 공식','문서별로 각 목록에서의 1 / (60 + rank)를 합합니다. rank는 1부터 시작합니다.');
    sources = vec;
  } else if (id === '07') {
    let rows = [['일반 이월 조항',0.42],['재택근무 조항',0.01],['퇴직 예정자 예외 조항',0.87]];
    if(key==='missing') rows=rows.slice(0,2);
    if(key!=='before') rows.sort((a,b)=>b[1]-a[1]);
    html = box('질문','퇴직 예정자도 연차 이월이 가능한가요?')+table(['순위','후보','교육용 관련도 점수'],rows.map(([s,v],i)=>[i+1,s,v]))+box('진단',key==='missing'?'정답 예외 조항이 후보에 없습니다. 점수 정렬만으로 복구할 수 없습니다.':'점수가 높은 후보부터 정렬합니다. 실제 cross-encoder를 실행한 점수는 아닙니다.');
    sources=['handbook-3.1','handbook-2.3'];
  } else if (id === '08') {
    if(key==='decompose') html=box('원래 질문','PD-2000과 PD-3000의 출력·무게·생산 공장은 어떻게 다른가요?')+box('분해 1','PD-2000의 출력·무게·생산 공장은?')+box('분해 2','PD-3000의 출력·무게·생산 공장은?')+box('조립 시 확인','두 결과의 속성을 같은 기준으로 나란히 비교합니다.');
    else html=box('대화 맥락','이전: 연차 규정 알려줘. 현재: 그럼 이월은 돼?')+box('검색에 전달할 질문',key==='raw'?'그럼 이월은 돼? → 무엇을 이월하는지 생략되어 있습니다.':'연차 휴가를 다음 해로 이월할 수 있는가? → 가능 여부를 미리 단정하지 않습니다.');
    sources=key==='decompose'?product:['handbook-3.1'];
  } else if (id === '09') {
    const ids={rank2:['X','A','Y'],rank3:['X','Y','A'],missing:['X','Y','Z']}[key];
    const m=metrics(ids,['A','B'],3);
    html=box('입력',`검색 [${ids.join(', ')}], 정답 {A, B}, k=3. A·B·X·Y·Z는 계산 연습용 ID입니다.`)+`<div class="metric-grid">${[['Hit@3',m.hit],['Recall@3',m.recall],['이 질문의 RR',m.rr]].map(([name,v])=>`<div class="metric"><small>${name}</small><strong>${v.toFixed(3)}</strong></div>`).join('')}</div>`+box('비교 기준','질문별 RR을 평균하면 MRR입니다. 검색 지표만 높아도 답변이 근거와 다를 수 있습니다. 실제 평가에서는 문항·정답 라벨·k를 고정하고 결과를 비교합니다.');
    sources=['handbook-3.1'];
  } else if (id === '10') {
    const found=neighbors('김철수',key==='one'?1:2);
    html=key==='two'?'<div class="route-path">김철수 → PD-2000 → 제조1공장</div>':'';
    html+=table(['주어','관계','목적어','출처'],found.map(([s,r,o,source])=>[s,r,o,source]))+box('관찰','2-hop에서 생산 공장까지 연결됩니다. 트리플은 원문에서 직접 확인하며, 깊이를 늘릴수록 불필요한 근거도 늘 수 있습니다.');
    sources=[...new Set(found.map(e=>e[3]))];
  } else if (id === '11') {
    if(key==='local') { html=box('질문','김철수 과장의 담당 제품은?')+box('Local 근거 조립','개체: 김철수 / 관계: 김철수 → 담당 → PD-2000 / 원문: 영업 담당자 문서')+box('예시 답변','김철수 과장은 PD-2000 제품 판매를 담당합니다. [03-sales-people]');sources=['03-sales-people']; }
    else {html=box('질문','회사의 사업 구조와 전환 방향을 전체적으로 요약해줘')+table(['강사 작성 리포트','부분 답변 (Map)'],[['사업 구조','전원장치 60% · ESS 25% · 유지보수 15% (2025년 기준)'],['제품 전환','PD-2000 단종 결정과 PD-3000 전환'],['ESS 성장','ESS-500 개발과 아틀라스 공급 사업']])+box('종합 예시 (Reduce)','퍼플델타는 전원장치·ESS·유지보수의 세 축을 운영하며, PD-3000 전환과 ESS 성장을 추진합니다. 출시·공급 목표를 완료된 사실로 바꾸지 않습니다.');sources=['01-company-overview','04-product-pd2000','06-product-ess500','09-project-atlas'];}
  } else if (id === '12') {
    const examples={
      single:{q:'PD-3000의 최대 출력은?',answer:'650W. [05-product-pd3000]',rows:[['기본 RAG','제품 원문 한 개로 충분할 수 있음'],['Local','제품 개체·속성 주변을 확인'],['Global','넓은 요약은 이 단순 질문에 불필요할 수 있음']],src:['05-product-pd3000']},
      multi:{q:'김철수가 담당하는 제품의 생산 공장은?',answer:'김철수 → PD-2000 → 제조1공장(충남 천안).',rows:[['기본 RAG','담당·생산 문서가 함께 검색되는지 확인'],['Local','담당 → 제품 → 공장 관계를 연결'],['Global','리포트에 세부 관계가 빠졌는지 확인']],src:['03-sales-people','04-product-pd2000','07-factories']},
      global:{q:'회사의 전체 사업 구조는?',answer:'전원장치·ESS·유지보수의 세 축. PD-3000 전환과 ESS 성장을 추진.',rows:[['기본 RAG','여러 사업 관련 원문이 충분히 포함되는지 확인'],['Local','한 개체 주변만으로 사업 전체가 보이는지 확인'],['Global','여러 리포트에서 부분 답변을 모아 종합']],src:['01-company-overview','04-product-pd2000','06-product-ess500','09-project-atlas']}
    };
    const e=examples[key];html=box('비교 질문',e.q)+table(['경로','확인할 근거 범위'],e.rows)+box('원문으로 확인할 정답 예시',e.answer)+box('선택 기준','같은 질문과 근거로 비교하세요. 이 표는 경로를 설명하며, 모델 실행의 정답률·지연시간을 측정한 결과는 아닙니다.');sources=e.src;
  }
  return {html,sources,label:labs[id].label,config:key};
}

export const codeCards = {
  '01':['question','context','model','temperature'], '02':['fit','query','encode','decode'],
  '03':['where','delete','ids','update'], '04':['chunk_size','step','overlap','len(text)'],
  '05':['question','chunks','client','context'], '06':['1 / (k + rank)','1 / k + rank','k / rank','rank / k'],
  '07':['reverse=False','ascending=True','reverse=True','sort=False'], '08':['history','question','standalone_query','answer'],
  '09':['rank / 1','1 / rank','rank','1 / (rank + 1)'], '10':['1','2','3','0'],
  '11':['G','chunks','question','reports'], '12':['global_search','local_search','G','client']
};
