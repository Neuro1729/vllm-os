'use strict';

// All request timings and responses originate from the configured vLLM server.
// No fake answers or artificial waiting are introduced by this UI.
const question = 'What is 2 + 2?';
const scenarios = [
  {title: 'All short requests', subtitle: 'Baseline · Four lightweight questions', chats: [
    makeChat(1, 'Say hello.', 'short'),
    makeChat(2, 'What is 5 + 3?', 'short'),
    makeChat(3, 'Name a planet.', 'short'),
    makeChat(4, question, 'short', 2)
  ]},
  {title: 'Long + short requests', subtitle: 'Contention · Three long prefills before the same tiny question', chats: [
    makeChat(1, 'Summarize this large technical report.', 'long'),
    makeChat(2, 'Explain the key findings in this research paper.', 'long'),
    makeChat(3, 'Analyze the following extensive technical document.', 'long'),
    makeChat(4, question, 'short', 2)
  ]}
];
let active = 0;
let record = [];
let wl = {state:'idle'};          // latest /api/workload/status
let wlError = '';                 // start/stop error shown in the panel
let wlPolling = false;
const wlRunning = () => wl.state === 'running' || wl.state === 'stopping';
const $ = id => document.getElementById(id);
const format = n => n == null ? '—' : `${(n / 1000).toFixed(n < 1000 ? 3 : 2)}s`;
function makeChat(id, prompt, kind, priority=0) {
  return {id, prompt, instruction:prompt, kind, priority, genTokens:null, genBusy:false, genFailAt:0, edited:false, status:'ready', reply:'', ttft:null, total:null,
    submittedAt:null, startedAt:null, pending:false, completed:0, ctrl:null,
    lastPromptTokens:null, error:null, runId:0};
}
function escapeHTML(x) {
  return String(x).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function changeSlide(next) {
  active = (next + scenarios.length) % scenarios.length;
  render();ensureLongPrompts();
}
function render() {
  const scenario = scenarios[active];
  $('experimentCount').textContent = `EXPERIMENT ${active+1} OF 2`;
  $('experimentTitle').textContent = scenario.title;
  $('experimentSubtitle').textContent = scenario.subtitle;
  $('scenarioPill').textContent = active === 0 ? 'BASELINE' : 'CONTENTION';
  $('longTokens').disabled = active === 0;
  $('repeat').disabled = active === 0;
  $('repeat').checked = wlRunning();
  $('bgWrap').style.display = active === 0 ? 'none' : '';
  $('repeat').parentElement.style.display = active === 0 ? 'none' : '';
  $('workload').hidden = active === 0;
  document.querySelectorAll('.indicator').forEach((button, i) => button.classList.toggle('selected', i===active));
  $('chats').innerHTML = scenario.chats.map(chat => `
    <article class="chat-card ${chat.id===4?'highlight':''}" id="chat-${chat.id}">
      <div class="card-head"><b>Chat ${chat.id}</b><div><span class="card-label">P${chat.priority} · </span><span class="tag ${chat.kind}" id="tag-${chat.id}">${chat.kind==='long'?'Long prefill':'Short prompt'}</span></div></div>
      <div id="view-${chat.id}" class="chat-view empty">Ready to send</div>
      <div class="composer"><textarea id="prompt-${chat.id}" class="${chat.kind==='long'?'long-prompt':''}" aria-label="Prompt for chat ${chat.id}" placeholder="Type a message...">${escapeHTML(chat.prompt)}</textarea><button class="send" id="send-${chat.id}" ${chat.pending?'disabled':''}>${chat.pending?'Sending…':'Send ↗'}</button></div>
      <div class="metrics"><div class="metric"><span>TTFT</span><strong id="ttft-${chat.id}">—</strong></div><div class="metric"><span>Completion</span><strong id="total-${chat.id}">—</strong></div><div id="status-${chat.id}" class="status">Ready</div></div>
    </article>`).join('');
  scenario.chats.forEach(chat => {
    $(`prompt-${chat.id}`).addEventListener('input', ev => {
      chat.prompt = ev.target.value;
      if(chat.kind==='long')chat.edited=true;
      if(chat.id===4) scenarios[1-active].chats.find(c=>c.id===4).prompt=chat.prompt;
    });
    $(`send-${chat.id}`).addEventListener('click', () => sendChat(active, chat.id));
    updateCard(chat);
  });
  compare();renderWorkload();
}
function updateCard(chat) {
  const elem = $( `chat-${chat.id}` );
  if (!elem || !scenarios[active].chats.includes(chat)) return;
  const view=$(`view-${chat.id}`);
  const tag=$(`tag-${chat.id}`);
  if(tag) tag.textContent=chat.lastPromptTokens!=null?`${chat.lastPromptTokens.toLocaleString()} tokens`:(chat.kind==='long'?`~${Number($('longTokens').value).toLocaleString()} tokens`:'Short prompt');
  if(chat.status==='ready') {
    view.className='chat-view empty'; view.textContent='Ready to send';
  } else {
    view.className='chat-view';
    const shown = chat.prompt.length>300 ? chat.prompt.slice(0,220)+` … [${chat.prompt.length.toLocaleString()} characters; scroll the text box below to read it all]` : chat.prompt;
    let output = `<div class="question">${escapeHTML(shown)}</div>`;
    if(chat.reply) output += `<div class="response">${escapeHTML(chat.reply)}</div>`;
    else if(chat.error) output += `<div class="error">${escapeHTML(chat.error)}</div>`;
    else if(chat.pending) output += `<div class="waiting">${escapeHTML(chat.status)}</div>`;
    view.innerHTML = output;
    view.scrollTop=view.scrollHeight;
  }
  const elapsed = chat.startedAt == null ? 0 : performance.now()-chat.startedAt;
  $(`ttft-${chat.id}`).textContent = chat.ttft==null ? (chat.startedAt!=null && chat.pending?`Waiting ${format(elapsed)}`:'—') : format(chat.ttft);
  $(`ttft-${chat.id}`).className = chat.ttft==null && chat.pending?'waiting':chat.ttft!=null?'complete':'';
  $(`total-${chat.id}`).textContent = chat.total!=null ? format(chat.total) : '—';
  $(`status-${chat.id}`).textContent = chat.error?'Error':chat.pending?'In progress':chat.status==='done'?`Done${chat.completed>1?' · '+chat.completed+' runs':''}`:'Ready';
  const button=$(`send-${chat.id}`);
  button.disabled=chat.pending;
  button.textContent=chat.pending?'Sending…':'Send ↗';
  button.classList.toggle('busy',chat.pending);
}
function compare() {
  for(const [index,id] of [[0,'baselineTTFT'],[1,'heavyTTFT']]){
    const chat=scenarios[index].chats.find(c=>c.id===4);
    $(id).textContent=chat.ttft!=null?format(chat.ttft):chat.pending?'Waiting…':'—';
  }
}
async function sendChat(scenarioIndex, id) {
  const chat=scenarios[scenarioIndex].chats.find(c=>c.id===id);
  if(!chat || chat.pending || !chat.prompt.trim())return;
  chat.pending=true;chat.status='Preparing prompt…';chat.reply='';chat.error=null;
  chat.ttft=null;chat.total=null;chat.startedAt=null;chat.submittedAt=Date.now();
  const runId=++chat.runId;
  const controller = new AbortController();chat.ctrl=controller;
  const row={scenario:scenarioIndex===0?'all-short':'long-plus-short',chat:id,
    kind:chat.kind,priority:chat.priority,submitted_at:new Date().toISOString(),
    target_tokens:chat.kind==='long'?Number($('longTokens').value):null,
    actual_prompt_tokens:null,ttft_ms:null,completion_ms:null,status:'started'};
  record.push(row);
  updateCard(chat);compare();
  try {
    const res=await fetch('/api/chat',{method:'POST',signal:controller.signal,
      headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:chat.prompt,
      kind:chat.kind,exact:chat.kind==='long'&&chat.prompt.length>2000,priority:chat.priority,target_tokens:row.target_tokens||4096,
      seed:crypto.randomUUID ? crypto.randomUUID() : String(Math.random()),max_output_tokens:32})});
    if(!res.ok)throw Error(`UI server returned HTTP ${res.status}`);
    const reader=res.body.getReader();const decoder=new TextDecoder();let pending='';
    while(true){
      const {value,done}=await reader.read();if(done)break;
      pending+=decoder.decode(value,{stream:true});
      let newline;
      while((newline=pending.indexOf('\n'))>=0){
        const raw=pending.slice(0,newline);pending=pending.slice(newline+1);if(!raw.trim())continue;
        let msg;try{msg=JSON.parse(raw)}catch{continue}
        if(chat.runId!==runId)break;
        if(msg.event==='prepared'){
          chat.lastPromptTokens=msg.prompt_tokens;
          row.actual_prompt_tokens=msg.prompt_tokens;
          chat.status='Submitting to vLLM…';
        }
        else if(msg.event==='started'){chat.startedAt=performance.now();chat.status='Waiting for first token…'}
        else if(msg.event==='token'){
          if(chat.ttft==null)chat.ttft=msg.ttft_ms;
          chat.reply+=msg.text;
          chat.status='Receiving response…';
          row.ttft_ms=chat.ttft;
        }
        else if(msg.event==='done'){
          chat.pending=false;chat.status='done';chat.completed++;
          chat.total=msg.total_ms;chat.ttft=msg.ttft_ms;
          row.status='done';row.completion_ms=msg.total_ms;row.ttft_ms=msg.ttft_ms;
          row.actual_prompt_tokens=msg.prompt_tokens;
        }
        else if(msg.event==='error'){
          chat.pending=false;chat.error=msg.message;chat.status='error';row.status='error';row.error=msg.message;
        }
        updateCard(chat);compare();
      }
    }
    if(chat.pending){
      chat.pending=false;chat.status='error';chat.error='Stream closed before completion';
      row.status='error';row.error=chat.error;
    }
  }catch(err){
    if(chat.runId!==runId){row.status='cancelled';return}
    chat.pending=false;chat.status='error';chat.error=err.name==='AbortError'?'Request cancelled':String(err.message||err);
    row.status='error';row.error=chat.error;
  }finally{
    if(chat.runId!==runId)return;
    chat.ctrl=null;updateCard(chat);compare();
  }
}
function addChat() {
  const scenario=scenarios[active];
  if(scenario.chats.length>=12){alert('Maximum 12 chats per experiment.');return}
  const id=1+Math.max(...scenario.chats.map(c=>c.id));
  scenario.chats.push(makeChat(id,active===0?'Say hello.':'Summarize the technical report.',active===0?'short':'long'));
  render();ensureLongPrompts();
}
function resetScenario(){
  const s=scenarios[active];
  s.chats.forEach(c=>{if(c.ctrl)c.ctrl.abort();c.runId++;c.pending=false;c.status='ready';c.reply='';c.error=null;c.ttft=null;c.total=null;c.startedAt=null;c.completed=0});
  stopWorkload();render();
}
function exportResults(){
  const payload={tool:'vLLM Scheduler Playground',exported_at:new Date().toISOString(),
    vllm_url:$('connectionText').dataset.url||'',model:$('connectionText').textContent,background_workload:wl,measurements:record};
  const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
  a.href=url;a.download='vllm-playground-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function status(){
  try{
    const result=await (await fetch('/api/health',{cache:'no-store'})).json();
    const box=$('connection');box.className='connection '+(result.online?'online':'offline');
    $('connectionText').textContent=result.online?result.model:'vLLM disconnected';
    $('connectionText').title=result.online?result.url:result.message;
    $('connectionText').dataset.url=result.url;
  }catch(e){$('connection').className='connection offline';$('connectionText').textContent='UI backend unavailable'}
}
$('prev').addEventListener('click',()=>changeSlide(active-1));
$('next').addEventListener('click',()=>changeSlide(active+1));
document.querySelectorAll('.indicator').forEach(btn=>btn.addEventListener('click',()=>changeSlide(Number(btn.dataset.slide))));
$('addChat').addEventListener('click',addChat);$('reset').addEventListener('click',resetScenario);$('export').addEventListener('click',exportResults);
$('repeat').addEventListener('change',ev=>{ev.target.checked?startWorkload():stopWorkload()});
setInterval(()=>{
  scenarios[active].chats.filter(c=>c.pending&&c.startedAt!=null).forEach(updateCard);
  compare();renderWorkload();
},200);
setInterval(status,10000);
setInterval(()=>{if(active===1&&!wlRunning())refreshWorkload().then(renderWorkload)},2000);

// ---- Real long prompts shown (and editable) in the long chats' text boxes ----
// The backend generates a unique document whose token length is verified by vLLM; you can scroll
// the text box to read it, edit it, and Send submits exactly that text.
async function ensureLongPrompts(){
  const target=Number($('longTokens').value);
  for(const chat of scenarios[1].chats){
    if(chat.kind!=='long'||chat.genBusy||chat.pending||chat.genTokens===target)continue;
    if(chat.edited&&chat.genTokens!=null)continue;                    // keep the user's own edits
    if(Date.now()-chat.genFailAt<5000)continue;
    chat.genBusy=true;
    try{
      const res=await fetch('/api/long-prompt',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({prompt:chat.instruction,target_tokens:target})});
      if(!res.ok){let d='';try{d=(await res.json()).detail}catch{}throw Error(d||`HTTP ${res.status}`)}
      const doc=await res.json();
      if(Number($('longTokens').value)!==target||chat.pending)continue; // size changed meanwhile
      chat.prompt=doc.prompt;chat.genTokens=target;chat.lastPromptTokens=doc.prompt_tokens;chat.edited=false;
      const box=$(`prompt-${chat.id}`);if(box&&scenarios[active].chats.includes(chat))box.value=chat.prompt;
      updateCard(chat);
    }catch(err){chat.genFailAt=Date.now();chat.genError=String(err.message||err)}
    finally{chat.genBusy=false}
  }
}
$('longTokens').addEventListener('change',()=>{ensureLongPrompts()});

// ---- Background P0 workload (server-side; the UI only starts/stops/observes it) ----
async function startWorkload(){
  wlError='';
  try{
    const res=await fetch('/api/workload/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({concurrency:Number($('bgConcurrency').value),target_tokens:Number($('longTokens').value)})});
    if(!res.ok){
      let detail='';try{detail=(await res.json()).detail}catch{}
      throw Error(detail||`UI server returned HTTP ${res.status}`);
    }
    wl={state:'running',concurrency:Number($('bgConcurrency').value)};
    pollWorkload();
  }catch(err){wlError=String(err.message||err);$('repeat').checked=false}
  renderWorkload();
}
async function stopWorkload(){
  try{await fetch('/api/workload/stop',{method:'POST'})}catch{}
  await refreshWorkload();renderWorkload();
}
async function refreshWorkload(){
  try{wl=await (await fetch('/api/workload/status',{cache:'no-store'})).json()}catch{}
}
async function pollWorkload(){
  if(wlPolling)return;wlPolling=true;
  try{while(wlRunning()){await refreshWorkload();renderWorkload();await new Promise(r=>setTimeout(r,500))}}
  finally{wlPolling=false;$('repeat').checked=wlRunning();renderWorkload()}
}
function renderWorkload(){
  if(active!==1)return;
  const running=wlRunning(),sch=wl.scheduler||{};
  $('repeat').checked=running;$('bgConcurrency').disabled=running;$('longTokens').disabled=running;
  const st=$('wlState');st.textContent=wl.state==='error'?'ERROR':(wl.state||'idle').toUpperCase();st.className='wl-state '+(wl.state||'');
  $('wlNote').textContent=running?`${wl.concurrency} × ~${(wl.target_tokens||0).toLocaleString()}-token prompts · priority 0 · stops automatically at ${wl.duration_s||60}s`
    :wl.state==='finished'?`Finished: ${wl.reason}`:'Tick “Keep long chats busy” to start (60 s max).';
  const num=n=>n==null?'—':n.toLocaleString();
  $('wlInflight').textContent=running||wl.state==='finished'||wl.state==='error'?`${num(wl.in_flight)} / ${num(wl.concurrency)}`:'—';
  $('wlDone').textContent=wl.completed==null?'—':`${num(wl.completed)}${wl.errors?` · ${wl.errors} err`:''}`;
  $('wlAwait').textContent=num(wl.awaiting_first_token);
  const hot=(sch.waiting||0)>0;
  $('wlQueue').textContent=sch.available?`${Math.round(sch.running||0)} / ${Math.round(sch.waiting||0)}`:'n/a';
  $('wlQueue').className=hot?'hot':'';
  $('wlKv').textContent=sch.kv_usage!=null?`${(sch.kv_usage*100).toFixed(1)}%`:'—';
  $('wlTime').textContent=running?`${(wl.remaining_s||0).toFixed(0)}s`:'—';
  const c4=scenarios[1].chats.find(c=>c.id===4);
  let line='<b>Chat 4 (P2, “'+escapeHTML(c4.prompt.slice(0,40))+'”)</b> · ';
  if(c4.ttft!=null)line+=`<span class="ok">TTFT ${format(c4.ttft)}</span> · completion ${c4.total!=null?format(c4.total):'…'}`;
  else if(c4.pending&&c4.startedAt!=null)line+=`<span class="wait">waiting ${format(performance.now()-c4.startedAt)} for first token…</span>`;
  else if(c4.error)line+=`<span class="wait">${escapeHTML(c4.error)}</span>`;
  else line+='not sent yet';
  if(sch.available){line+=hot?` <span class="contend">CONTENTION: ${Math.round(sch.waiting)} WAITING IN vLLM</span>`
    :(running?' <span class="contend none">NO vLLM QUEUE</span>':'')}
  $('wlChat4').innerHTML=line;
  const d=[];
  if(wlError)d.push(`<span class="err">${escapeHTML(wlError)}</span>`);
  if(wl.state==='error'&&wl.message)d.push(`<span class="err">${escapeHTML(wl.message)}</span>`);
  else if(wl.last_error)d.push(`<span class="err">last error: ${escapeHTML(wl.last_error)}</span>`);
  if(wl.prompt_tokens&&wl.prompt_tokens.server_min!=null)
    d.push(`Prompt tokens (vLLM-reported): ${num(wl.prompt_tokens.server_min)}–${num(wl.prompt_tokens.server_max)} · ${num(wl.verified_prompts)} verified, ${wl.token_mismatches} off-target · unique per request`);
  if(wl.ttft_ms&&wl.ttft_ms.p50!=null)
    d.push(`Background TTFT p50 ${format(wl.ttft_ms.p50)} · p95 ${format(wl.ttft_ms.p95)} · prefill ≈ ${num(wl.prefill_tokens_per_s)} tok/s · replacement gap avg ${wl.gap_ms.avg??'—'} ms`);
  if(sch.available&&wl.scheduler&&wl.scheduler.samples)
    d.push(`vLLM queue: peak ${Math.round(wl.scheduler.max_waiting_seen)} waiting / ${Math.round(wl.scheduler.max_running_seen)} running · waiting seen in ${wl.scheduler.waiting_samples}/${wl.scheduler.samples} samples · peak KV ${(wl.scheduler.max_kv_usage_seen*100).toFixed(1)}%`);
  if(!sch.available)d.push('vLLM /metrics not reachable: scheduler queue indicator unavailable (latency measurements are still real).');
  $('wlDetail').innerHTML=d.join('<br>');
}
refreshWorkload().then(()=>{if(wlRunning())pollWorkload();renderWorkload()});
ensureLongPrompts();
setInterval(()=>{if(active===1)ensureLongPrompts()},3000);   // retry if vLLM was briefly unreachable
let startX=null;
$('chats').addEventListener('touchstart',e=>{startX=e.touches[0]?.clientX??null},{passive:true});
$('chats').addEventListener('touchend',e=>{
  if(startX===null)return;
  if(e.target.closest('textarea,button')){startX=null;return}
  const dx=(e.changedTouches[0]?.clientX??startX)-startX;
  if(Math.abs(dx)>100)changeSlide(active+(dx<0?1:-1));startX=null;
},{passive:true});
window.addEventListener('keydown',e=>{if(/TEXTAREA|INPUT|SELECT/.test(document.activeElement?.tagName||''))return;
  if(e.key==='ArrowLeft')changeSlide(active-1);if(e.key==='ArrowRight')changeSlide(active+1)});
render();status();
