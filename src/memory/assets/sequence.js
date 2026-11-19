(function(){
var gd=document.getElementById('memory-map');
var turns=JSON.parse(document.getElementById('turns-data').textContent);
if(!turns.length)return;
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
var COL={faithful:'#2dd4bf',stored_corruption:'#ff9f43',noisy:'#ef4444'};
function hitColor(n){return !n?COL.faithful:(n==='stored_corruption'?COL.stored_corruption:COL.noisy);}
var NOISE={swap:'a neighbouring memory surfaced instead',substitution:'details were altered at retrieval',
 'swap+substitution':'wrong memory AND altered details',stored_corruption:'the stored record was already degraded'};
var STEPS=[
 {t:'Trainee utterance',mod:'cli/app.py -> PatientEngine.respond()',
  why:'The trainee says something. It is the cue for everything that follows.',
  d:function(T){return '<div class="quote">'+esc(T.trainee)+'</div><p>Turn '+T.n+'. The cue is embedded into the same vector space as the stored memories (green/cyan star on the map).</p>';}},
 {t:'Trigger assessment',mod:'behavior/distress.py -> evaluate_trigger(), DistressTracker',
  why:'Lexical heuristics check for sensitive topics, pressuring language or redirection, reweight a Markov transition matrix, and sample the next emotional state.',
  d:function(T){return '<div class="kv"><span class="on">distress state: '+esc(T.distress)+'</span></div><p>The state machine moves between calm, confused, agitated and withdrawn. The transition matrix is an assumption anchored to prevalence ranges in Steinberg et al. 2008.</p>';}},
 {t:'Symptom sampling',mod:'behavior/symptom_sampler.py -> sample_constraints()',
  why:'Per-turn behavioural constraints are drawn from the statistical priors and modulated by the distress state.',
  d:function(T){var c=T.constraints||{};function f(k,l){return '<span class="'+(c[k]?'on':'')+'">'+l+': '+(c[k]?'yes':'no')+'</span>';}
   return '<div class="kv"><span class="on">max words: '+esc(c.max_words)+'</span>'+f('memory_fog','memory fog')+f('factual_inversion','factual inversion')+f('anomia','anomia')+f('distress_escalation','distress escalation')+'<span class="'+(c.nonverbal_cue?'on':'')+'">non-verbal: '+esc(c.nonverbal_cue||'none')+'</span></div><p>Stage 7 caps speech at 5 words per turn (FAST 7a: fewer than six intelligible words a day, Mitchell 2015).</p>';}},
 {t:'Memory recall',mod:'memory/recall.py -> MemoryRecall.recall()',
  why:'Cue noise is added, Milvus returns nearest memories, scores are attenuated by era (Ribot gradient), time decay and rehearsal, then a swap or substitution may corrupt the top hit.',
  d:function(T){if(T.constraints&&T.constraints.memory_fog)return '<p><b>Memory fog:</b> total retrieval failure this turn, so recall is skipped and nothing is injected into the prompt.</p>';
   if(!T.hits.length)return '<p>Nothing surfaced.</p>';
   var r=T.hits.map(function(h){var n=h.noise;return '<tr><td><span class="dot" style="background:'+hitColor(n)+'"></span>'+esc(h.coll==='episodic_memory'?'episode':'fact')+'</td><td>'+esc(h.text)+'</td><td>'+Number(h.score).toFixed(3)+'</td><td>'+(n?esc(n)+': '+esc(NOISE[n]||''):'faithful')+'</td></tr>';}).join('');
   return '<p>Highlighted on the map (rings): teal = faithful, amber = stored record already degraded, red = noise injected at retrieval.</p><figure><table><thead><tr><th>Kind</th><th>Surfaced text</th><th>Score</th><th>Reliability</th></tr></thead><tbody>'+r+'</tbody></table></figure>';}},
 {t:'Prompt assembly',mod:'core/patient_engine.py -> build_system_prompt()',
  why:'Persona, family, distress state, constraint rules and the degraded memories are composed into the system prompt. The model sees degraded text, never the clean original.',
  d:function(T){return '<p>Hazy or uncertain memories are tagged as such so the model conveys them loosely or wrongly instead of quoting them. '+(T.hits.length?T.hits.length+' fragment(s) injected.':'No fragments injected.')+'</p>';}},
 {t:'LLM generation',mod:'core/llm_providers.py -> generate()',
  why:'The configured backend (Ollama, Gemini, Azure OpenAI or the offline mock) role-plays the patient under those constraints.',
  d:function(T){return '<div class="kv"><span class="on">latency: '+Number(T.latency_ms||0).toFixed(0)+' ms</span></div><p>Only the last few conversation messages are sent as short-term context.</p>';}},
 {t:'Post-processing',mod:'core/patient_engine.py -> _postprocess()',
  why:'A hard word cap and the required non-verbal cue are enforced after generation, so the sampled constraints always hold.',
  d:function(T){return '<p>Patient response (max '+esc((T.constraints||{}).max_words)+' words):</p><div class="quote">'+esc(T.patient)+'</div>';}},
 {t:'Memory encoding',mod:'memory/encoding.py -> encode_episode()',
  why:'What the trainee said is stored as a new episodic memory, unless encoding fails (80% at stage 7), and what is stored may already be garbled.',
  d:function(T){var e=T.encoded;return '<p>'+(e===true?'Encoded as a new (possibly corrupted) episodic memory.':(e===false?'Encoding failed or was skipped: the patient did not retain this turn.':'Not recorded in this transcript.'))+'</p>';}}
];
var turn=0,step=0,timer=null;
var idx={};gd.data.forEach(function(t,i){idx[t.name]=i;});
function setTrace(name,upd){if(idx[name]!=null)Plotly.restyle(gd,upd,[idx[name]]);}
function draw(){
 var T=turns[turn],show=step>=3&&!(T.constraints&&T.constraints.memory_fog);
 var lx=[],ly=[],hx=[],hy=[],hc=[];
 if(show)T.hits.forEach(function(h){lx.push(T.q[0],h.x,null);ly.push(T.q[1],h.y,null);hx.push(h.x);hy.push(h.y);hc.push(hitColor(h.noise));});
 setTrace('Sequence recall',{x:[lx],y:[ly]});
 setTrace('Sequence hits',{x:[hx],y:[hy],'marker.color':[hc]});
 setTrace('Sequence cue',{x:[[T.q[0]]],y:[[T.q[1]]],text:[['Turn '+T.n]]});
 document.querySelectorAll('.step').forEach(function(b,i){b.classList.toggle('active',i===step);b.classList.toggle('done',i<step);});
 document.querySelectorAll('.pill').forEach(function(b,i){b.setAttribute('aria-pressed',i===turn);});
 var S=STEPS[step];
 document.getElementById('seq-detail').innerHTML='<div class="mod">'+esc(S.mod)+'</div><h3>'+(step+1)+'. '+esc(S.t)+'</h3><p>'+esc(S.why)+'</p>'+S.d(T);
}
function next(){if(step<STEPS.length-1){step++;}else if(turn<turns.length-1){turn++;step=0;}else{stop();return;}draw();}
function prev(){if(step>0){step--;}else if(turn>0){turn--;step=STEPS.length-1;}draw();}
function stop(){if(timer){clearInterval(timer);timer=null;}var b=document.getElementById('seq-play');if(b)b.textContent='Play';}
function play(){if(timer){stop();return;}document.getElementById('seq-play').textContent='Pause';timer=setInterval(next,2600);}
document.querySelectorAll('.step').forEach(function(b,i){b.addEventListener('click',function(){stop();step=i;draw();});});
document.querySelectorAll('.pill').forEach(function(b,i){b.addEventListener('click',function(){stop();turn=i;step=0;draw();});});
document.getElementById('seq-next').addEventListener('click',function(){stop();next();});
document.getElementById('seq-prev').addEventListener('click',function(){stop();prev();});
document.getElementById('seq-play').addEventListener('click',play);
document.addEventListener('keydown',function(e){if(e.key==='ArrowRight'){stop();next();}else if(e.key==='ArrowLeft'){stop();prev();}});
draw();
})();
