import {icons, levelOf, describeAction, makeFrames, findLoop, verifyReplayShape, objectiveReward, objectiveRemaining, fishAbility} from './model.js';

const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
const probability = n => n > 0 && n < .001 ? '<0.1%' : `${(n*100).toFixed(1)}%`;
let manifest;
let dataRoot = 'data/round5/';
const lanes = [];
const cache = new Map();

async function getReplay(entry) {
  if (!cache.has(entry.id)) {
    cache.set(entry.id, fetch(`${dataRoot}${entry.file}`).then(response => {
      if (!response.ok) throw new Error('Could not load replay');
      return response.json();
    }).then(replay => {verifyReplayShape(replay); return replay;}));
  }
  return cache.get(entry.id);
}

function abilityText(pet, catalogId) {
  const data = manifest.pets[pet.spec_id];
  const lvl = levelOf(pet);
  const name = data?.name || pet.spec_id;
  const effects = {
    fish: fishAbility(lvl, catalogId),
    otter:`Buy: give ${lvl} random friend${lvl > 1 ? 's' : ''} +1 health.`,
    ant:`Faint: give one random friend +${lvl}/+${lvl}.`,
    cricket:`Faint: summon a ${lvl}/${lvl} Zombie Cricket.`,
    horse:`Friend summoned: give it +${lvl} temporary attack. Must still be alive in battle.`,
    mosquito:`Start of battle: deal 1 damage to ${lvl} random enem${lvl > 1 ? 'ies' : 'y'}.`,
    pig:`Sell: gain ${lvl} extra gold, in addition to the ${lvl} base sell value.`,
    pigeon:`Sell: stock ${lvl} free Bread Crumbs (+1 attack each).`,
    bee:'Summoned by Honey. No ability.', zombie_cricket:'Summoned by Cricket. No ability.',
  };
  return `${name} · level ${lvl}, ${pet.experience}/6 XP. ${effects[pet.spec_id] || ''}${pet.perk === 'honey' ? ' Honey: summon a 1/1 Bee on faint.' : ''}`;
}

function petCard(pet, slot, highlighted = false, zone = 'team') {
  if (!pet) return `<div class="pet empty"><span class="slot">${slot}</span>Empty</div>`;
  const isShop = zone === 'shop';
  const id = isShop ? pet.item_id : pet.spec_id;
  const meta = manifest.pets[id];
  const name = meta?.name || id.replaceAll('_',' ');
  const isFood = isShop && pet.kind === 'food';
  const attack = isShop ? meta?.attack : pet.attack;
  const health = isShop ? meta?.health : pet.health;
  const temporary = !isShop && (pet.temporary_attack || pet.temporary_health);
  const fullLabel = isShop ? `Shop ${slot}: ${name}, ${pet.cost} gold${pet.frozen ? ', frozen':''}` : `Slot ${slot}: ${name}, ${attack} attack, ${health} health, level ${levelOf(pet)}`;
  return `<button class="pet ${highlighted ? 'highlight':''} ${pet.frozen ? 'frozen':''} ${!isShop && health <= 0 ? 'dead':''}" data-pet="${zone}:${slot}" aria-label="${escapeHtml(fullLabel)}">
    <span class="slot">${slot}</span><span class="emoji" aria-hidden="true">${icons[id] || '•'}</span><span class="pet-name">${escapeHtml(name)}</span>
    ${!isFood ? `<span class="stats"><span class="attack" aria-label="attack">⚔ ${attack}</span><span class="health" aria-label="health">♥ ${health}</span></span>` : `<span class="stats">${id === 'honey' ? 'Bee on faint' : id === 'apple' ? '+1 / +1' : '+1 attack'}</span>`}
    <span class="pet-meta">${isShop ? `${pet.cost} gold${pet.frozen ? ' ❄':''}` : `Lv ${levelOf(pet)} · ${pet.experience}/6 XP${pet.perk === 'honey' ? ' 🍯':''}`}</span>
    <span class="temporary">${temporary ? `+${pet.temporary_attack}/+${pet.temporary_health} temp` : '&nbsp;'}</span>
  </button>`;
}

function teamMarkup(team, highlights = [], zone = 'team') {
  return Array.from({length:5}, (_, i) => petCard(team[i], i, highlights.includes(i), zone)).join('');
}

function chart(replay, index, width) {
  const height = 138, left = 35, right = 10, top = 14, bottom = 25;
  const all = replay.steps.flatMap(s => [s.value,objectiveRemaining(s)]);
  const low = Math.floor(Math.min(0,...all)), high = Math.ceil(Math.max(1,...all));
  const x = i => left+i/Math.max(1,replay.steps.length-1)*(width-left-right);
  const y = v => height-bottom-(v-low)/(high-low)*(height-top-bottom);
  const line = key => replay.steps.map((s,i) => `${i?'L':'M'}${x(i).toFixed(1)},${y(key === 'remaining_return' ? objectiveRemaining(s) : s[key]).toFixed(1)}`).join(' ');
  const ticks = [...new Set([low, Math.round((low+high)/2), high])];
  return `<svg class="value-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Critic value and realized remaining training-objective reward across ${replay.steps.length} decisions">
    ${ticks.map(v => `<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="#e2e8ed"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end" fill="#546879" font-size="12">${v}</text>`).join('')}
    <path d="${line('remaining_return')}" fill="none" stroke="#889ba7" stroke-width="1.7" stroke-dasharray="4 3"/>
    <path d="${line('value')}" fill="none" stroke="#3464b6" stroke-width="2"/>
    <line x1="${x(index)}" y1="${top}" x2="${x(index)}" y2="${height-bottom}" stroke="#172a3a" opacity=".6"/>
    <circle cx="${x(index)}" cy="${y(replay.steps[index].value)}" r="3" fill="#3464b6"/>
    <text x="${left}" y="${height-5}" fill="#546879" font-size="12">1</text><text x="${width-right}" y="${height-5}" text-anchor="end" fill="#546879" font-size="12">${replay.steps.length}</text><text x="${width/2}" y="${height-5}" text-anchor="middle" fill="#546879" font-size="12">Shop decision</text>
  </svg>`;
}

class Lane {
  constructor(number, entry) {
    this.number = number; this.position = 0; this.timer = null; this.loadSerial = 0;
    this.element = document.createElement('article'); this.element.className = 'lane'; this.element.id = `lane-${number}`;
    this.element.innerHTML = `<header class="lane-header"><div class="lane-heading"><h2>${number === 0 ? 'A · Failure inspection':'B · Comparison'}</h2><span class="badge outcome-badge"></span></div><label class="sr-label" for="replay-${number}" hidden>Choose replay ${number+1}</label><select class="replay-picker" id="replay-${number}" aria-label="Choose replay ${number+1}">${['cutoff','success'].map(category => `<optgroup label="${category === 'cutoff' ? 'Cutoff episodes':'Successful episodes'}">${manifest.episodes.filter(e=>e.category===category).map(e=>`<option value="${e.id}">${e.policy.replace('-seed',' · seed ')} / ${e.family} / #${e.seed}</option>`).join('')}</optgroup>`).join('')}</select></header><div class="lane-content"></div>`;
    $('#lanes').append(this.element);
    this.chartWidth = Math.max(220, this.element.clientWidth - 36);
    this.resizeObserver = new ResizeObserver(() => {
      const width = Math.max(220, this.element.clientWidth - 36);
      if (Math.abs(width - this.chartWidth) > 1) {this.chartWidth = width; this.render();}
    });
    this.resizeObserver.observe(this.element);
    $('.replay-picker',this.element).addEventListener('change',event=>this.load(event.target.value));
    this.element.addEventListener('click',event=>this.click(event));
    this.element.addEventListener('input',event=>{if(event.target.classList.contains('scrubber')) {this.pause();this.position=Number(event.target.value);this.render();}});
    this.load(entry.id, true);
  }
  async load(id, initial = false) {
    this.pause(); const serial = ++this.loadSerial;
    this.replay = null;
    $('.replay-picker',this.element).value = id;
    const content = $('.lane-content',this.element); content.textContent = 'Loading verified replay…';
    try {
      const entry = manifest.episodes.find(e=>e.id===id);
      const replay = await getReplay(entry);
      if(serial!==this.loadSerial) return;
      this.replay=replay; this.frames=makeFrames(replay); this.position=0;
      this.loop=findLoop(replay);
      if(initial) this.position=this.frames.findIndex(f=>f.kind==='decision' && f.index===Math.max(0,replay.steps.length-10));
      this.render();
    } catch(error) { if(serial===this.loadSerial) content.innerHTML=`<p class="error">${escapeHtml(error.message)}</p>`; }
  }
  pause() {if(this.timer) clearInterval(this.timer);this.timer=null;const button=$('[data-command="play"]',this.element);if(button)button.textContent='▶ Play';}
  play() {
    if(!this.replay) return;this.pause();
    if(this.position===this.frames.length-1)this.position=0;
    this.timer=setInterval(()=>{if(this.position>=this.frames.length-1){this.pause();return;}this.position++;this.render();},Number($('#speed').value));
    this.render();
  }
  jump(index) {this.pause();this.position=Math.max(0,Math.min(this.frames.length-1,index));this.render();}
  ending() {if(this.replay)this.jump(this.frames.findIndex(f=>f.kind==='decision' && f.index===Math.max(0,this.replay.steps.length-10)));}
  click(event) {
    const button=event.target.closest('button');if(!button || !this.replay)return;
    const command=button.dataset.command;
    if(command==='play')this.timer?this.pause():this.play();
    if(command==='prev')this.jump(this.position-1);
    if(command==='next')this.jump(this.position+1);
    if(command==='start')this.jump(0);
    if(command==='end')this.jump(this.frames.length-1);
    if(command==='loop')this.jump(this.frames.findIndex(f=>f.kind==='decision'&&f.index===this.loop));
    if(button.dataset.pet) {
      const [zone,index]=button.dataset.pet.split(':');
      const frame=this.frames[this.position],step=this.replay.steps[frame.index];
      const state=frame.kind==='terminal'?step.after:step.state;
      const teams=frame.kind==='battle'?step.battle[frame.battleIndex].teams:null;
      const pet=zone==='shop'?state.shop[index]:zone==='enemy'?teams?.[1][index]:(teams?.[0]||state.team)[index];
      if(!pet)return;
      const detail=$('.pet-detail',this.element);detail.hidden=false;
      detail.textContent=zone==='shop'?(pet.kind==='pet'?abilityText({spec_id:pet.item_id,experience:1},this.replay.catalog_id):`${pet.item_id==='apple'?'Apple: permanent +1 attack and +1 health.':pet.item_id==='honey'?'Honey: summon a 1/1 Bee on faint.':'Bread Crumbs: permanent +1 attack.'}`):abilityText(pet,this.replay.catalog_id);
    }
  }
  render() {
    if(!this.replay)return;
    const active = document.activeElement;
    const focusCommand = this.element.contains(active) ? active.dataset.command : null;
    const focusScrubber = this.element.contains(active) && active.classList.contains('scrubber');
    const openDetails = [...this.element.querySelectorAll('.lane-content details[open]')].map(el => el.className);
    const replay=this.replay,frame=this.frames[this.position],step=replay.steps[frame.index];
    const state=frame.kind==='terminal'?step.after:step.state;
    const battle=frame.kind==='battle'?step.battle[frame.battleIndex]:null;
    const finished=frame.kind==='terminal';
    const badge=$('.outcome-badge',this.element);badge.textContent=replay.category==='cutoff'?'Cutoff':'10-win success';badge.className=`badge outcome-badge ${replay.category==='success'?'success':''}`;
    const [kind,suffix='']=step.action.split(':'),args=suffix.split(',').map(Number);
    const teamHighlights=kind==='swap_adjacent'?args:kind==='sell'?[args[0]]:['merge','buy_food'].includes(kind)?[args[1]]:[];
    const shopHighlights=['buy_pet','buy_food','merge','toggle_freeze'].includes(kind)?[args[0]]:[];
    const status=finished?(step.truncated?`Episode stopped · ${step.reason==='turn_limit'?'turn limit':'shop action limit'}`:`Episode finished · ${state.wins} wins`):battle?battle.event:`Next: ${describeAction(step.action,state,manifest.pets)}`;
    const probs=step.legal_actions;
    const isCutoff=step.truncated&&!step.terminated;
    const bootstrapExplanation=step.bootstrap>0?'The positive bootstrap offsets some or all of the penalty.':step.bootstrap<0?'The negative bootstrap makes the reconstructed penalty more negative.':'No bootstrap adjustment is added here.';
    const controls=`<div class="playback"><button data-command="start" aria-label="Start replay">↤ Start</button><button data-command="prev" aria-label="Previous frame" ${this.position===0?'disabled':''}>←</button><button data-command="play">${this.timer?'❚❚ Pause':'▶ Play'}</button><button data-command="next" aria-label="Next frame" ${finished?'disabled':''}>→</button><button data-command="end" aria-label="End replay">End ↦</button>${this.loop!==null?'<button data-command="loop">Jump to loop</button>':''}<span class="frame-count">${this.position+1}/${this.frames.length}</span></div>`;
    const probRows=probs.slice(0,5).map(a=>`<div class="prob-row"><span class="prob-name ${a.id===step.action_id?'chosen':''}">${a.id===step.action_id?'→ ':''}${escapeHtml(describeAction(a.label,step.state,manifest.pets))}</span><span class="prob-value">${escapeHtml(probability(a.probability))}</span><div class="prob-track"><div class="prob-fill ${a.id===step.action_id?'chosen':''}" style="width:${a.probability*100}%"></div></div></div>`).join('');
    $('.lane-content',this.element).innerHTML=`<div class="lane-body">
      <div class="scoreboard"><span>Turn <b>${state.turn}</b></span><span>Gold <b>${state.gold}</b></span><span>Wins <b>${state.wins}</b>/10</span><span>Lives <b>${state.lives}</b>/5</span><span class="phase">${battle?'BATTLE':finished?'FINISHED':'SHOP'}</span></div>
      <div class="limit-row"><span>Shop actions used</span><span>${state.actions_this_turn}/30</span></div><div class="limit-track"><div class="limit-fill ${state.actions_this_turn>=24?'warning':''}" style="width:${Math.min(100,state.actions_this_turn/30*100)}%"></div></div>
      ${battle?`<div class="section-label"><span>Opponent</span><span>Front → back</span></div><div class="pet-row">${teamMarkup(battle.teams[1],[],'enemy')}</div>`:''}
      <div class="section-label"><span>Your team</span><span>Front → back</span></div><div class="pet-row">${teamMarkup(battle?battle.teams[0]:state.team,battle||finished?[]:teamHighlights)}</div>
      ${!battle?`<div class="section-label"><span>Shop</span><span>${finished?'Final state':'Highlighted slots belong to the next action'}</span></div><div class="pet-row">${teamMarkup(state.shop,finished?[]:shopHighlights,'shop')}</div>`:''}
      <div class="pet-detail" aria-live="polite" hidden></div>
      <div class="stage-message ${finished?replay.category:''}">${escapeHtml(status)}</div>
      ${controls}<input class="scrubber" aria-label="Replay ${this.number+1} timeline" type="range" min="0" max="${this.frames.length-1}" value="${this.position}">
      <div class="decision-info"><span>Decision ${frame.index+1}/${replay.steps.length}${battle?` · battle event ${frame.battleIndex+1}/${step.battle.length}`:''}</span><span>${escapeHtml(replay.policy)} · ${escapeHtml(replay.family)}</span></div>
    </div><div class="analysis">
      <div class="analysis-header"><h3>Policy at decision ${frame.index+1}</h3><span>Legal actions only</span></div>${probRows}
      <details class="all-actions"><summary>All ${probs.length} legal actions</summary><div>${probs.map(a=>`<div class="all-action-row"><span>${escapeHtml(describeAction(a.label,step.state,manifest.pets))}</span><span>${escapeHtml(probability(a.probability))}</span></div>`).join('')}</div></details>
      <div class="metric-grid"><div class="metric"><span>Critic before move</span><b>${step.value.toFixed(2)}</b></div><div class="metric"><span>Raw game reward</span><b>${signed(step.reward)}</b></div><div class="metric"><span>Objective reward left</span><b>${signed(objectiveRemaining(step))}</b></div></div>
      <p class="neutral-note">Training objective: game reward − ${replay.swap_cost || 0} per swap. This move: ${signed(objectiveReward(step))}. Raw game reward left: ${signed(step.remaining_return)}.</p>
      ${isCutoff?`<div class="bootstrap-note"><strong>Cutoff is a truncation, not a terminal loss</strong><div class="equation">${signed(objectiveReward(step))} + ${replay.gamma} × ${step.next_value.toFixed(2)} = ${signed(step.rollout_reward)}</div>Training-objective reward + discounted terminal-state value = reconstructed PPO rollout reward.<br>Critic after move: ${step.next_value.toFixed(2)}. One-step TD residual: ${signed(step.td_target-step.value)}.<br><strong>${bootstrapExplanation}</strong>This is a diagnostic at the selected checkpoint, not proof of the training-time cause.</div>`:`<p class="neutral-note">${step.terminated?'True terminal ending: no value is bootstrapped.':`Critic after move: ${step.next_value.toFixed(2)}. Next-state value enters return/advantage estimates.`} Critic values refer to this model’s training objective, not win probabilities.</p>`}
      <div class="chart-legend"><span>Critic V(s) · γ=${replay.gamma}</span><span>Objective reward left</span></div>${chart(replay,frame.index,this.chartWidth)}
      <details class="evidence"><summary>Replay evidence & limitations</summary><p>Seed ${replay.seed} · ${replay.actions} decisions · ${replay.wins} wins. Outcome verified against the saved test row.</p><p>Rules: <code>${escapeHtml(replay.catalog_id)}</code>. Model SHA: <code>${replay.model_sha256}</code>.</p><p>Battle frames are observed from the archived Python engine. Probabilities come from the selected policy. Realized reward left is one deterministic rollout, not the critic’s training target.</p></details>
    </div>`;
    openDetails.forEach(name => {const detail = $(`details.${name}`, this.element); if(detail) detail.open = true;});
    if(focusCommand) $(`[data-command="${focusCommand}"]`,this.element)?.focus({preventScroll:true});
    if(focusScrubber) $('.scrubber',this.element)?.focus({preventScroll:true});
  }
}

async function loadCollection(name) {
  $('#collection').disabled=true;
  $('#global-error').hidden=true;
  lanes.splice(0).forEach(l=>{l.pause();l.loadSerial++;l.resizeObserver.disconnect();});
  $('#lanes').replaceChildren();cache.clear();
  dataRoot=name==='round5'?'data/round5/':'data/';
  try {
    const response=await fetch(`${dataRoot}manifest.json`);if(!response.ok)throw new Error('Replay manifest could not be loaded');manifest=await response.json();
    $('#rules-note').textContent=manifest.rules;
    $('#selection-note').textContent=manifest.selection;
    $('#objective-note').textContent=manifest.diagnostic_note;
    const cutoffs=manifest.episodes.filter(e=>e.category==='cutoff').length;
    $('#collection-count').textContent=`${cutoffs} cutoffs · ${manifest.episodes.length-cutoffs} successes · no training in this viewer`;
    const a=manifest.benchmark;
    $('#reliability-note').hidden=!a;
    if(a) $('#reliability-note').textContent=`${Object.values(manifest.checks).every(Boolean)?'Reliability screen passed.':'Reliability target NOT met.'} Worst swap-cost seed: ${(a.swap_cost.truncation_rate.max*100).toFixed(2)}% cutoff (target: every seed <1%). Frozen educational candidate: ${manifest.delivery_model}, selected on validation before test. Residual failures are included below.`;
    $('#benchmark-note').textContent=a?`Fresh test · mean across 3 training seeds · Unshaped: ${(a.unshaped.success_rate.mean*100).toFixed(2)}% 10-win / ${(a.unshaped.truncation_rate.mean*100).toFixed(2)}% cutoff. Swap −0.005: ${(a.swap_cost.success_rate.mean*100).toFixed(2)}% 10-win / ${(a.swap_cost.truncation_rate.mean*100).toFixed(2)}% cutoff. 3,000 test episodes per model; selected replays below are illustrative.`:'Historical Round 3: old Fish rules. These selected episodes are not a new benchmark.';
    const left=manifest.episodes.find(e=>e.id===manifest.default_ids?.[0]) || manifest.episodes.find(e=>e.category==='cutoff') || manifest.episodes[0];
    const right=manifest.episodes.find(e=>e.id===manifest.default_ids?.[1]) || manifest.episodes.find(e=>e.category==='success'&&e.policy===left.policy) || manifest.episodes.find(e=>e.category==='success') || left;
    lanes.push(new Lane(0,left));lanes.push(new Lane(1,right));
  } catch(error) {const el=$('#global-error');el.hidden=false;el.textContent=error.message;}
  finally {$('#collection').disabled=false;}
}

async function init() {
  await loadCollection($('#collection').value);
  $('#collection').addEventListener('change',event=>loadCollection(event.target.value));
  $('#play-both').addEventListener('click',()=>lanes.forEach(l=>l.play()));
  $('#pause-both').addEventListener('click',()=>lanes.forEach(l=>l.pause()));
  $('#start-both').addEventListener('click',()=>lanes.forEach(l=>l.replay&&l.jump(0)));
  $('#ending-both').addEventListener('click',()=>lanes.forEach(l=>l.ending()));
  $('#speed').addEventListener('change',()=>lanes.forEach(l=>{if(l.timer)l.play();}));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)lanes.forEach(l=>l.pause());});
}
init().catch(error=>{const el=$('#global-error');el.hidden=false;el.textContent=error.message;});
