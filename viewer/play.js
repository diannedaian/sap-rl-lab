import {icons as oldIcons,levelOf} from './model.js';
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icons={...oldIcons,beaver:'🦫',duck:'🦆',moth:'🦋',bluebird:'🐦',ladybug:'🐞',bat:'🦇',crab:'🦀',dodo:'🦤',dog:'🐕',elephant:'🐘',flamingo:'🦩',hedgehog:'🦔',peacock:'🦚',rat:'🐀',shrimp:'🦐',spider:'🕷️',swan:'🦢',badger:'🦡',blowfish:'🐡',camel:'🐪',giraffe:'🦒',kangaroo:'🦘',ox:'🐂',rabbit:'🐇',sheep:'🐑',snail:'🐌',turtle:'🐢',bison:'🦬',deer:'🦌',dolphin:'🐬',hippo:'🦛',parrot:'🦜',penguin:'🐧',rooster:'🐓',skunk:'🦨',squirrel:'🐿️',whale:'🐋',cow:'🐄',crocodile:'🐊',monkey:'🐒',rhino:'🦏',scorpion:'🦂',seal:'🦭',shark:'🦈',turkey:'🦃',boar:'🐗',cat:'🐈',dragon:'🐉',fly:'🪰',gorilla:'🦍',leopard:'🐆',mammoth:'🦣',snake:'🐍',tiger:'🐅',zombie_fly:'🪰',ram:'🐏',bus:'🚌',chick:'🐥',dirty_rat:'🐀',garlic:'🧄',meat_bone:'🍖',salad_bowl:'🥗',pear:'🍐',canned_food:'🥫',sushi:'🍣',melon:'🍈',mushroom:'🍄',steak:'🥩',pizza:'🍕',chocolate:'🍫',chili:'🌶️',milk:'🥛',cake:'🍰',bread:'🍞',sleeping_pill:'💊',cupcake:'🧁'};
let catalog,manifest,state,selection=null,busy=true,serial=0,frame=0,timer=null;
const name=id=>catalog?.pets.find(p=>p.id===id)?.name||catalog?.foods.find(p=>p.id===id)?.name||id?.replaceAll('_',' ')||'';
const randomSeed=()=>crypto.getRandomValues(new Uint32Array(1))[0];
const has=kind=>state?.legal.find(a=>a.kind===kind);
function request(payload){if(busy)return;busy=true;stop();$('#error').hidden=true;worker.postMessage({...payload,id:++serial,revision:state?.revision});if(state)render();}
function fail(message){busy=false;$('#error').hidden=false;$('#error').textContent=`${message} 若无法恢复，请重新加载页面；原训练模型不受影响。`;if(state)render();}
function card(p,index,zone,shop=false){
  if(!p)return `<div class="pet empty"><span class="slot">${index}</span>空位</div>`;
  const id=shop?p.item_id:p.spec_id,spec=catalog.pets.find(x=>x.id===id),food=shop&&p.kind==='food';
  const pet=shop?(p.pet||spec):p,selected=selection?.zone===zone&&selection.index===index;
  const atk=(pet?.attack||0)+(pet?.temporary_attack||0),hp=(pet?.health||0)+(pet?.temporary_health||0);
  const label=`${shop?'商店':'队伍'} ${index}：${name(id)}${!food?`，攻击 ${atk}，生命 ${hp}`:''}`;
  return `<button class="pet ${food?'food-card':''} ${p.frozen?'frozen':''} ${selected?'selected':''} ${!shop&&hp<=0?'dead':''}" data-zone="${zone}" data-index="${index}" aria-pressed="${selected}" aria-label="${esc(label)}"><span class="slot">${index}${shop&&p.choice_group!=null?' · 奖励':''}</span><span class="emoji" aria-hidden="true">${icons[id]||'🐾'}</span><span class="pet-name">${esc(name(id))}</span>${food?'<span class="stats">食物</span>':`<span class="stats"><span class="attack">⚔ ${atk}</span><span class="health">♥ ${hp}</span></span>`}<span class="pet-meta">${shop?`${p.cost} 金币${p.frozen?' · ❄':''}`:`Lv ${levelOf(p)} · ${p.experience}/6 XP`}</span>${pet?.perk?`<span class="temporary">${esc(name(pet.perk))}</span>`:''}</button>`;
}
function team(pets,zone){return Array.from({length:5},(_,i)=>card(pets[i],i,zone)).join('');}
function actionLabel(a){
  const item=state.human.shop[a.source],pet=state.human.team[a.source],target=state.human.team[a.target];
  return ({end_turn:'结束回合 → 战斗',roll:'刷新商店 · 1 金币',buy_pet:`购买 ${name(item?.item_id)} · ${item?.cost} 金币`,buy_food:`喂给 ${a.target} 号 ${name(target?.spec_id)}`,merge:`买入并合成 → ${a.target} 号 ${name(target?.spec_id)}`,sell:`卖出 ${name(pet?.spec_id)}`,swap_adjacent:`交换 ${a.source} ↔ ${a.target} 号位`,toggle_freeze:`${item?.frozen?'解冻':'冻结'} ${name(item?.item_id)}`,merge_team:`合成到 ${a.target} 号 ${name(target?.spec_id)}`})[a.kind]||a.kind;
}
const triggers={buy:'买入',sell:'卖出',level_up:'升级',start_battle:'战斗开始',faint:'阵亡',friend_summoned:'友方被召唤',start_turn:'回合开始',end_turn:'回合结束',hurt:'受伤',after_attack:'攻击后',friend_ahead_attacks:'前方友方攻击',friend_ahead_faints:'前方友方阵亡',friendly_ate_food:'友方吃食物',knock_out:'击杀',summoned:'被召唤',friend_faints:'友方阵亡',before_attack:'攻击前',friend_bought:'买入友方',shop_food:'使用商店食物'};
function ability(a){
  // Show the catalog's exact fields; avoid inventing descriptions for level-sensitive effects.
  const p=a.params||Object.fromEntries(Object.entries(a).filter(([k])=>!['trigger','effect','id','name','tier','cost','token','curriculum_only'].includes(k)));
  if(!a.effect)return '<p class="small-note">没有触发能力。</p>';
  const labels={attack:'攻击',health:'生命',count:'数量',damage:'伤害',gold:'金币',amount:'数量',perk:'附加效果',summon_id:'召唤',food_id:'生成食物',summon_attack:'召唤物攻击',summon_health:'召唤物生命',max_uses:'每回合次数上限',position:'位置',percent:'百分比',experience:'经验'};
  const effects={buff:'增加属性',buff_random_friend:'随机友方加属性',buff_subject:'触发对象加属性',damage_random_enemy:'伤害随机敌人',gain_gold:'获得金币',stock_food:'生成食物',summon:'召唤',buff_shop_pets:'商店宠物加属性',buff_self:'自身加属性',buff_position:'指定位置加属性',damage_lowest_enemy:'伤害生命最低的敌人',damage_behind:'伤害后方',damage_all:'伤害所有宠物',gain_melon:'获得 Melon',faint_pet:'使目标阵亡',buff_level_friends:'高等级友方加属性',buff_if_level_friend:'满足友方等级条件时加属性',copy_ahead_ability:'复制前方能力',swallow_ahead:'吞下前方宠物',buff_random_team:'随机队友加属性',buff_all_pets:'所有友方加属性',buff_front_pet:'最前方加属性',replace_milk:'生成 Milk',damage_last_enemy:'伤害最后方敌人',gain_peanut:'获得 Peanut',buff_future_shop:'未来商店宠物加属性',fly_summon:'召唤 Zombie Fly',gain_coconut:'获得 Coconut',repeat_ahead:'重复前方能力',gain_experience:'增加经验',set_perk:'附加效果'};
  return `<div class="ability-row"><b>${esc(triggers[a.trigger]||'食物效果')}</b>：${esc(effects[a.effect]||a.effect.replaceAll('_',' '))}${Object.entries(p).filter(([k])=>labels[k.replace('_by_level','')]).map(([k,v])=>`<div>${esc(labels[k.replace('_by_level','')])}${k.endsWith('_by_level')?'（Lv1/2/3）':''}：${esc(Array.isArray(v)?v.join(' / '):v)}</div>`).join('')}<details><summary>完整规则参数</summary><code>${esc(JSON.stringify({effect:a.effect,...p}))}</code></details></div>`;
}
function renderSelection(){
  if(!selection){$('#selection-title').textContent='选择一个宠物或食物';$('#selection-detail').innerHTML='<p>点击商店卡片购买；食物和合成会让你选择目标。点击队伍卡片可以卖出、换位或合成。</p>';$('#selection-actions').innerHTML='';return;}
  let p;
  if(selection.zone==='shop')p=state.human.shop[selection.index];
  else if(state.phase==='battle')p=state.battle.frames[frame].teams[selection.zone==='human'?0:1][selection.index];
  else p=selection.zone==='human'?state.human.team[selection.index]:state.ai.team[selection.index];
  if(!p){selection=null;renderSelection();return;}
  const id=p.item_id||p.spec_id,meta=catalog.pets.find(x=>x.id===id)||catalog.foods.find(x=>x.id===id);
  $('#selection-title').textContent=`${icons[id]||'🐾'} ${name(id)} · ${selection.index} 号位`;
  $('#selection-detail').innerHTML=`<p>Tier ${meta.tier}${p.choice_group!=null?' · 二选一升级奖励，买一个后另一个消失。':''}</p>${(meta.abilities||[meta]).map(ability).join('')}`;
  const actions=state.legal.filter(a=>selection.zone==='shop'?a.source===selection.index&&['buy_pet','buy_food','merge','toggle_freeze'].includes(a.kind):selection.zone==='human'&&(a.source===selection.index||a.kind==='swap_adjacent'&&a.target===selection.index)&&['sell','swap_adjacent','merge_team'].includes(a.kind));
  $('#selection-actions').innerHTML=actions.map(a=>`<button data-action="${a.id}" ${busy?'disabled':''} class="${['buy_pet','buy_food','merge'].includes(a.kind)?'primary':''}">${esc(actionLabel(a))}</button>`).join('')||'<p class="small-note">当前没有可执行操作。</p>';
}
function render(){
  const s=state.human,inBattle=state.phase==='battle';
  $('#match').hidden=false;$('#new-game').disabled=busy;
  $('#round-title').textContent=state.done?({human:'你赢下了这局！',ai:'模型赢下了这局',turn_limit:'达到 40 回合上限 · 本局结束'})[state.result]:`第 ${inBattle?state.battle.turn:s.turn} 回合 · ${inBattle?'战斗回放':'购物阶段'}`;
  $('#status').textContent=busy?'正在处理动作…':state.done?'可回看最后一场战斗，或重新开一局。':inBattle?'双方使用同一场战斗的结果。看完后进入下一轮。':'选择商店卡片开始购买。模型不会看到你的当前队伍。';
  $('#your-score').innerHTML=`<span>♥ ${s.lives}</span><span>🏆 ${s.wins}</span>${!inBattle?`<span class="gold">● ${s.gold} 金币</span>`:''}`;
  $('#ai-score').innerHTML=`<span>♥ ${state.ai.lives}</span><span>🏆 ${state.ai.wins}</span>`;
  $('#action-count').textContent=inBattle?'真实战斗状态':`${s.actions_this_turn}/30 动作`;
  $('#shop-area').hidden=inBattle;$('#battle-panel').hidden=!inBattle;
  $('#tier-label').textContent=`已解锁 Tier ${Math.min(6,1+Math.floor((s.turn-1)/2))}`;
  $('#shop').innerHTML=s.shop.map((p,i)=>p?card(p,i,'shop',true):'').join('');
  $('#roll').disabled=busy||!has('roll');$('#end-turn').disabled=busy||!has('end_turn');
  $('#budget-note').textContent=s.gold>=3?'还有金币可用；结束回合后不会保留。':s.actions_this_turn>=26?'即将达到动作上限，随后强制开战。':'';
  $('#next-round').hidden=state.done;$('#next-round').disabled=busy;
  $('#opponent-note').textContent=inBattle?'模型当前参战队伍 · 与你共同结算这一场战斗':'显示上一场揭晓的队伍；这一轮的购买暂不展示。';
  $('#your-team').innerHTML=team(inBattle?state.battle.frames[frame].teams[0]:s.team,'human');
  $('#ai-team').innerHTML=team(inBattle?state.battle.frames[frame].teams[1]:state.ai.team,'ai');
  if(inBattle){const b=state.battle;$('#battle-result').textContent=({win:'本轮你赢了',loss:'本轮模型获胜',draw:'本轮平局'})[b.outcome]+(b.human_forced?' · 你达到动作上限':'')+(b.ai_forced?' · 模型达到动作上限':'');$('#battle-slider').max=b.frames.length-1;$('#battle-slider').value=frame;$('#frame-count').textContent=`${frame+1} / ${b.frames.length}`;$('#battle-event').textContent=b.frames[frame].message;}
  $('#model-evidence').innerHTML=`本局 seed：${state.seed}<br>模型 SHA：<code>${esc(manifest.checkpoint_sha256)}</code>`;
  $('#ai-decisions').innerHTML=state.ai_decisions.map((d,i)=>`<div class="decision-row"><b>${i+1}. ${esc(d.label||d.action)}</b>价值估计 ${d.value.toFixed(3)}（不是胜率）<ol>${d.top.map(a=>`<li>${esc(a.label||a.action)} · ${(a.probability*100).toFixed(1)}%</li>`).join('')}</ol></div>`).join('')||'<p>第一场战斗后显示。</p>';
  renderSelection();
}
function stop(){if(timer)clearInterval(timer);timer=null;$('#battle-play').textContent='▶ 播放';}
function jump(n){stop();frame=Math.max(0,Math.min(n,state.battle.frames.length-1));render();}
function play(){if(timer){stop();return;}if(frame>=state.battle.frames.length-1)frame=0;timer=setInterval(()=>{if(frame>=state.battle.frames.length-1){stop();return;}frame++;render();},Number($('#battle-speed').value));$('#battle-play').textContent='Ⅱ 暂停';}
document.addEventListener('click',event=>{const button=event.target.closest('button');if(!button||!state)return;if(button.dataset.zone){selection={zone:button.dataset.zone,index:Number(button.dataset.index)};render();if(innerWidth<=800)$('#selection-title').scrollIntoView({behavior:'smooth',block:'center'});}if(button.dataset.action){const id=Number(button.dataset.action);selection=null;request({type:'action',action:id});}});
$('#roll').onclick=()=>{selection=null;request({type:'action',action:1});};
$('#end-turn').onclick=()=>{selection=null;request({type:'action',action:0});};
$('#next-round').onclick=()=>{selection=null;request({type:'next'});};
$('#new-game').onclick=()=>{if(!state||state.done||confirm('放弃当前对局，重新开始？')){selection=null;request({type:'start',seed:randomSeed()});}};
$('#battle-start').onclick=()=>jump(0);$('#battle-prev').onclick=()=>jump(frame-1);$('#battle-next').onclick=()=>jump(frame+1);$('#battle-play').onclick=play;$('#battle-slider').oninput=event=>jump(Number(event.target.value));$('#battle-speed').onchange=()=>{if(timer){stop();play();}};
const worker=new Worker('duel-worker.js',{type:'module'});
worker.onerror=()=>fail('浏览器运行环境启动失败。请确认网络能连接 jsDelivr，或使用较新的浏览器。');
worker.onmessage=({data})=>{
  if(data.status)$('#status').textContent=data.status;
  if(data.error){fail(data.error);return;}
  if(data.metadata){({catalog,manifest}=data.metadata);busy=false;request({type:'start',seed:randomSeed()});}
  if(data.state){const wasBattle=state?.phase==='battle';state=data.state;busy=false;if(state.phase==='battle'&&!wasBattle)frame=0;render();if(state.phase==='battle'&&!wasBattle)play();}
};
