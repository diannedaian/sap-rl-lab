export const icons = {ant:'🐜', cricket:'🦗', fish:'🐟', horse:'🐴', mosquito:'🦟', otter:'🦦', pig:'🐷', pigeon:'🐦', zombie_cricket:'🦗', bee:'🐝', apple:'🍎', honey:'🍯', bread_crumbs:'🍞'};
export const levelOf = p => p.experience >= 6 ? 3 : p.experience >= 3 ? 2 : 1;
export const objectiveReward = step => step.objective_reward ?? step.reward;
export const objectiveRemaining = step => step.objective_remaining_return ?? step.remaining_return;
export function fishAbility(level, catalogId) {
  const text = level === 3 ? 'No level-up ability remains.' : `Level up: give two friends +${level}/+${level}, using the departing level.`;
  return text + (catalogId === 'turtle-v0.46-tier1' ? ' Historical Round 3 bug: merges used the destination-level entry instead.' : ' Corrected rules-v2.');
}
export function describeAction(label, state, pets = {}) {
  const [kind, suffix = ''] = label.split(':');
  const [source, target] = suffix.split(',').map(Number);
  const name = id => pets[id]?.name || id?.replaceAll('_', ' ') || 'item';
  const item = state.shop[source];
  const pet = state.team[source];
  const actions = {
    end_turn: 'End turn → battle', roll: 'Roll the shop',
    buy_pet: `Buy ${name(item?.item_id)} · shop ${source}`,
    buy_food: `Feed ${name(item?.item_id)} · shop ${source} → slot ${target}`,
    merge: `Merge ${name(item?.item_id)} · shop ${source} → slot ${target}`,
    sell: `Sell ${name(pet?.spec_id)} · slot ${source}`,
    swap_adjacent: `Swap slots ${source} ↔ ${target}`,
    toggle_freeze: `${item?.frozen ? 'Unfreeze' : 'Freeze'} shop ${source}`,
  };
  return actions[kind] || label;
}
export function makeFrames(replay) {
  const frames = [];
  replay.steps.forEach((step, index) => {
    frames.push({kind:'decision', index});
    step.battle.forEach((battle, battleIndex) => frames.push({kind:'battle', index, battleIndex}));
  });
  frames.push({kind:'terminal', index:replay.steps.length - 1});
  return frames;
}
export function findLoop(replay) {
  for (let i = 2; i < replay.steps.length; i++) {
    const a = replay.steps[i];
    if (!/^(swap_adjacent|toggle_freeze)/.test(a.action)) continue;
    if ([i-1, i-2].every(j => replay.steps[j].action === a.action && replay.steps[j].state.turn === a.state.turn)) return i-2;
  }
  return null;
}
export function cutoffMath(step, gamma) {
  const bootstrap = step.truncated && !step.terminated ? gamma * step.next_value : 0;
  return {bootstrap, storedReward:objectiveReward(step) + bootstrap};
}
export function verifyReplayShape(replay) {
  if (!replay.verified || !Array.isArray(replay.steps) || !replay.steps.length) throw new Error('Replay has no verified trajectory');
  for (const s of replay.steps) {
    if (!Array.isArray(s.battle) || (s.action === 'end_turn' && !s.battle.length)) throw new Error('Missing battle frames');
    if (!s.legal_actions.some(a => a.id === s.action_id)) throw new Error('Chosen action is not legal');
    if (s.legal_actions.some(a => !Number.isFinite(a.probability) || a.probability < 0 || a.probability > 1)) throw new Error('Invalid probability');
    const sum = s.legal_actions.reduce((n,a) => n+a.probability, 0);
    if (Math.abs(sum-1) > 0.00001) throw new Error('Probabilities do not sum to one');
    if (Math.abs(cutoffMath(s,replay.gamma).storedReward-s.rollout_reward) > 0.00001) throw new Error('Bootstrap mismatch');
    if (![s.value, s.next_value, s.reward, s.rollout_reward, s.remaining_return].every(Number.isFinite)) throw new Error('Non-finite reward/value');
    if (![objectiveReward(s), objectiveRemaining(s)].every(Number.isFinite)) throw new Error('Non-finite objective reward');
    if (replay.swap_cost !== undefined && Math.abs(objectiveReward(s) - (s.reward - (s.action.startsWith('swap_adjacent:') ? replay.swap_cost : 0))) > 1e-8) throw new Error('Swap objective mismatch');
  }
}
