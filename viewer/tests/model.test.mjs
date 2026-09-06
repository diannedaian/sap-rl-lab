import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {levelOf, describeAction, makeFrames, findLoop, cutoffMath, verifyReplayShape, objectiveRemaining, fishAbility} from '../model.js';

test('level boundaries are 3 and 6 experience', () => {
  assert.deepEqual([1,2,3,4,5,6].map(experience => levelOf({experience})), [1,1,2,2,2,3]);
});
test('action labels use the current item and freeze state', () => {
  const state={shop:[{item_id:'apple',frozen:true}],team:[{spec_id:'fish'}]};
  assert.equal(describeAction('toggle_freeze:0',state),'Unfreeze shop 0');
  assert.equal(describeAction('buy_food:0,0',state),'Feed apple · shop 0 → slot 0');
});
test('bootstrapping distinguishes true termination, truncation and ordinary moves', () => {
  const positive=cutoffMath({truncated:true,terminated:false,reward:-2,next_value:3},.9);
  assert.ok(Math.abs(positive.bootstrap-2.7)<1e-12);
  assert.ok(Math.abs(positive.storedReward-.7)<1e-12);
  for(const flags of [{truncated:false,terminated:true},{truncated:true,terminated:true},{truncated:false,terminated:false}]) {
    assert.deepEqual(cutoffMath({...flags,reward:-2,next_value:3},.9),{bootstrap:0,storedReward:-2});
  }
  assert.equal(cutoffMath({truncated:true,terminated:false,reward:-2,next_value:-3},1).storedReward,-5);
});
test('loop hint requires repeated reversible actions within a turn', () => {
  const steps=[1,1,1].map(turn=>({action:'swap_adjacent:0,1',state:{turn}}));
  assert.equal(findLoop({steps}),0);
  assert.equal(findLoop({steps:steps.map((s,i)=>({...s,state:{turn:i}}))}),null);
  assert.equal(findLoop({steps:steps.map(s=>({...s,action:'roll'}))}),null);
});
test('critic cutoff math uses the training objective, not raw game reward', () => {
  const step={truncated:true,terminated:false,reward:-1,objective_reward:-1.005,next_value:2};
  assert.ok(Math.abs(cutoffMath(step,1).storedReward-.995)<1e-12);
  assert.equal(objectiveRemaining({remaining_return:3,objective_remaining_return:2.995}),2.995);
  assert.equal(objectiveRemaining({remaining_return:3}),3);
});
test('Fish descriptions distinguish historical and corrected catalogs', () => {
  assert.match(fishAbility(1,'turtle-v0.46-tier1'),/Historical Round 3 bug/);
  assert.match(fishAbility(2,'turtle-v0.46-tier1-rules-v2'),/\+2\/\+2/);
  assert.doesNotMatch(fishAbility(2,'turtle-v0.46-tier1-rules-v2'),/Historical/);
  assert.match(fishAbility(3,'turtle-v0.46-tier1-rules-v2'),/No level-up ability/);
});

const root=new URL('../data/',import.meta.url);
const manifest=JSON.parse(await readFile(new URL('manifest.json',root)));
test('the fixed collection contains 12 cutoffs and 6 successes across all six models',()=>{
  assert.equal(manifest.episodes.length,18);
  assert.equal(manifest.episodes.filter(e=>e.category==='cutoff').length,12);
  assert.equal(new Set(manifest.episodes.map(e=>e.policy)).size,6);
});
for(const entry of manifest.episodes) {
  const replay=JSON.parse(await readFile(new URL(entry.file,root)));
  test(`verified trajectory, battle playback, and cutoff math: ${entry.id}`,()=>{
    verifyReplayShape(replay);
    assert.equal(replay.catalog_id,'turtle-v0.46-tier1');
    assert.equal(replay.steps.length,entry.actions);
    const frames=makeFrames(replay);
    assert.equal(frames.length,replay.steps.length+1+replay.steps.reduce((n,s)=>n+s.battle.length,0));
    assert.equal(frames[0].kind,'decision');
    assert.equal(frames.at(-1).kind,'terminal');
    const last=replay.steps.at(-1);
    assert.equal(last.truncated,entry.category==='cutoff');
    assert.equal(last.terminated,entry.category==='success');
    assert.equal(replay.final_state.wins,entry.wins);
    let remaining=0;
    for(const step of replay.steps.toReversed()) {
      remaining+=step.reward;
      assert.ok(Math.abs(remaining-step.remaining_return)<1e-8);
      if(step.action==='end_turn') assert.ok(step.battle.length>=2);
      for(const battle of step.battle) {
        assert.equal(battle.teams.length,2);
        assert.equal(typeof battle.event,'string');
      }
    }
    const corrupt=structuredClone(replay);
    corrupt.steps[0].legal_actions[0].probability=NaN;
    assert.throws(()=>verifyReplayShape(corrupt),/Invalid probability/);
    const endTurn=replay.steps.findIndex(s=>s.action==='end_turn');
    const missing=structuredClone(replay);
    missing.steps[endTurn].battle=[];
    assert.throws(()=>verifyReplayShape(missing),/Missing battle/);
  });
}
