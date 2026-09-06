import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {verifyReplayShape, objectiveReward} from '../model.js';

const root = new URL('../data/round5/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', root)));
test('new collection covers six models and default pair shares initial conditions', () => {
  assert.equal(new Set(manifest.episodes.map(e=>e.policy)).size,6);
  assert.ok(manifest.episodes.some(e=>e.policy.startsWith('swap_cost')&&e.category==='success'));
  assert.equal(manifest.benchmark.swap_cost.success_rate.per_seed.length,3);
  if (manifest.default_ids) {
    const [a,b]=manifest.default_ids.map(id=>manifest.episodes.find(e=>e.id===id));
    assert.equal(a.seed,b.seed);assert.equal(a.family,b.family);
    assert.equal(a.category,'cutoff');assert.equal(b.category,'success');
  }
});
for (const entry of manifest.episodes) {
  const replay=JSON.parse(await readFile(new URL(entry.file,root)));
  test(`corrected rules, raw vs shaped returns, exact continuity: ${entry.id}`, () => {
    verifyReplayShape(replay);
    assert.equal(replay.catalog_id,'turtle-v0.46-tier1-rules-v2');
    assert.equal(replay.swap_cost,entry.policy.startsWith('swap_cost')?.005:0);
    assert.equal(replay.steps.length,entry.actions);
    let raw=0,shaped=0;
    for(const step of replay.steps.toReversed()) {
      raw+=step.reward;shaped+=objectiveReward(step);
      assert.ok(Math.abs(raw-step.remaining_return)<1e-8);
      assert.ok(Math.abs(shaped-step.objective_remaining_return)<1e-8);
      assert.ok(Math.abs(step.td_target-(objectiveReward(step)+(step.terminated?0:replay.gamma*step.next_value)))<1e-8);
    }
    for(let i=1;i<replay.steps.length;i++) assert.deepEqual(replay.steps[i-1].after,replay.steps[i].state);
    assert.equal(replay.steps.at(-1).truncated,entry.category==='cutoff');
    assert.deepEqual(replay.steps.at(-1).after,replay.final_state);
  });
}
