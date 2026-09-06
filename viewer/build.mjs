import { cp, mkdir, readFile } from 'node:fs/promises';
import { verifyReplayShape } from './model.js';
let total = 0;
for (const folder of ['./data/', './data/round5/']) {
  const root = new URL(folder, import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('manifest.json', root)));
  if (!manifest.episodes.length) throw new Error('Empty replay collection');
  for (const entry of manifest.episodes) {
    const replay = JSON.parse(await readFile(new URL(entry.file, root)));
    verifyReplayShape(replay);
  }
  total += manifest.episodes.length;
}
await mkdir(new URL('./dist', import.meta.url), { recursive: true });
for (const name of ['index.html', 'styles.css', 'app.js', 'model.js', 'data']) {
  await cp(new URL(`./${name}`, import.meta.url), new URL(`./dist/${name}`, import.meta.url), { recursive: true });
}
console.log(`Built replay viewer with ${total} verified episodes in two collections`);
