import {lstat, readdir, readFile, realpath} from 'node:fs/promises';
import {isAbsolute, join, relative} from 'node:path';

const runs = join(process.cwd(), '..', 'runs');
const safePart = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const headers = {'Cache-Control': 'no-store'};

// Preserve root filenames. Nested IDs are one URL segment and cannot collide
// with root names, whose allowed characters do not include ~.
function nestedId(path: string) {
  return 'rlcr~' + Buffer.from(path).toString('base64url') + '.jsonl';
}

function records(data: string) {
  const lines = data.split('\n');
  const rows = [];
  for (let index = 0; index < lines.length; index++) {
    if (!lines[index].trim()) continue;
    try { rows.push(JSON.parse(lines[index])); }
    catch (error) {
      if (index === lines.length - 1 && !data.endsWith('\n')) break;
      throw error;
    }
  }
  const first = rows[0];
  if (!first || first.kind !== 'episode_start' || typeof first.episode_id !== 'string'
      || !['secret_hitler', 'liars_bar'].includes(first.game)) return null;
  return rows;
}

export async function GET() {
  let root: string;
  try { root = await realpath(runs); }
  catch { return Response.json([], {headers}); }
  const files: {file: string; path: string}[] = [];
  try {
    for (const entry of await readdir(root, {withFileTypes: true})) {
      if (entry.isFile() && safePart.test(entry.name) && entry.name.endsWith('.jsonl')) {
        files.push({file: entry.name, path: entry.name});
      }
    }
    const rlcr = join(root, 'rlcr');
    // Only the declared experiment layout is traversed, never character data,
    // checkpoints, reviews, symlinked groups or historical archive directories.
    if ((await lstat(rlcr)).isDirectory()) {
      for (const group of await readdir(rlcr, {withFileTypes: true})) {
        if (!group.isDirectory() || !safePart.test(group.name)) continue;
        const directory = join(rlcr, group.name, 'episodes');
        try {
          if (!(await lstat(directory)).isDirectory()) continue;
          for (const entry of await readdir(directory, {withFileTypes: true})) {
            if (!entry.isFile() || !safePart.test(entry.name) || !entry.name.endsWith('.jsonl')) continue;
            const path = [group.name, 'episodes', entry.name].join('/');
            files.push({file: nestedId(path), path: join('rlcr', path)});
          }
        } catch { /* A group can exist before its episodes directory. */ }
      }
    }
  } catch { /* A missing rlcr directory does not hide root episodes. */ }
  const episodes = await Promise.all(files.map(async ({file, path}) => {
    try {
      const resolved = await realpath(join(root, path));
      const withinRoot = relative(root, resolved);
      if (withinRoot.startsWith('..') || isAbsolute(withinRoot)) return null;
      const info = await lstat(join(root, path));
      if (!info.isFile()) return null;
      const rows = records(await readFile(resolved, 'utf8'));
      if (!rows) return null;
      const first = rows[0], last = rows.at(-1)!;
      const ended = last.kind === 'episode_end';
      return {file, game: first.game, condition: first.condition, seed: first.seed, model: first.model,
        seatModels: first.seat_models ?? null, auditor: first.undercover ?? null,
        auditorModel: first.undercover ? (first.protocol_metadata?.auditor_model ?? first.seat_models?.[first.undercover] ?? first.model) : null,
        auditorInitialization: first.auditor_initialization ?? null,
        purpose: first.protocol_metadata?.purpose ?? null,
        rlcrUpdates: first.protocol_metadata?.rlcr_updates ?? null,
        events: rows.length, winner: ended ? last.winner : null, completed: ended && !!last.completed,
        validForAnalysis: ended && !!last.valid_for_analysis, invalidActions: last.invalid_actions ?? 0,
        mtime: info.mtimeMs};
    } catch { return null; }
  }));
  return Response.json(episodes.filter(Boolean).sort((a, b) => b!.mtime - a!.mtime), {headers});
}
