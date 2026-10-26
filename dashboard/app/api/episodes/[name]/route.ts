import {lstat, readFile, realpath} from 'node:fs/promises';
import {isAbsolute, join, relative} from 'node:path';

const safePart = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const headers = {'Cache-Control': 'no-store'};

export async function GET(_request: Request, {params}: {params: Promise<{name: string}>}) {
  const {name} = await params;
  let path: string;
  if (name.startsWith('rlcr~') && name.endsWith('.jsonl')) {
    const encoded = name.slice(5, -6);
    if (!/^[A-Za-z0-9_-]+$/.test(encoded)) return new Response('Bad name', {status: 400, headers});
    path = Buffer.from(encoded, 'base64url').toString('utf8');
    // Canonical IDs represent exactly <group>/episodes/<file>.jsonl.
    const parts = path.split('/');
    if (Buffer.from(path).toString('base64url') !== encoded || parts.length !== 3
        || !safePart.test(parts[0]) || parts[1] !== 'episodes'
        || !safePart.test(parts[2]) || !parts[2].endsWith('.jsonl')) {
      return new Response('Bad name', {status: 400, headers});
    }
    path = join('rlcr', path);
  } else {
    if (!safePart.test(name) || !name.endsWith('.jsonl')) return new Response('Bad name', {status: 400, headers});
    path = name;
  }
  try {
    const root = await realpath(join(process.cwd(), '..', 'runs'));
    const candidate = join(root, path);
    const resolved = await realpath(candidate);
    const withinRoot = relative(root, resolved);
    if (withinRoot.startsWith('..') || isAbsolute(withinRoot) || !(await lstat(candidate)).isFile()) {
      return new Response('Not found', {status: 404, headers});
    }
    const data = await readFile(resolved, 'utf8');
    const lines = data.split('\n'), rows = [];
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
        || !['secret_hitler', 'liars_bar'].includes(first.game)) {
      return new Response('Not found', {status: 404, headers});
    }
    return Response.json(rows, {headers});
  } catch { return new Response('Not found', {status: 404, headers}); }
}
