import {readdir,readFile,stat} from 'node:fs/promises';
import {join} from 'node:path';
const runs=join(process.cwd(),'..','runs');
export async function GET(){
  let files:string[]=[];try{files=(await readdir(runs)).filter(f=>f.endsWith('.jsonl'));}catch{return Response.json([])}
  const episodes=await Promise.all(files.map(async file=>{try{const path=join(runs,file),lines=(await readFile(path,'utf8')).trim().split('\n');const first=JSON.parse(lines[0]),last=JSON.parse(lines.at(-1)!);return {file,game:first.game,condition:first.condition,seed:first.seed,model:first.model,events:lines.length,winner:last.kind==='episode_end'?last.winner:null,completed:last.kind==='episode_end'&&!!last.completed,validForAnalysis:!!last.valid_for_analysis,invalidActions:last.invalid_actions??0,mtime:(await stat(path)).mtimeMs};}catch{return null}}));
  return Response.json(episodes.filter(Boolean).sort((a,b)=>b!.mtime-a!.mtime),{headers:{'Cache-Control':'no-store'}});
}
