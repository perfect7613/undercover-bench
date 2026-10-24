import {readFile} from 'node:fs/promises';
import {basename,join} from 'node:path';
export async function GET(_request:Request,{params}:{params:Promise<{name:string}>}){
  const {name}=await params;
  if(!name.endsWith('.jsonl')||basename(name)!==name)return new Response('Bad name',{status:400});
  try{const data=await readFile(join(process.cwd(),'..','runs',name),'utf8');return Response.json(data.trim().split('\n').map(line=>JSON.parse(line)),{headers:{'Cache-Control':'no-store'}})}catch{return new Response('Not found',{status:404})}
}
