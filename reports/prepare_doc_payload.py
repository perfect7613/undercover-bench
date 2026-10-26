import json,re
from pathlib import Path
s=Path('reports/undercover-bench-report.md').read_text()
abstract=s.split('## Abstract\n\n')[1].split('\n\n## 1.')[0]
body=s[s.index('## 1.'):]
paragraphs=[]
for line in body.splitlines():
 if not line.strip():continue
 level=len(line)-len(line.lstrip('#')) if line.startswith('#') else 0
 raw=line[level:].strip() if level else line
 links=[];pieces=[];pos=0;length=0
 for m in re.finditer(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',raw):
  before=raw[pos:m.start()];pieces.append(before);length+=len(before)
  label,url=m.groups();links.append({'start':length,'end':length+len(label),'url':url});pieces.append(label);length+=len(label);pos=m.end()
 pieces.append(raw[pos:]);text=''.join(pieces)
 paragraphs.append({'text':text,'level':level,'links':links})
result={'title':s.splitlines()[0][2:],'abstract':abstract,'paragraphs':paragraphs,'body':'\n'.join(p['text'] for p in paragraphs)+'\n'}
Path('reports/google-doc-payload.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
